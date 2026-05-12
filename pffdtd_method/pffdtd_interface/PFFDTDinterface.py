"""CHORAS interface for PFFDTD (Brian Hamilton, MIT).

Pure pass-through wrapper around the upstream PFFDTD wave-equation FDTD
solver. No ROM. No surrogate. One JSON in, one impulse response out.

GPU dispatch is *opportunistic*: if Hamilton's `fdtd_gpu` binary is on
PATH inside the container AND the runtime has GPU access (which only
happens when the CHORAS executor passes `--gpus` to docker run, which
the upstream executor does NOT do), we subprocess the c_cuda binary.
Otherwise we fall back to the pure-Python numba CPU engine. So this
method works on every host CHORAS supports; GPU is a bonus when wired.

Conforms to the CHORAS `SimulationMethod` ABC:
  - constructor takes the input JSON path (validated by the base class)
  - parameterless `run_simulation()`
  - inherited `save_results()` POSTs the mutated JSON back to the backend
"""
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .definition import SimulationMethod


# PFFDTD's python/ subtree is added to sys.path by the Dockerfile (via
# PYTHONPATH=/app/pffdtd_python) and via this explicit insert at import time
# (so unit tests and `python -c` calls into the package also work).
_PFFDTD_PY = Path("/app/pffdtd_python")
if _PFFDTD_PY.is_dir() and str(_PFFDTD_PY) not in sys.path:
    sys.path.insert(0, str(_PFFDTD_PY))


# Python 3.11+ tightened multiprocessing.shared_memory.SharedMemory.close()
# to raise BufferError when memoryview consumers still exist. PFFDTD's
# voxelizer holds a numpy view over the shared block while it copies data
# out, then closes the block -- on 3.10 this was permissive; on 3.11 it
# errors. The voxelization has already completed by the time close() runs,
# so swallowing this specific BufferError is safe and keeps sim_setup()
# from aborting before it writes vox_out.h5.
def _patch_shared_memory_close():
    try:
        from multiprocessing import shared_memory
    except ImportError:
        return
    _orig_close = shared_memory.SharedMemory.close

    def _tolerant_close(self):
        try:
            _orig_close(self)
        except BufferError:
            pass

    shared_memory.SharedMemory.close = _tolerant_close


_patch_shared_memory_close()


def _truthy(v, default=False):
    """Accept multiple truthy shapes the front-end may serialize."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "on", "1")
    return bool(v)


def _gpu_runtime_available():
    """Return True iff the GPU c_cuda binary is on PATH AND the container
    actually has GPU access (the binary will run if so, fail if not).
    Cheap check: just verify the binary exists. Real runtime failures are
    caught by the subprocess call and we fall back to CPU."""
    return shutil.which("fdtd_gpu") is not None


class PFFDTDMethod(SimulationMethod):
    """Raw PFFDTD CHORAS interface (no ROM)."""

    def __init__(self, input_json_path):
        super().__init__(input_json_path)

    def run_simulation(self):
        json_path = Path(self.input_json_path)
        with open(json_path, "r") as f:
            config = json.load(f)

        # ── Settings (with sensible defaults) ──
        s = config.get("simulationSettings", {})
        c0        = float(s.get("pffdtd_c0", 343.0))
        fmax      = float(s.get("pffdtd_fmax", 1000.0))
        ppw       = int(s.get("pffdtd_ppw", 6))
        ir_length = float(s.get("pffdtd_ir_length", 1.0))
        Tc        = float(s.get("pffdtd_temperature", 20.0))
        rh        = float(s.get("pffdtd_humidity", 50.0))
        use_gpu   = _truthy(s.get("pffdtd_use_gpu"), default=False)

        h = c0 / (fmax * ppw)

        # ── Geometry paths ──
        geo_path = config.get("geo_path", "")
        msh_path = config.get("msh_path", "")
        if geo_path and not os.path.isabs(geo_path):
            geo_path = str(json_path.parent / geo_path)
        if msh_path and not os.path.isabs(msh_path):
            msh_path = str(json_path.parent / msh_path)

        # ── Materials ──
        abs_coeffs = config.get("absorption_coefficients", {})

        # ── Source / receivers ──
        result_block = config["results"][0]
        src_xyz = np.array([
            float(result_block["sourceX"]),
            float(result_block["sourceY"]),
            float(result_block["sourceZ"]),
        ])
        receivers = [
            np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            for r in result_block["responses"]
        ]
        frequencies = result_block.get(
            "frequencies", [125, 250, 500, 1000, 2000])

        print(f"PFFDTD: c0={c0}, fmax={fmax}, h={h:.4f} m, IR={ir_length}s, "
              f"GPU={use_gpu} (binary {'present' if _gpu_runtime_available() else 'absent'})")
        print(f"PFFDTD: {len(abs_coeffs)} surfaces, "
              f"src={src_xyz.tolist()}, {len(receivers)} receivers")

        # Per-run work dir under the uploads folder
        work_dir = json_path.parent / f"pffdtd_run_{json_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Cheap progress signal: 5% during setup, then per-band updates
        self._write_progress(json_path, config, 5)

        self._build_pffdtd_setup(
            msh_path, geo_path, abs_coeffs, src_xyz, receivers,
            h, c0, Tc, rh, ir_length, fmax, work_dir,
        )
        self._write_progress(json_path, config, 15)

        ir, fs_out = self._run_fdtd(work_dir, use_gpu=use_gpu)
        self._write_progress(json_path, config, 90)

        per_band = self._compute_metrics(ir, fs_out, frequencies)

        # ── Write results (slide-10 schema) ──
        result_block["resultType"] = "PFFDTD"
        result_block["percentage"] = 100
        for i, resp in enumerate(result_block["responses"]):
            if i == 0:
                resp["receiverResults"] = ir.tolist()
                resp["receiverResultsUncorrected"] = ir.tolist()
                resp["parameters"] = per_band
            else:
                resp["receiverResults"] = []
                resp["receiverResultsUncorrected"] = []
                resp["parameters"] = {k: [None] * len(frequencies) for k in per_band}

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # DG-style auralization sidecar CSV
        csv_path = str(json_path).replace(".json", "_pressure.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "pressure"])
            t = np.arange(len(ir)) / fs_out
            for ti, pi in zip(t, ir):
                w.writerow([f"{ti:.6f}", f"{pi:.9f}"])
        print(f"PFFDTD: results written to {json_path}")

    # ── Progress (writes directly to JSON; UI polls this) ──

    @staticmethod
    def _write_progress(json_path, config, pct):
        try:
            config["results"][0]["percentage"] = int(pct)
            tmp = str(json_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(config, f, indent=4)
            os.replace(tmp, str(json_path))
        except Exception as e:
            print(f"PFFDTD: progress-write failed ({e})")

    # ── Setup ──

    def _build_pffdtd_setup(self, msh_path, geo_path, abs_coeffs, src_xyz,
                            receivers, h, c0, Tc, rh, ir_length, fmax,
                            work_dir):
        import gmsh

        gmsh.initialize()
        try:
            if msh_path and os.path.exists(msh_path):
                gmsh.open(msh_path)
            elif geo_path and os.path.exists(geo_path):
                gmsh.open(geo_path)
                gmsh.model.mesh.generate(2)
                gmsh.write(str(work_dir / "room.msh"))
            else:
                raise FileNotFoundError(
                    f"Neither msh_path nor geo_path exists: "
                    f"{msh_path!r}, {geo_path!r}")
            mats_hash = self._extract_gmsh_model(abs_coeffs)
        finally:
            gmsh.finalize()

        model_json = {
            "mats_hash": mats_hash,
            "sources":   [{"xyz": src_xyz.tolist()}],
            "receivers": [{"xyz": r.tolist()} for r in receivers],
        }
        model_json_path = work_dir / "model_export.json"
        with open(model_json_path, "w") as f:
            json.dump(model_json, f, indent=2)

        mat_dir = work_dir / "materials"
        mat_dir.mkdir(exist_ok=True)
        mat_files = self._fit_materials(abs_coeffs, mat_dir)

        from sim_setup import sim_setup

        mat_files_dict = {name: os.path.basename(path)
                          for name, path in mat_files.items()}
        # Force single-process voxelization. PFFDTD's voxelizer uses
        # multiprocessing.shared_memory; on Python 3.11+ the parent's
        # SharedMemory.close() raises BufferError when children leave
        # exported pointers alive. Single-proc avoids the issue (and is
        # already plenty fast for typical CHORAS room sizes).
        Nprocs = 1
        PPW = round(c0 / (fmax * h))

        sim_setup(
            model_json_file=str(model_json_path),
            mat_folder=str(mat_dir),
            mat_files_dict=mat_files_dict,
            source_num=1,
            insig_type="dhann30",
            diff_source=False,
            duration=ir_length,
            Tc=Tc, rh=rh,
            fcc_flag=False, PPW=PPW, fmax=fmax,
            save_folder=str(work_dir),
            compress=0, draw_vox=False, Nprocs=Nprocs,
        )

    def _extract_gmsh_model(self, abs_coeffs):
        import gmsh
        phys_groups = gmsh.model.getPhysicalGroups(2)
        mats_hash = {}
        default_colors = [
            [180, 180, 180], [200, 100, 100], [100, 200, 100],
            [100, 100, 200], [200, 200, 100], [200, 100, 200],
            [100, 200, 200], [150, 150, 150],
        ]
        for idx, (dim, phys_tag) in enumerate(phys_groups):
            name = gmsh.model.getPhysicalName(dim, phys_tag).split("$")[0]
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            all_nodes = {}
            pts, tris = [], []
            for entity_tag in entity_tags:
                elem_types, _, node_tags_list = gmsh.model.mesh.getElements(
                    dim, entity_tag)
                for et, ntags in zip(elem_types, node_tags_list):
                    if et != 2:
                        continue
                    for tri_nodes in ntags.reshape(-1, 3):
                        tri = []
                        for nid in tri_nodes:
                            if nid not in all_nodes:
                                coord, *_ = gmsh.model.mesh.getNode(int(nid))
                                all_nodes[nid] = len(pts)
                                pts.append(coord.tolist())
                            tri.append(all_nodes[nid])
                        tris.append(tri)
            if tris:
                mats_hash[name] = {
                    "pts": pts, "tris": tris,
                    "color": default_colors[idx % len(default_colors)],
                    "sides": [1] * len(tris),
                }
        return mats_hash

    def _fit_materials(self, abs_coeffs, mat_dir):
        from materials.adm_funcs import fit_to_Sabs_oct_11
        mat_files = {}
        for i, (name, alpha_str) in enumerate(abs_coeffs.items()):
            alphas_5 = [float(x.strip()) for x in alpha_str.split(",")]
            alphas_11 = self._expand_5_to_11_bands(alphas_5)
            alphas_11 = np.clip(alphas_11, 0.01, 0.99)
            mat_file = str(mat_dir / f"mat_{i:02d}.h5")
            fit_to_Sabs_oct_11(alphas_11, mat_file)
            mat_files[name] = mat_file
        return mat_files

    @staticmethod
    def _expand_5_to_11_bands(alphas_5):
        a = np.asarray(alphas_5, dtype=float)
        out = np.zeros(11)
        out[0:4] = a[0]; out[4] = a[1]; out[5] = a[2]
        out[6] = a[3]; out[7] = a[4]; out[8:11] = a[4]
        return out

    # ── FDTD execution ──

    def _run_fdtd(self, work_dir, use_gpu=False):
        """If use_gpu and the c_cuda binary is reachable, try GPU first.
        Fall back to the pure-Python numba CPU engine on any failure."""
        if use_gpu and _gpu_runtime_available():
            ir = self._run_fdtd_gpu(work_dir)
            if ir is not None:
                return self._postprocess(work_dir, ir_raw=ir)
            print("PFFDTD: GPU path unavailable, falling back to CPU numba")
        return self._run_fdtd_cpu(work_dir)

    def _run_fdtd_gpu(self, work_dir):
        binary = shutil.which("fdtd_gpu")
        if binary is None:
            return None
        try:
            subprocess.run(
                [binary], cwd=str(work_dir), check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=3600,
            )
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            err = getattr(e, "stderr", b"")
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            print(f"PFFDTD: GPU FDTD failed ({type(e).__name__}); "
                  f"stderr tail: {err[-400:]}")
            return None

        import h5py
        with h5py.File(work_dir / "sim_outs.h5", "r") as f:
            u_out = f["u_out"][...]
        # c_cuda's write_outputs already applies out_reorder; row 0 is the
        # first receiver in the canonical order.
        return np.asarray(u_out[0, :])

    def _run_fdtd_cpu(self, work_dir):
        from fdtd.sim_fdtd import SimEngine
        engine = SimEngine(str(work_dir), energy_on=False)
        engine.load_h5_data()
        engine.setup_mask()
        engine.allocate_mem()
        engine.set_coeffs()
        engine.checks()
        engine.run_all()
        engine.save_outputs()
        ir = engine.u_out[engine.out_reorder[0], :]
        return self._postprocess(work_dir, ir_raw=ir)

    def _postprocess(self, work_dir, ir_raw):
        """HP at 10 Hz + LP at 0.9 × Nyquist + resample to 48 kHz."""
        from scipy.signal import butter, sosfiltfilt
        import h5py

        with h5py.File(work_dir / "sim_consts.h5", "r") as f:
            Ts = float(f["Ts"][()])
            c = float(f["c"][()])
            h_grid = float(f["h"][()])
        fs_native = 1.0 / Ts
        fmax_grid = c / (2 * h_grid)

        ir = np.asarray(ir_raw, dtype=np.float64)
        sos_hp = butter(4, 10.0, btype="high", fs=fs_native, output="sos")
        ir = sosfiltfilt(sos_hp, ir)
        fcut = min(0.9 * fmax_grid, 0.45 * fs_native)
        sos_lp = butter(8, fcut, btype="low", fs=fs_native, output="sos")
        ir = sosfiltfilt(sos_lp, ir)

        try:
            import resampy
            ir = resampy.resample(ir, fs_native, 48000)
        except ImportError:
            from scipy.signal import resample
            n_out = int(round(len(ir) * 48000 / fs_native))
            ir = resample(ir, n_out)
        return ir, 48000.0

    # ── Metrics ──

    @staticmethod
    def _compute_metrics(ir, fs, frequencies):
        from scipy.signal import butter, sosfiltfilt
        ir = np.asarray(ir, dtype=np.float64)
        out = {k: [] for k in
               ("edt", "t20", "t30", "c80", "d50", "ts", "spl_t0_freq")}
        peak = float(np.max(np.abs(ir)) + 1e-30)

        for fc in frequencies:
            lo = fc / np.sqrt(2)
            hi = min(fc * np.sqrt(2), 0.49 * fs)
            if hi <= lo:
                for k in out:
                    out[k].append(None)
                continue
            sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
            ir_b = sosfiltfilt(sos, ir)
            edc, t = _edc(ir_b, fs)
            out["edt"].append(_decay_time(edc, t, -0.1, -10.1))
            out["t20"].append(_decay_time(edc, t, -5, -25))
            out["t30"].append(_decay_time(edc, t, -5, -35))
            out["c80"].append(_c80(ir_b, fs))
            out["d50"].append(_d50(ir_b, fs))
            out["ts"].append(_ts(ir_b, fs))
            spl = 20.0 * np.log10(
                np.sqrt(np.mean(ir_b ** 2) + 1e-30) /
                (peak + 1e-30) + 1e-30) + 94.0
            out["spl_t0_freq"].append(float(spl))
        return out


def _edc(ir, fs):
    e = np.cumsum(ir[::-1] ** 2)[::-1]
    e0 = e[0] + 1e-30
    return 10.0 * np.log10(e / e0 + 1e-30), np.arange(len(e)) / fs


def _decay_time(edc_db, t, db_start, db_end):
    try:
        i0 = np.argmax(edc_db < db_start)
        i1 = np.argmax(edc_db < db_end)
        if i1 <= i0 + 10:
            return None
        slope = np.polyfit(t[i0:i1], edc_db[i0:i1], 1)[0]
        if slope >= 0:
            return None
        return float(-60.0 / slope)
    except Exception:
        return None


def _c80(ir, fs):
    n80 = int(0.08 * fs)
    if n80 >= len(ir):
        return None
    early = np.sum(ir[:n80] ** 2)
    late = np.sum(ir[n80:] ** 2) + 1e-30
    return float(10.0 * np.log10(early / late))


def _d50(ir, fs):
    n50 = int(0.05 * fs)
    if n50 >= len(ir):
        return None
    early = np.sum(ir[:n50] ** 2)
    total = np.sum(ir ** 2) + 1e-30
    return float(early / total)


def _ts(ir, fs):
    t = np.arange(len(ir)) / fs
    ir2 = ir ** 2
    denom = np.sum(ir2) + 1e-30
    return float(np.sum(t * ir2) / denom)
