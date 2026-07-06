# DEISM–CHORAS Coupling Issues

Review of `deism_method/` against other CHORAS simulation backends (DE, DG, pyroomacoustics), CHORAS JSON conventions, and the DEISM library API.

**Reviewed files:** `deism_interface/DEISMinterface.py`, `example_settings/deism_setting.json`, `methods-config.json`, test fixtures, DEISM `room_check` helpers.

---

## Summary

| Severity | Count | Note |
|----------|-------|------|
| Critical | 0 | original 3 re-triaged after verification (see below) |
| High     | 5 | +1: #1 downgraded from Critical to High |
| Medium   | 6 | |
| Low      | 3 | |
| Resolved | 2 | #2 (wall order) fixed; #3 (CI) addressed |

Several integration paths are implemented correctly (settings mapping, room rotation override, progress reporting, parameter conflict diagnostics).

**Update after verification:** the three original "Critical" items were re-triaged.
#2 (hardcoded wall-tag order) is **resolved** — replaced with centroid-based
ordering plus a regression test, and its risk was in any case scoped to DEISM's
shoebox path, which CHORAS does not use. #3 (CI) is **addressed** with a new
workflow and a discovered `coverage>=7.6.1` floor. #1 was **corrected and
downgraded to High** — `MeasurementRoom.geo` is convex, not the claimed non-convex
L-shape, so it is a defensive-validation gap rather than silent-wrong-physics. The
remaining top risk is **#1: no explicit room-shape guard for future non-convex
geometry**.

---

## Critical (re-triaged after verification)

### 1. No explicit room-shape validation before the solve — **now High**

> **Correction (verified):** The original report claimed `MeasurementRoom.geo` is
> "the standard CHORAS **L-shaped** room" and geometrically non-convex. That is
> **false** — `MeasurementRoom.geo` is 8 points / 6 surfaces forming a **convex**
> quadrilateral prism, and the test room is a convex slanted-ceiling box. Neither
> shipped room is non-convex, so there is currently no fixture that produces
> "silent wrong physics." This is therefore a **defensive-validation gap (High)**,
> not a Critical live-data bug.

**Location:** `DEISMinterface.py` — `get_room_geometry(geo_file=geo_path)` → `create_deism_instance("RIR", room)`

**Problem:** The interface passes `room` straight from `get_room_geometry` into
DEISM and never validates the classification itself. It does not use
`is_shoebox_corners` from `collect_room_geometry_data`, and there is no guard that
would reject a genuinely non-convex geometry (e.g. a user-supplied L-shape) before
the DEISM-ARG solve — which assumes a convex polyhedron.

**Impact:** A future non-convex CHORAS room could be solved with DEISM-ARG and
produce incorrect results without raising an error. Not triggered by any shipped
geometry today.

**Recommendation:**

- Validate room shape explicitly before the solve.
- Use `"convex"` only when geometry is truly convex.
- Fail with a clear error for unsupported non-convex rooms (e.g. L-shapes).

---

### 2. Hardcoded wall-tag order — **RESOLVED**

> **Status: fixed.** `DEISM_WALL_TAG_ORDER = [2, 5, 4, 6, 1, 3]` has been removed;
> `get_deism_surface_order` now sorts the 6 surfaces by centroid (see
> `DEISMinterface.py`) and no longer depends on Gmsh tag/declaration order.
> Regression test: `tests/test_wall_order.py` proves the order is invariant under
> permuted declaration order and rejects rooms without exactly 6 surfaces.

> **Scope correction (verified against the DEISM library):** The
> "silently permuted boundary conditions" risk was real **only in DEISM's shoebox
> path** (`deism.core_deism`), where reflection coefficients `Z_S` are indexed
> positionally as `[x1, x2, y1, y2, z1, z2]`. CHORAS rooms take the **convex/ARG
> path** (`deism.core_deism_arg`), which re-matches each wall to its absorption by
> **centroid proximity** — so wall order never affected the physics for CHORAS
> default rooms as long as `absorption[i]` stayed paired with `wall_center[i]`
> (which the interface guarantees by keying both off the same surface UUID).

**Original problem (for reference):** Gmsh physical tags were mapped to DEISM wall
order `[x1, x2, y1, y2, z1, z2]` via a fixed `[2, 5, 4, 6, 1, 3]` list, validated
at runtime only for tag *presence*. A change in `.geo` surface declaration order
would have permuted the mapping. This is no longer possible with the centroid-based
ordering.

**Note:** If DEISM shoebox support is added later, the shoebox path *does* require a
correct axis ordering — derive it from `wall_centers` vs the bounding-box min/max
per axis, not from tag integers.

---

### 3. No CI workflow for DEISM — **ADDRESSED**

> **Status: workflow added.** `.github/workflows/deism.yml` mirrors
> `pyroomacoustics.yml`: a `uv run --extra tests pytest` matrix (Python
> 3.11–3.13) plus an amd64 Docker build. Local dev guidance is now in
> `deism_method/DEVELOPMENT.md`.

**Location:** `.github/workflows/` — previously no DEISM workflow (pyroomacoustics has `pyroomacoustics.yml`)

**Problem:** The DEISM integration was not built or tested in CI. Two concrete
environment hazards were confirmed:

- **Shadowed install:** a miniconda `base` env had a stale `deism 2.0.2` directory
  physically shadowing a newer editable install, so `import deism.room_check`
  failed there. Dev/test must run in the `choras` conda env (or via `uv` /
  Docker), not `base`.
- **Transitive coverage floor:** `deism==2.2.1.13` pulls `numba 0.66.0`, whose
  `coverage_support` requires `coverage.types.Tracer`, added in **`coverage
  7.6.1`** (bisected: 7.6.0 lacks it; 7.4/7.5 do not work). Older `coverage`
  breaks `import deism` at test collection with
  `AttributeError: module 'coverage.types' has no attribute 'Tracer'`. The
  `tests` extra in `pyproject.toml` now pins `coverage>=7.6.1`.

**Impact:** Regressions in coupling logic could reach production undetected.

**Recommendation (done):** Workflow added; `deism==2.2.1.13` stays pinned in
`pyproject.toml` and `coverage>=7.6.1` is enforced via the `tests` extra.

---

## High

### 4. Only one source / one receiver supported

**Location:** `DEISMinterface.py` — all reads/writes use `results[0]` and `responses[0]`

**Problem:** CHORAS JSON supports multiple `results[]` entries and multiple `responses[]` per source. The interface only processes the first source and first receiver.

**Impact:** Multi-point simulations silently ignore extra sources and receivers.

**Recommendation:** Loop over all source/receiver pairs, or document the limitation in `methods-config.json` and frontend settings.

---

### 5. No room-acoustic parameters exported

**Location:** `DEISMinterface.py` — `parameters` block left empty in output JSON

**Problem:** Pyroomacoustics computes and writes `t20`, `t30`, `edt`, `d50`, `c80`, `ts`, `spl_t0_freq`. DEISM leaves these as empty arrays.

**Impact:** CHORAS UI may show blank acoustic metrics for DEISM runs if it expects the same fields as other RIR methods.

**Recommendation:** Compute parameters from the RIR (e.g. via pyrato/pyfar, as pyroomacoustics does) or document that DEISM does not provide them.

---

### 6. RIR peak normalization removes absolute level

**Location:** `DEISMinterface.py`:

```python
rir = deism.get_results()
rir = rir / np.max(np.abs(rir))
```

**Problem:** DG writes raw impulse responses; pyroomacoustics keeps physical scaling. Peak normalization removes absolute level information.

**Impact:** Inconsistent auralization levels vs other methods; `fs_auralization` may be correct but amplitude is not.

**Recommendation:** Remove normalization unless the CHORAS frontend explicitly expects unit peak. Align with DG/pyroomacoustics behavior.

---

### 7. `sourceOrientation` / `receiverOrientation` schema mismatch

**Location:** `example_settings/deism_setting.json` vs `tests/test_input_Deism.json`

**Problem:**

- `deism_setting.json` declares orientation as `"type": "string"` with default `"0, 0, 0"`.
- Test JSON uses arrays: `[0, 0, 0]`.

`parse_array_value` handles both at runtime, but the frontend schema and actual serialized payloads may diverge.

**Impact:** Orientation may fail to parse or be misinterpreted depending on how CHORAS serializes settings.

**Recommendation:** Align `deism_setting.json` type with the JSON format CHORAS actually sends (array vs comma-separated string).

---

## Medium

### 8. Test fixture inconsistency (`.geo` vs `.msh`)

**Location:** `tests/test_room_Deism.geo`, `tests/test_room_Deism.msh`

**Problem:**

- `.geo` uses UUID physical names (matches production CHORAS).
- `.msh` uses legacy names (`floor`, `wall1`, …).

The code always opens `geo_path`, so tests work, but the stale `.msh` is misleading.

**Recommendation:** Regenerate or remove the outdated `.msh`, or align its physical names with the `.geo` file.

---

### 9. `msh_path` is unused

**Location:** `DEISMinterface.py` — only `geo_path` is read

**Problem:** Other mesh-based methods use `msh_path` for mesh generation or simulation input. DEISM re-meshes from `.geo` inside `sync_room_geometry` and ignores `msh_path`.

**Impact:** If CHORAS expects the container to consume a pre-generated mesh, that path is ignored. Potential mismatch with backend assumptions.

---

### 10. Cancellation only checked once

**Location:** `DEISMinterface.py` — `check_should_cancel` at ~10% progress only

**Problem:** Long DEISM runs cannot be cancelled mid-solve.

**Recommendation:** Poll `should_cancel` between major pipeline stages (similar to progress updates at 25%, 45%, …).

---

### 11. Errors not persisted to JSON (except parameter conflicts)

**Location:** `DEISMinterface.py` — exception handler logs to stdout and re-raises

**Problem:** Only `parameterDiagnostics` from conflict checks is written to the result JSON. General failures appear only in container logs.

**Impact:** CHORAS backend/UI may not surface error messages unless it reads stderr.

**Recommendation:** On failure, write an `error` or `parameterDiagnostics.error` field to the JSON before re-raising.

---

### 12. Missing shared example input

**Location:** `common/` — no `exampleInput_DEISM.json`

**Problem:** Other methods provide `common/exampleInput_*.json` for docs and integration testing.

**Recommendation:** Add `common/exampleInput_DEISM.json` consistent with `test_input_Deism.json`.

---

### 13. Missing DEISM parameters in CHORAS settings

**Location:** `example_settings/deism_setting.json`

**Problem:** DEISM’s bundled yaml (`configSingleParam_ARG_RIR.yml`) includes parameters not exposed in CHORAS settings, e.g.:

- `overSamplingFactor`
- Alternative boundary inputs (impedance, reverberation time vs absorption only)

**Impact:** Users cannot tune these from the frontend without extending `deism_setting.json` and `DEISM_JSON_KEY_MAP`.

---

## Low

### 14. Misleading docstring on `create_vgroups_names`

**Location:** `DEISMinterface.py` — docstring says “mesh file” but the function opens `geo_path`

**Recommendation:** Update docstring to say “Gmsh geometry file (`.geo`)”.

---

### 15. No DEISM documentation in repo docs

**Location:** `docs/` — no DEISM-specific contributing or usage page

**Recommendation:** Add a section under contributing docs, matching DE/DG/pyroomacoustics.

---

### 16. `entryFile` in `methods-config.json` vs Docker CMD

**Location:** `methods-config.json` — `"entryFile": "DEISMinterface.py"`; Dockerfile — `CMD ["python", "-m", "deism_interface"]`

**Note:** Consistent with other methods (metadata `entryFile` vs module entrypoint). Not a bug, but may confuse readers of `methods-config.json`.

---

## What works correctly

| Area | Status |
|------|--------|
| `simulationSettings` → DEISM param mapping (`DEISM_JSON_KEY_MAP`) | Aligns with `deism_setting.json` |
| `ifRotateRoom` / `roomRotation` override | Prevents bundled yaml `[90,90,90]` from rotating CHORAS orientations |
| Absorption coefficient keys via UUID material names | Matches CHORAS geo export |
| `sync_room_geometry` + wall reordering | Consistent when tag order contract holds |
| Progress via `update_result_percentage` | Full JSON rewrite includes final RIR at 100% |
| `fs_auralization` from `sampleRate` | Suitable for auralization playback rate |
| `parameterDiagnostics` for conflict checks | Better error surfacing than some other methods |
| Docker / CLI pattern | Matches DE and pyroomacoustics |
| `use_real_stdio` / argv sanitization | Required for DEISM inside containers |

---

## Suggested fix priority

1. **Room-type logic** — Validate geometry; reject or document unsupported non-convex CHORAS rooms.
2. **Wall-order validation** — Verify tag-to-axis mapping, not just tag presence.
3. **CI workflow** — Docker build + pytest with pinned `deism==2.2.1.13`.
4. **Multi source/receiver** — Implement loops or document single-pair limitation.
5. **RIR normalization** — Remove or justify vs other methods.
6. **Orientation setting type** — Align `deism_setting.json` with CHORAS serialization.

---

## References

- Interface: `deism_interface/DEISMinterface.py`
- CHORAS settings schema: `example_settings/deism_setting.json`
- Method registration: `methods-config.json`
- Test input: `tests/test_input_Deism.json`
- Standard CHORAS room geo: `common/MeasurementRoom.geo`
- DEISM bundled defaults: `deism_interface/examples/configSingleParam_ARG_RIR.yml`
