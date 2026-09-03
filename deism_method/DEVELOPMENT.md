# DEISM method — development environment

Scope: developing and testing **`simulation-backend/deism_method`** in isolation.
(The full CHORAS app runs via Docker — see [`setup_instructions.md`](../../setup_instructions.md)
and `CHORAS_BUILD.sh` at the repo root. This doc is only for iterating on the
DEISM method locally without spinning up the whole stack.)

## Use the `choras` conda environment

Run DEISM development and tests in the **`choras`** conda environment (Python
3.11), not the miniconda `base` env. `base` has a stale `deism 2.0.2` directory in
site-packages that shadows a newer editable install, so `import deism.room_check`
fails there.

```bash
# One-time (if the env does not already exist):
conda create -n choras python=3.11

# Install just this method (its pyproject pulls deism + test deps):
conda run -n choras python -m pip install -e simulation-backend/deism_method
```

The method's [`pyproject.toml`](pyproject.toml) declares everything needed:
`deism==2.2.1.16` plus the `tests` extra (`pytest`, `pytest-cov`,
`coverage>=7.6.1`). No dependency from the `backend/` submodule is required.

## DEISM library version

- Pinned in [`pyproject.toml`](pyproject.toml): **`deism==2.2.1.16`** (matches the
  Docker image and current PyPI latest).
- The `choras` env was updated to `2.2.1.16` on 2026-09-03 (was 2.2.1.15 since
  2026-08-21):

  ```bash
  conda run -n choras python -m pip install "deism==2.2.1.16"
  ```

- Verify:

  ```bash
  conda run -n choras python -m pip show deism        # -> Version: 2.2.1.16
  conda run -n choras python -c "from deism.room_check import sync_room_geometry"
  ```

From `2.2.1.14` onward, impedance outside the supplied material bands holds the
nearest endpoint value instead of unconstrained cubic continuation. Catalog
materials that used to become non-passive above the highest band (negative
resistance, reflection gain greater than one) stay passive up to Nyquist. The
wrapper sends wall data as `"absorption"`.

From `2.2.1.16` onward `deism` reads room geometry with `meshio` instead of the
Gmsh Python bindings and accepts **only a pre-generated `.msh` file** (MSH 2 or
4.1). The wrapper therefore reads `msh_path` (not `geo_path`) from the input
JSON; the CHORAS backend writes both (absolute paths). A relative `msh_path`
is resolved against the directory of the JSON file, so a hand-written JSON next
to its mesh works from any working directory. The wall names come from the
`wall_centers` mapping that `deism.room_check.sync_room_geometry` writes into
the JSON (one named physical surface per wall), so the wrapper does not parse
the mesh itself and has no gmsh or meshio dependency; the Docker image no
longer installs gmsh. Mesh fixtures carry Gmsh's trailing spaces; `.gitattributes`
exempts `*.msh` from `git diff --check`. The test fixture
`tests/test_room_Deism.msh` must be meshed from `tests/test_room_Deism.geo` so
its physical surface names match the UUID keys in `test_input_Deism.json`.

## One gotcha: `coverage` must be ≥ 7.6.1

`deism==2.2.1.16` pulls **numba 0.66.0**, whose `numba.misc.coverage_support`
requires `coverage.types.Tracer`. That attribute was added in **coverage 7.6.1**
(verified by bisection: 7.6.0 lacks it, 7.6.1 has it — 7.4/7.5 do NOT work). If
the env has an older coverage, importing `deism` (hence collecting the tests)
fails with:

```
AttributeError: module 'coverage.types' has no attribute 'Tracer'
```

The `choras` env had a stale `coverage 7.2.7`; it is pinned to `7.6.1` (the floor):

```bash
conda run -n choras python -m pip install "coverage==7.6.1"
```

The `tests` extra already requires `coverage>=7.6.1`. The separate `backend/`
submodule still pins `coverage==7.2.7`; avoid installing its requirements into
this env.

## Running the tests

```bash
cd simulation-backend/deism_method
conda run -n choras python -m pytest -q
```

Expected: **13 passed** (the DEISM run uses the DEISM-ARG convex path,
`roomType: convex`).

Ruff is part of the `tests` extra; run `conda run -n choras python -m ruff check
deism_interface tests`. The 12 findings it reports today (module name, import
ordering, one bare `except`) all predate the deism 2.2.1.16 migration.

## Wall-ordering note (convex-only)

`get_deism_surface_order` returns the wall surfaces sorted by centroid, with no
dependency on Gmsh physical-tag declaration order. This is safe because CHORAS
rooms take DEISM's **convex/ARG path**, which re-matches each wall to its
absorption value by centroid proximity — so wall order does not affect the
physics as long as each surface's absorption stays paired with its own centroid
(guaranteed by keying both off the same surface UUID). The former hardcoded
`DEISM_WALL_TAG_ORDER = [2, 5, 4, 6, 1, 3]` (only meaningful for the axis-aligned
shoebox path) has been removed. If shoebox support is added later, restore an
axis-derived ordering for that path.
