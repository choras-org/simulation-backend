=======
History
=======

Unreleased
----------

0.2.0 (2026-11-24)
------------------

Added
~~~~~

- Tests with repaired room geometries exported by the CHORAS frontend (#52)
- GitHub Actions workflow running the tests and building the Docker image (#52)
- Declare which geometry issues the method tolerates, for the backend's geometry compatibility check (#32)

Changed
~~~~~~~

- Compute patch visibility by ray casting with trimesh, embree where available, instead of sparrowpy's own check (#52)
- Limit numba's threads to the container's CPU limit (#52)
- Fill numba's compilation cache when building the Docker image (#52)
- Require Python 3.13 or later and sparrowpy 1.1.2 or later (#52)

Bug Fixes
~~~~~~~~~

- Fix wall outlines for non-rectangular surfaces (#50)
- Orient wall normals into the room for non-convex rooms, and keep up vectors perpendicular to tilted walls (#52)
- Normalize the energy decay curve before computing EDT, T20 and T30 (#53)
- Apply each surface's absorption to its own walls (#55)

0.1.0 (2026-09-01)
------------------

Initial release.
