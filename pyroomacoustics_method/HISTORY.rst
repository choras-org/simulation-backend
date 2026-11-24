=======
History
=======

Unreleased
----------

0.2.0 (2026-11-24)
------------------

Added
~~~~~

- Test configurations based on simulation-run data exported by the CHORAS backend (#43)
- Declare which geometry issues the method tolerates, for the backend's geometry compatibility check (#32)

Bug Fixes
~~~~~~~~~

- Import all surfaces of a physical group, not only the first (#42)
- Use the coarsest surface mesh, so large rooms stay tractable (#41)
- Require pyroomacoustics 0.10.1 or a later 0.10 release, which fixes NaN values in ray traced RIRs (#47)

0.1.0 (2026-09-01)
------------------

Initial release.