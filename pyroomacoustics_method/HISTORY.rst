=======
History
=======

Unreleased
----------

Added
~~~~~

- Test configurations based on simulation-run data exported by the CHORAS backend.

Bug Fixes
~~~~~~~~~

- Use the coarsest surface mesh, so large rooms stay tractable (#41)
- Require pyroomacoustics 0.10.1 or a later 0.10 release, which fixes NaN values in ray traced RIRs (#47)

0.1.0 (2026-09-01)
------------------

Initial release.