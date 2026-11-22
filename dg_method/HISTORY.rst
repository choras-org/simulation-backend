=======
History
=======

Unreleased
----------

Bug Fixes
~~~~~~~~~

- Assign materials by physical group instead of surface order (#48)
- Update edg-acoustics to 6b40d10, which fixes the receiver interpolation:
  receivers at exactly z = 1 m returned wrong pressures, receivers with
  y + z = 0 crashed with "Singular matrix", and receivers outside the mesh
  silently used the last element instead of raising an error

0.1.0 (2026-09-01)
------------------

Initial release.
