=======
History
=======

Unreleased
----------

0.2.0 (2026-11-24)
------------------

Features
~~~~~~~~

- Treat CHORAS absorption coefficients as random-incidence values: they are
  averaged over the octave bands up to and including the band containing
  ``dg_freq_upper_limit`` and converted to the normal-incidence reflection
  coefficient of a locally reacting wall with the same random-incidence
  absorption (previously ``R = sqrt(1 - alpha)`` over all bands, which made
  DG walls too absorptive) (#51)
- Declare which geometry issues the method tolerates, for the backend's geometry compatibility check, and add a method-specific validation hook (#32)

Bug Fixes
~~~~~~~~~

- Assign materials by physical group instead of surface order (#48)
- Update edg-acoustics to 6b40d10, which fixes the receiver interpolation:
  receivers at exactly z = 1 m returned wrong pressures, receivers with
  y + z = 0 crashed with "Singular matrix", and receivers outside the mesh
  silently used the last element instead of raising an error (#49)

0.1.0 (2026-09-01)
------------------

Initial release.
