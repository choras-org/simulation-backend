=======
History
=======

Unreleased
----------

Features
~~~~~~~~

- Treat CHORAS absorption coefficients as random-incidence values: they are
  averaged over the octave bands up to and including the band containing
  ``dg_freq_upper_limit`` and converted to the normal-incidence reflection
  coefficient of a locally reacting wall with the same random-incidence
  absorption (previously ``R = sqrt(1 - alpha)`` over all bands, which made
  DG walls too absorptive)

Bug Fixes
~~~~~~~~~

- Assign materials by physical group instead of surface order (#48)

0.1.0 (2026-09-01)
------------------

Initial release.
