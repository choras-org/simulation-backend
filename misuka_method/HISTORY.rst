=======
History
=======

Unreleased
----------

Added
^^^^^

- Atmospheric rendering: speed of sound from the propagation medium or a user value, and optional air attenuation (PR #34)
- Maximum energy loss setting, 100 dB by default (PR #34)

Changed
^^^^^^^

- Build misuka from the tag ``choras-v1.0.1`` instead of the ``lgpl`` branch (PR #34)

Fixed
^^^^^

- Compute T20, T30 and EDT from the normalized EDC, and all parameters from the direct sound (PR #34)


0.1.1 (2026-09-03)
------------------

Fixed
^^^^^

- On Docker build: avoid GitHub clone rate limits by sequentially retrying misuka fetches including submodules (PR #37)


0.1.0 (2026-09-01)
------------------

Initial release.
