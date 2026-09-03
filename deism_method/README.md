# deism_interface

CHORAS coupling for the DEISM acoustic simulation method.

Copyright (c) 2026 Zeyu Xu and the CHORAS developers.

## License

This interface (`deism_interface`) is released under the MIT License, like the
other CHORAS simulation methods (see the `LICENSE` file at the root of the
CHORAS repository). It is a thin client of the public API of the `deism`
library and contains no code from it.

The `deism` library is **not** part of this repository. It is installed from
PyPI at build time and is distributed by Fraunhofer under the Fraunhofer
Software Copyright License, which permits non-commercial use for evaluation,
testing and academic research only. Any commercial use of the DEISM method
therefore requires a separate license from Fraunhofer IIS
(info@iis.fraunhofer.de). The MIT license of this interface does not extend to
`deism`.
