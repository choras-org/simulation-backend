"""Recompile sparrowpy's numba functions with fixes applied.

Both changes belong in sparrowpy itself. Remove this module once
sparrowpy ships them.

1. Nested parallel regions
   sparrowpy compiles ``integration.pt_solution`` and
   ``integration.load_stokes_entries`` with ``parallel=True``, but they are
   only called from inside ``prange`` loops of other ``parallel=True``
   functions (``_source2patch_energy_universal``,
   ``_patch2receiver_energy_universal`` and, via ``stokes_integration``,
   ``patch2patch_ff_universal``). That is a nested parallel launch. The
   default ``workqueue`` threading layer does not support it and aborts the
   process with "Concurrent access has been detected". Only the OpenMP or
   TBB threading layers handle it, and on macOS arm64 neither is available
   from pip. These two functions are compiled serially here; the outer
   functions keep their own parallelism.

2. Compilation cache
   sparrowpy compiles without ``cache=True``, so every new process spends
   about 30 s on JIT compilation. Here every function is compiled with
   ``cache=True``, so the machine code is stored on disk (``__pycache__``
   next to the sparrowpy sources, or numba's user cache directory if that
   is not writable) and reused by later processes.

numba resolves module globals when it compiles a function, so rebinding
the module attributes before the first call is enough for callers to pick
up the new dispatchers.
"""
import importlib
import pkgutil
import sys

import sparrowpy

try:
    import numba
    from numba.core.registry import CPUDispatcher
except ImportError:  # sparrowpy runs in pure python without numba
    numba = None

_SERIAL_FUNCTIONS = {
    ("sparrowpy.form_factor.integration", "pt_solution"),
    ("sparrowpy.form_factor.integration", "load_stokes_entries"),
}


def _sparrowpy_modules():
    """Import and return all sparrowpy modules."""
    for info in pkgutil.walk_packages(
            sparrowpy.__path__, prefix="sparrowpy."):
        importlib.import_module(info.name)
    return [m for name, m in sys.modules.items()
            if name == "sparrowpy" or name.startswith("sparrowpy.")]


def apply():
    """Rebind all sparrowpy numba dispatchers to fixed, cached versions."""
    if numba is None:
        return
    modules = _sparrowpy_modules()

    replacements = {}
    for module in modules:
        for name, func in vars(module).items():
            if not isinstance(func, CPUDispatcher) or id(func) in replacements:
                continue
            options = dict(func.targetoptions)
            options.pop("nopython", None)
            options["cache"] = True
            key = (func.py_func.__module__, func.py_func.__name__)
            if key in _SERIAL_FUNCTIONS:
                options["parallel"] = False
            replacements[id(func)] = numba.njit(**options)(func.py_func)

    # replace every reference, including re-exports in package __init__s
    for module in modules:
        for name, func in list(vars(module).items()):
            if id(func) in replacements and isinstance(func, CPUDispatcher):
                setattr(module, name, replacements[id(func)])


apply()
