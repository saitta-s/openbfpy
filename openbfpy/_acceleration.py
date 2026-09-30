"""Optional compilation of array kernels; no fast-math or changed equations."""

import os


if os.environ.get("OPENBFPY_DISABLE_JIT") == "1":
    njit = None
else:
    try:
        from numba import njit
    except ImportError:
        njit = None

ACCELERATED = njit is not None


def kernel(function):
    return njit(cache=True, fastmath=False)(function) if ACCELERATED else function
