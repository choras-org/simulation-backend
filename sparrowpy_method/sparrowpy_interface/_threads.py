"""Match numba's thread count to the CPUs the process may actually use.

numba starts one thread per CPU of the host (``os.cpu_count()``). In a
container with a CPU limit (e.g. ``docker run --cpus=4``) that is too
many: the threads compete for the few CPUs the container gets and slow
each other down. This module lowers numba's thread count to the CPU limit
of the container (cgroup v2 or v1) and the CPU affinity of the process.

An explicitly set ``NUMBA_NUM_THREADS`` always wins.
"""
import math
import os
from pathlib import Path

import numba


def _cgroup_cpu_limit() -> float | None:
    """Return the CPU limit of the cgroup, or None if there is none."""
    # cgroup v2: "<quota> <period>" or "max <period>"
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota, period = cpu_max.read_text().split()
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass
    # cgroup v1: quota is -1 if unlimited
    try:
        quota_us = int(Path(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period_us = int(Path(
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota_us > 0 and period_us > 0:
            return quota_us / period_us
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """Return the number of CPUs this process may use."""
    n_cpus = os.process_cpu_count() or 1  # honours the CPU affinity
    limit = _cgroup_cpu_limit()
    if limit is not None:
        n_cpus = min(n_cpus, max(1, math.ceil(limit)))
    return n_cpus


def apply() -> None:
    """Limit numba's threads to the available CPUs."""
    if "NUMBA_NUM_THREADS" in os.environ:
        return
    # numba.config attributes are set at runtime, hence getattr
    max_threads = getattr(numba.config, "NUMBA_NUM_THREADS")
    n_threads = min(available_cpus(), max_threads)
    numba.set_num_threads(n_threads)


apply()
