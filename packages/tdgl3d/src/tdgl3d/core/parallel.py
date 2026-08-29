"""Thread pool for the right-hand side.

The evaluation is a stencil over a few hundred megabytes of state and is
limited by memory bandwidth, not by arithmetic — which is exactly the case a
thread pool helps, because each core comes with its own share of that
bandwidth.  NumPy releases the GIL for the ufuncs and fancy-index gathers the
stencil is made of, so plain Python threads scale here: measured 1.88x on two
cores, 2.90x on four, for the ψ-Laplacian kernel at 1.8 M nodes.

The work splits by *interior node*, not by output block: each thread computes
all four right-hand-side blocks for one contiguous run of nodes, so it reads
the same neighbourhood of the state throughout rather than sweeping the whole
grid four times.

Threads are off by default in small problems, where the pool costs more than it
saves; :func:`chunk_count` picks the number from the problem size.  Set
``TDGL3D_NUM_THREADS`` to override the pool size, or call :func:`set_num_threads`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

__all__ = ["get_num_threads", "set_num_threads", "chunk_count", "run_chunks"]

#: Below this many interior nodes a single thread wins: the pool's dispatch
#: overhead is tens of microseconds and the whole evaluation is not much more.
MIN_NODES_PER_THREAD = 40_000

_pool: Optional[ThreadPoolExecutor] = None
_pool_size: int = 0
_configured: Optional[int] = None


def _default_num_threads() -> int:
    from_env = os.environ.get("TDGL3D_NUM_THREADS")
    if from_env:
        try:
            return max(int(from_env), 1)
        except ValueError:
            pass
    return max((os.cpu_count() or 1), 1)


def get_num_threads() -> int:
    """Threads the right-hand side may use."""
    return _configured if _configured is not None else _default_num_threads()


def set_num_threads(n: int) -> None:
    """Set the pool size.  ``1`` runs everything on the calling thread."""
    global _configured, _pool, _pool_size
    if n < 1:
        raise ValueError("n must be >= 1")
    _configured = n
    if _pool is not None and _pool_size != n:
        _pool.shutdown(wait=False)
        _pool = None
        _pool_size = 0


def chunk_count(n_nodes: int) -> int:
    """How many chunks to split *n_nodes* into — 1 when threading would not pay."""
    return max(1, min(get_num_threads(), n_nodes // MIN_NODES_PER_THREAD))


def _get_pool(n: int) -> ThreadPoolExecutor:
    global _pool, _pool_size
    if _pool is None or _pool_size < n:
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="tdgl3d")
        _pool_size = n
    return _pool


def run_chunks(fn: Callable[[slice], None], n_nodes: int, n_chunks: int) -> None:
    """Call ``fn(slice)`` over *n_chunks* contiguous blocks covering *n_nodes*.

    Runs inline when there is only one chunk, so the single-threaded path never
    touches the pool.  Exceptions from the workers propagate.
    """
    if n_chunks <= 1:
        fn(slice(0, n_nodes))
        return

    step = -(-n_nodes // n_chunks)  # ceiling, so the last chunk is the short one
    bounds = [slice(lo, min(lo + step, n_nodes)) for lo in range(0, n_nodes, step)]
    if len(bounds) == 1:
        fn(bounds[0])
        return

    # list() forces every future, so a worker exception is re-raised here.
    list(_get_pool(len(bounds)).map(fn, bounds))
