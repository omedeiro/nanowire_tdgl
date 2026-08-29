"""Where saved frames go.

A frame is the whole state vector — ``4 × n_interior`` complex128, which is
116 MB on the 1.8 M-node grid and 960 MB at 15 M.  Keeping sixty of those in
memory is 58 GB, so on a large mesh the run length a machine can hold is set by
the history, not by the solver.

Two stores, same interface:

:class:`MemoryHistory`
    Frames in one preallocated block.  The obvious implementation — append
    copies to a list, then ``np.column_stack`` — holds every frame twice at the
    moment the run finishes, because the stack allocates the whole output
    before the list is released.  That doubling is what ends an otherwise
    successful overnight run.

:class:`HDF5History`
    Frames written to disk as they are produced, so memory holds one frame
    regardless of how many are saved.  The file it writes is a complete
    :class:`~tdgl3d.core.solution.Solution` artifact — same schema
    ``Solution.load`` reads — not a scratch file needing conversion, and the
    returned ``states`` is the dataset itself, sliced lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["MemoryHistory", "HDF5History", "make_history"]

#: Rows per HDF5 chunk.  One chunk per whole frame would be hundreds of
#: megabytes, far past any sensible chunk cache; this keeps chunks around 8 MB
#: while staying contiguous down a column, which is how frames are both written
#: and read back.
_CHUNK_ROWS = 1 << 19


class MemoryHistory:
    """Saved frames in one preallocated ``(n_state, n_saved)`` block."""

    __slots__ = ("_buf", "_n", "times")

    def __init__(self, n_state: int, capacity: int, dtype=np.complex128) -> None:
        self._buf = np.empty((n_state, max(capacity, 1)), dtype=dtype)
        self._n = 0
        self.times: list[float] = []

    def append(self, t: float, X: NDArray) -> None:
        if self._n == self._buf.shape[1]:
            self._buf = np.concatenate(
                [self._buf, np.empty_like(self._buf[:, : max(self._n, 1)])], axis=1
            )
        self._buf[:, self._n] = X
        self._n += 1
        self.times.append(t)

    def finish(self) -> tuple[NDArray, Any]:
        # A slice of the buffer is a view, so this returns the frames without
        # copying them when the capacity estimate was exact.
        return np.array(self.times), self._buf[:, : self._n]

    def close(self) -> None:
        """Nothing to release."""


class HDF5History:
    """Saved frames streamed to an HDF5 file as they are produced.

    The dataset is resizable along the frame axis and grown one frame at a
    time, so peak memory is one frame however long the run is.  Slicing it
    afterwards — ``states[:n, step]``, ``states[:, -1]`` — reads only what is
    asked for, which is what every :class:`~tdgl3d.core.solution.Solution`
    accessor does.

    The file is left open for reading; :meth:`close` releases it, and
    ``Solution.close()`` calls that.  Reading a frame after closing raises,
    rather than returning silently wrong data.
    """

    __slots__ = ("path", "_file", "_states", "_times", "_n", "times")

    def __init__(self, path: str | Path, n_state: int, dtype=np.complex128) -> None:
        import h5py

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._states = self._file.create_dataset(
            "states",
            shape=(n_state, 0),
            maxshape=(n_state, None),
            dtype=dtype,
            chunks=(min(n_state, _CHUNK_ROWS), 1),
        )
        self._times = self._file.create_dataset(
            "times", shape=(0,), maxshape=(None,), dtype=np.float64
        )
        self._n = 0
        self.times: list[float] = []

    def append(self, t: float, X: NDArray) -> None:
        self._n += 1
        self._states.resize(self._n, axis=1)
        self._states[:, self._n - 1] = X
        self._times.resize(self._n, axis=0)
        self._times[self._n - 1] = t
        self.times.append(t)

    def finish(self) -> tuple[NDArray, Any]:
        self._file.flush()
        return np.array(self.times), self._states

    def write_solution_metadata(self, params, idx, metadata: dict | None) -> None:
        """Write the rest of the ``Solution`` schema, so the file stands alone.

        Without this the streamed file would hold frames and nothing to
        interpret them with, and would need a conversion pass before anything
        could read it back.
        """
        from ..core.solution import write_solution_context

        write_solution_context(self._file, params, idx, metadata)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def make_history(
    n_state: int, capacity: int, stream_path: str | Path | None,
    dtype=np.complex128,
) -> MemoryHistory | HDF5History:
    """An :class:`HDF5History` when *stream_path* is given, else in memory."""
    if stream_path is None:
        return MemoryHistory(n_state, capacity, dtype=dtype)
    return HDF5History(stream_path, n_state, dtype=dtype)
