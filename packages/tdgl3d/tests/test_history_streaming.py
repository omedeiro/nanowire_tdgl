"""Streaming the history to disk must change where frames live, nothing else.

A frame is the whole state vector, so on a large mesh the history is what
limits how long a run can be — 960 MB per frame at 15 M nodes. ``stream_path``
moves the frames to an HDF5 file as they are produced. The tests below pin the
two properties that makes it worth having: the frames are the same ones, and
memory no longer grows with their number.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdgl3d
from tdgl3d.core.material import Layer, Trilayer
from tdgl3d.core.solution import Solution
from tdgl3d.solvers.history import HDF5History, MemoryHistory, make_history


@pytest.fixture
def device():
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=4, kappa=2.0)
    return tdgl3d.Device(
        params, applied_field=tdgl3d.AppliedField(Bz=0.3, t_on_fraction=1.0)
    )


RUN = dict(t_stop=0.4, dt=0.01, save_every=5, noise_seed=3, progress=False,
           log_metadata=False)


@pytest.mark.parametrize("method", ["euler", "trapezoidal"])
def test_streamed_frames_are_identical_to_in_memory(device, tmp_path, method):
    """Same frames, bit for bit — the only difference is where they are kept."""
    in_memory = tdgl3d.solve(device, method=method, **RUN)
    path = tmp_path / "run.h5"
    streamed = tdgl3d.solve(device, method=method, stream_path=path, **RUN)

    try:
        assert streamed.n_steps == in_memory.n_steps
        assert in_memory.n_steps > 2, "a one-frame run would not test anything"
        assert np.array_equal(streamed.times, in_memory.times)
        for step in range(in_memory.n_steps):
            assert np.array_equal(streamed.psi(step), in_memory.psi(step))
            assert np.array_equal(streamed.phi_x(step), in_memory.phi_x(step))
        # Read lazily from the file, not pulled into an array behind our backs.
        assert not isinstance(streamed.states, np.ndarray)
    finally:
        streamed.close()


def test_streamed_file_is_a_complete_solution(device, tmp_path):
    """The streamed file loads on its own — frames plus the grid to read them.

    Frames alone are not interpretable; without the parameters and index arrays
    the file would need a conversion pass before anything could use it.
    """
    path = tmp_path / "run.h5"
    streamed = tdgl3d.solve(device, method="euler", stream_path=path, **RUN)
    expected_psi = streamed.psi(-1)
    expected_steps = streamed.n_steps
    streamed.close()

    reloaded = Solution.load(str(path))
    assert reloaded.n_steps == expected_steps
    assert np.array_equal(reloaded.psi(-1), expected_psi)
    assert reloaded.params.Nx == device.params.Nx
    assert reloaded.params.Nz == device.params.Nz
    assert reloaded.idx.interior_to_full.size == device.params.n_interior
    # Non-vacuous: the state must actually be superconducting somewhere.
    assert np.abs(reloaded.psi(-1)).max() > 0.5


def test_save_to_the_streamed_path_is_a_no_op(device, tmp_path):
    """Saving a streamed run back over its own file must not truncate it."""
    path = tmp_path / "run.h5"
    streamed = tdgl3d.solve(device, method="euler", stream_path=path, **RUN)
    expected = streamed.psi(-1)
    streamed.save(str(path))
    streamed.close()

    reloaded = Solution.load(str(path))
    assert np.array_equal(reloaded.psi(-1), expected)


def test_save_elsewhere_copies_frame_by_frame(device, tmp_path):
    """A streamed run can still be written somewhere else, without a full read.

    ``save`` must not ask h5py for the whole dataset — that would pull a history
    that may not fit in memory back into it, which is the situation streaming
    exists to avoid.
    """
    path = tmp_path / "run.h5"
    copy = tmp_path / "copy.h5"
    streamed = tdgl3d.solve(device, method="euler", stream_path=path, **RUN)
    expected = [streamed.psi(k) for k in range(streamed.n_steps)]
    streamed.save(str(copy))
    streamed.close()

    reloaded = Solution.load(str(copy))
    assert reloaded.n_steps == len(expected)
    for step, frame in enumerate(expected):
        assert np.array_equal(reloaded.psi(step), frame)


def test_closing_releases_the_file(device, tmp_path):
    """After close, reading a frame raises rather than returning wrong data."""
    path = tmp_path / "run.h5"
    with tdgl3d.solve(device, method="euler", stream_path=path, **RUN) as streamed:
        assert np.isfinite(streamed.psi(-1)).all()
    with pytest.raises(Exception):
        streamed.psi(-1)


def test_streaming_holds_one_frame_regardless_of_run_length(tmp_path):
    """Memory must not grow with the number of saved frames.

    Checked on the store directly rather than through RSS, which a test cannot
    measure reliably: the in-memory store's buffer grows with the frame count
    and the streaming store's does not exist.
    """
    n_state = 4096
    frame = np.arange(n_state, dtype=np.complex128)

    memory = make_history(n_state, 2, None)
    assert isinstance(memory, MemoryHistory)
    streaming = make_history(n_state, 2, tmp_path / "s.h5")
    assert isinstance(streaming, HDF5History)

    try:
        for k in range(40):
            memory.append(float(k), frame + k)
            streaming.append(float(k), frame + k)

        _, mem_states = memory.finish()
        stream_times, stream_states = streaming.finish()

        assert mem_states.shape == (n_state, 40)
        assert stream_states.shape == (n_state, 40)
        assert mem_states.nbytes == n_state * 40 * 16
        # The streaming store keeps only the times list in memory; the frames
        # are on disk, so nothing here scales with 40 except that list.
        assert stream_times.shape == (40,)
        for k in (0, 17, 39):
            assert np.array_equal(stream_states[:, k], frame + k)
    finally:
        streaming.close()


def test_streaming_works_for_a_trilayer_with_a_hole(tmp_path):
    """The path a real device takes, not just a bare film."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0, is_superconductor=True),
        insulator=Layer(thickness_z=2, kappa=2.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0, is_superconductor=True),
    )
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=trilayer.Nz, kappa=2.0)
    dev = tdgl3d.Device(
        params, applied_field=tdgl3d.AppliedField(Bz=0.2, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    dev.add_hole([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)],
                 z_range=(0, trilayer.Nz))

    path = tmp_path / "run.h5"
    in_memory = tdgl3d.solve(dev, method="euler", **RUN)
    with tdgl3d.solve(dev, method="euler", stream_path=path, **RUN) as streamed:
        assert np.array_equal(streamed.psi(-1), in_memory.psi(-1))
        assert np.abs(streamed.psi(-1)).max() > 0.5
