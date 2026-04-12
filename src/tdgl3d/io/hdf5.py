"""HDF5 I/O for saving and loading simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.parameters import SimulationParameters
from ..core.solution import Solution
from ..mesh.indices import GridIndices, construct_indices


def save_solution(solution: Solution, filepath: str | Path) -> Path:
    """Save a Solution to an HDF5 file.

    Parameters
    ----------
    solution : Solution
    filepath : str or Path

    Returns
    -------
    Path
        The written file path.
    """
    import h5py

    filepath = Path(filepath)
    p = solution.params

    with h5py.File(filepath, "w") as f:
        f.create_dataset("times", data=solution.times)
        f.create_dataset("states_real", data=np.real(solution.states))
        f.create_dataset("states_imag", data=np.imag(solution.states))

        grp = f.create_group("params")
        for attr in ("Nx", "Ny", "Nz", "hx", "hy", "hz", "kappa",
                      "periodic_x", "periodic_y", "periodic_z"):
            grp.attrs[attr] = getattr(p, attr)

    return filepath


def load_solution(filepath: str | Path) -> Solution:
    """Load a Solution from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    Solution
    """
    import h5py

    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        times = f["times"][:]
        states = f["states_real"][:] + 1j * f["states_imag"][:]

        grp = f["params"]
        params = SimulationParameters(
            Nx=int(grp.attrs["Nx"]),
            Ny=int(grp.attrs["Ny"]),
            Nz=int(grp.attrs["Nz"]),
            hx=float(grp.attrs["hx"]),
            hy=float(grp.attrs["hy"]),
            hz=float(grp.attrs["hz"]),
            kappa=float(grp.attrs["kappa"]),
            periodic_x=bool(grp.attrs["periodic_x"]),
            periodic_y=bool(grp.attrs["periodic_y"]),
            periodic_z=bool(grp.attrs["periodic_z"]),
        )

    idx = construct_indices(params)
    return Solution(times=times, states=states, params=params, idx=idx)
