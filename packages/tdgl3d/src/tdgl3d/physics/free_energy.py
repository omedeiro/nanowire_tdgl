"""Discrete Ginzburg-Landau free energy in the link-variable representation.

The TDGL system solved by :mod:`tdgl3d.physics.rhs` is a gradient flow of the
Ginzburg-Landau functional

.. math::
    F[\\psi, A] = \\int \\Big[ -|\\psi|^2 + \\tfrac{1}{2}|\\psi|^4
                 + |(\\nabla - iA)\\psi|^2 + \\kappa^2 |\\nabla \\times A|^2 \\Big]\\,dV .

The discretisation below is the *exact* lattice counterpart of that functional
for the operators in :mod:`tdgl3d.operators.sparse_operators`: the covariant
gradient uses the same Peierls link factors as ``construct_LPSI_*`` and the
magnetic term uses the same plaquette curl as :func:`tdgl3d.physics.bfield.
eval_bfield_full`.  Consequently ``F`` is a Lyapunov functional of the solver —
it must decrease monotonically along any trajectory taken with a stable time
step (see ``tests/test_verification_conservation.py``), which makes it a sharp
correctness check on the right-hand side.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.material import MaterialMap
from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices
from ..operators.sparse_operators import plaquette_kappa2
from .rhs import (
    BoundaryVectors,
    _apply_boundary_conditions,
    _expand_interior_to_full,
)

__all__ = ["gl_free_energy", "gl_free_energy_terms"]


def gl_free_energy_terms(
    state: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    boundary: Optional[BoundaryVectors] = None,
    kappa: Optional[float] = None,
    material: Optional[MaterialMap] = None,
) -> dict[str, float]:
    """Return the individual terms of the discrete GL free energy.

    Parameters
    ----------
    state : ndarray, shape (n_state,)
        Flat interior state vector ``[ψ; φ_x; φ_y; φ_z]``.
    params, idx
        Grid description.
    boundary : BoundaryVectors, optional
        Applied-field boundary vectors for this instant.  Defaults to zero
        field.  The boundary conditions are applied before the energy is
        evaluated, exactly as they are inside :func:`tdgl3d.physics.rhs.eval_f`.
    kappa : float, optional
        Ginzburg-Landau parameter for the magnetic term.  Defaults to
        ``params.kappa``.  Ignored when *material* carries an explicit
        :attr:`~tdgl3d.core.material.MaterialMap.magnetic_kappa`.
    material : MaterialMap, optional
        Per-node material properties.  Pass the device's map to get the
        energy of a layered device: the condensation term is then taken
        over superconducting nodes only, and the magnetic term uses the
        same per-plaquette coefficient as the operators do.

    Returns
    -------
    dict
        ``{"condensation", "kinetic", "magnetic", "total"}``, each already
        multiplied by the cell volume.

    Notes
    -----
    Insulator nodes are driven by a relaxation term ``-ψ/τ`` that is not
    the variational derivative of this functional, so the ψ part of the
    energy is not guaranteed to decrease monotonically there.  The
    magnetic part is: it uses the plaquette coefficients of
    :func:`~tdgl3d.operators.sparse_operators.plaquette_kappa2`, which is
    exactly what the curl-curl operator is the gradient of.
    """
    n = params.n_interior
    m = idx.interior_to_full
    mj, mk = params.mj, params.mk
    hx, hy, hz = params.hx, params.hy, params.hz
    kappa = params.kappa if kappa is None else float(kappa)

    if boundary is None:
        zeros = np.zeros(params.dim_x, dtype=np.float64)
        boundary = BoundaryVectors(zeros, zeros.copy(), zeros.copy())

    psi = _expand_interior_to_full(state[:n], params, idx)
    phi_x = _expand_interior_to_full(state[n : 2 * n], params, idx)
    phi_y = _expand_interior_to_full(state[2 * n : 3 * n], params, idx)
    if params.is_3d:
        phi_z = _expand_interior_to_full(state[3 * n : 4 * n], params, idx)
    else:
        phi_z = np.zeros(params.dim_x, dtype=np.complex128)

    psi, phi_x, phi_y, phi_z = _apply_boundary_conditions(
        psi, phi_x, phi_y, phi_z, params, idx, boundary
    )

    volume = hx * hy * (hz if params.is_3d else 1.0)

    psi_m = psi[m]
    # The condensation energy is only defined where there is a condensate;
    # insulator nodes are relaxed to psi = 0 by a term outside this functional.
    sc = 1.0 if material is None else material.interior_sc_mask
    condensation = float(
        np.sum(sc * (-np.abs(psi_m) ** 2 + 0.5 * np.abs(psi_m) ** 4))
    )

    kinetic = float(
        np.sum(np.abs(np.exp(-1j * phi_x[m]) * psi[m + 1] - psi_m) ** 2) / hx**2
        + np.sum(np.abs(np.exp(-1j * phi_y[m]) * psi[m + mj] - psi_m) ** 2) / hy**2
    )
    if params.is_3d:
        kinetic += float(
            np.sum(np.abs(np.exp(-1j * phi_z[m]) * psi[m + mk] - psi_m) ** 2) / hz**2
        )

    # Magnetic term, plaquette by plaquette.  Each plaquette carries its
    # own coefficient, matching the operators exactly, so that this is the
    # functional whose gradient the curl-curl term is.
    nu = plaquette_kappa2(params, material)
    uniform = kappa**2

    bz2 = np.abs((phi_x[m] - phi_x[m + mj] - phi_y[m] + phi_y[m + 1]) / (hx * hy)) ** 2
    magnetic = float(np.sum((uniform if nu is None else nu[2][m]) * bz2))
    if params.is_3d:
        bx2 = np.abs(
            (phi_y[m] - phi_y[m + mk] - phi_z[m] + phi_z[m + mj]) / (hy * hz)
        ) ** 2
        by2 = np.abs(
            (phi_z[m] - phi_z[m + 1] - phi_x[m] + phi_x[m + mk]) / (hz * hx)
        ) ** 2
        magnetic += float(np.sum((uniform if nu is None else nu[0][m]) * bx2))
        magnetic += float(np.sum((uniform if nu is None else nu[1][m]) * by2))

    terms = {
        "condensation": condensation * volume,
        "kinetic": kinetic * volume,
        "magnetic": magnetic * volume,
    }
    terms["total"] = terms["condensation"] + terms["kinetic"] + terms["magnetic"]
    return terms


def gl_free_energy(
    state: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    boundary: Optional[BoundaryVectors] = None,
    kappa: Optional[float] = None,
    material: Optional[MaterialMap] = None,
) -> float:
    """Total discrete Ginzburg-Landau free energy of *state*.

    See :func:`gl_free_energy_terms` for the definition and the parameters.
    """
    return gl_free_energy_terms(
        state, params, idx, boundary, kappa, material
    )["total"]
