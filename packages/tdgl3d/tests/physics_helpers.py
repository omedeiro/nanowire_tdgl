"""Helpers shared by the ``test_verification_*`` physics suites.

Everything here is deliberately small and independent of the solver internals it
is used to check: a helper that reuses the machinery under test cannot falsify
it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tdgl3d import AppliedField, Device, SimulationParameters
from tdgl3d.mesh.indices import GridIndices, construct_indices
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.rhs import (
    BoundaryVectors,
    _apply_boundary_conditions,
    _expand_interior_to_full,
)
from tdgl3d.solvers.integrators import forward_euler

__all__ = [
    "cfl_limit",
    "expand_state",
    "gauge_transform",
    "interior_strides",
    "landau_gauge_links",
    "random_state",
    "run_euler",
    "smooth_gauge_field",
    "zero_boundary",
]


def zero_boundary(params: SimulationParameters) -> BoundaryVectors:
    """Boundary vectors for zero applied field."""
    zeros = np.zeros(params.dim_x, dtype=np.float64)
    return BoundaryVectors(zeros, zeros.copy(), zeros.copy())


def applied_boundary(
    params: SimulationParameters, idx: GridIndices, bz: float = 0.0,
    bx: float = 0.0, by: float = 0.0,
) -> BoundaryVectors:
    """Boundary vectors for a constant applied field."""
    return BoundaryVectors(*build_boundary_field_vectors(bx, by, bz, params, idx))


def cfl_limit(params: SimulationParameters) -> float:
    r"""Forward-Euler stability limit, set by the stiff ``κ²∇×∇×`` term.

    The commonly quoted bound ``h²/(4κ²)`` is a **two-dimensional** result.  In
    3D each link variable acquires a second transverse Laplacian direction, the
    spectral radius of the curl-curl block doubles, and the limit halves.  The
    dimension-aware bound used here is

    .. math::  \Delta t < \frac{h^2}{4\kappa^2 (d - 1)},

    which reproduces the familiar 2D value and is about 30% conservative in 3D
    (measured limits: 1.06x in 2D, 0.72x in 3D — see
    ``test_forward_euler_is_stable_below_the_cfl_limit``).
    """
    h_min = min(params.hx, params.hy, params.hz if params.is_3d else np.inf)
    transverse = 2.0 if params.is_3d else 1.0
    return h_min**2 / (4.0 * params.kappa**2 * transverse)


def interior_strides(params: SimulationParameters) -> tuple[int, int, int]:
    """``(stride_i, stride_j, stride_k)`` for arrays in *interior* numbering.

    The interior numbering is i-slowest / k-fastest — the opposite of the full
    grid.  Tests use this to assert that library code agrees.
    """
    nz_int = max(params.Nz - 1, 1)
    return ((params.Ny - 1) * nz_int, nz_int, 1)


def random_state(
    params: SimulationParameters,
    seed: int = 0,
    psi_scale: float = 0.3,
    phi_scale: float = 0.2,
) -> NDArray[np.complex128]:
    """A generic, non-symmetric state vector for algebraic identity checks."""
    rng = np.random.default_rng(seed)
    n = params.n_interior
    psi = 0.8 + psi_scale * (rng.normal(size=n) + 1j * rng.normal(size=n))
    n_phi = 3 if params.is_3d else 2
    phi = phi_scale * rng.normal(size=n_phi * n)
    return np.concatenate([psi, phi]).astype(np.complex128)


def smooth_gauge_field(
    params: SimulationParameters,
    seed: int = 0,
    amplitude: float = 0.4,
    margin: int = 2,
) -> NDArray[np.float64]:
    """A gauge function χ on the full grid, vanishing near the boundary.

    The boundary conditions pin the normal link variables on the outer faces, so
    a gauge transformation is only a symmetry of the discrete system when χ is
    constant there.  Keeping χ supported strictly inside ``margin`` nodes of the
    boundary makes the transformation exact rather than approximate.
    """
    rng = np.random.default_rng(seed)
    n_full = params.dim_x
    chi = np.zeros(n_full, dtype=np.float64)

    flat = np.arange(n_full)
    ii = flat % params.mj
    jj = (flat // params.mj) % (params.Ny + 1)
    kk = flat // params.mk if params.is_3d else np.zeros(n_full, dtype=int)

    inside = (
        (ii >= margin) & (ii <= params.Nx - margin)
        & (jj >= margin) & (jj <= params.Ny - margin)
    )
    if params.is_3d:
        inside &= (kk >= margin) & (kk <= params.Nz - margin)

    chi[inside] = amplitude * rng.normal(size=int(inside.sum()))
    return chi


def gauge_transform(
    state: NDArray[np.complexfloating],
    chi_full: NDArray[np.float64],
    params: SimulationParameters,
    idx: GridIndices,
) -> NDArray[np.complex128]:
    """Apply ``ψ → ψ e^{iχ}``, ``φ_μ → φ_μ + Δ_μ χ`` to an interior state.

    This is the gauge transformation of the covariant derivative ``∇ - iA`` used
    throughout the solver.  Physical observables (``|ψ|``, ``B``, ``J_s``, the
    free energy, vortex counts) must be invariant under it, and the right-hand
    side must be covariant.
    """
    n = params.n_interior
    m = idx.interior_to_full
    mj, mk = params.mj, params.mk

    out = np.array(state, dtype=np.complex128, copy=True)
    out[:n] = state[:n] * np.exp(1j * chi_full[m])
    out[n : 2 * n] = state[n : 2 * n] + (chi_full[m + 1] - chi_full[m])
    out[2 * n : 3 * n] = state[2 * n : 3 * n] + (chi_full[m + mj] - chi_full[m])
    if params.is_3d:
        out[3 * n : 4 * n] = state[3 * n : 4 * n] + (chi_full[m + mk] - chi_full[m])
    return out


def landau_gauge_links(
    params: SimulationParameters, bz: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Full-grid link variables for a uniform ``Bz`` in the Landau gauge.

    ``A = (0, B x, 0)`` gives ``φ_x = 0`` and ``φ_y[i, j] = B h_x h_y i``, whose
    plaquette curl is exactly ``Bz`` everywhere.
    """
    n_full = params.dim_x
    ii = np.arange(n_full) % params.mj
    phi_x = np.zeros(n_full, dtype=np.complex128)
    phi_y = (bz * params.hx * params.hy * ii).astype(np.complex128)
    return phi_x, phi_y


def expand_state(
    state: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    boundary: BoundaryVectors | None = None,
):
    """Expand an interior state to the full grid and apply boundary conditions."""
    n = params.n_interior
    psi = _expand_interior_to_full(state[:n], params, idx)
    phi_x = _expand_interior_to_full(state[n : 2 * n], params, idx)
    phi_y = _expand_interior_to_full(state[2 * n : 3 * n], params, idx)
    if params.is_3d:
        phi_z = _expand_interior_to_full(state[3 * n : 4 * n], params, idx)
    else:
        phi_z = np.zeros(params.dim_x, dtype=np.complex128)
    if boundary is None:
        boundary = zero_boundary(params)
    return _apply_boundary_conditions(psi, phi_x, phi_y, phi_z, params, idx, boundary)


def run_euler(
    params: SimulationParameters,
    applied_bz: float,
    n_steps: int,
    dt: float,
    noise_amplitude: float = 0.01,
    seed: int | None = None,
    save_every: int = 1,
    x0: NDArray | None = None,
):
    """Run Forward Euler and return ``(times, states, device, idx)``."""
    device = Device(params, applied_field=AppliedField(Bz=applied_bz, t_on_fraction=1.0))
    idx = device.idx
    t_stop = n_steps * dt
    applied_field = device.applied_field

    def eval_u(t, X):
        bx, by, bz = applied_field.evaluate(t, t_stop)
        return BoundaryVectors(*build_boundary_field_vectors(bx, by, bz, params, idx))

    if x0 is None:
        x0 = device.initial_state(noise_amplitude=noise_amplitude, seed=seed).data

    times, states = forward_euler(
        x0, params, idx, eval_u, 0.0, t_stop, dt,
        save_every=save_every, progress=False,
    )
    return times, states, device, idx


def make_grid(**kwargs) -> tuple[SimulationParameters, GridIndices]:
    """Build ``(params, idx)`` in one call."""
    params = SimulationParameters(**kwargs)
    return params, construct_indices(params)
