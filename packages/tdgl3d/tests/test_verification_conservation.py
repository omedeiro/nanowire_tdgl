"""Conservation-law and structural-identity verification for the TDGL solver.

Three families of checks:

* **Exact discrete identities.**  ``∇·(∇×A) = 0`` and ``∇·(∇×∇×A) = 0`` hold to
  round-off *provided the divergence uses the stencil dual to the curl*.  A
  mismatched stencil turns a machine-precision identity into an O(h) residual,
  which is easy to mistake for "discretisation error" and paper over with a
  loose tolerance.
* **Lyapunov decay.**  The solver is a gradient flow of
  :func:`tdgl3d.physics.free_energy.gl_free_energy`.  At zero applied field the
  free energy must decrease at *every* step; a single increase means the
  right-hand side is not the gradient of the energy it claims to minimise.
* **Charge conservation.**  Because ``∇·(∇×∇×A) ≡ 0``, the link equation implies
  ``∂(∇·A)/∂t = ∇·J_s``, so a steady state must carry a divergence-free
  supercurrent.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d import AppliedField, Device, SimulationParameters
from tdgl3d.core.state import StateVector
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.current_density import eval_supercurrent_density
from tdgl3d.physics.free_energy import gl_free_energy, gl_free_energy_terms
from tdgl3d.physics.rhs import eval_f

from .physics_helpers import (
    applied_boundary,
    cfl_limit,
    expand_state,
    interior_strides,
    make_grid,
    random_state,
    run_euler,
    zero_boundary,
)


def _bulk_interior_mask(params, lo: int = 1) -> np.ndarray:
    """Interior indices whose ``-lo`` neighbours are still interior nodes."""
    n = params.n_interior
    si, sj, sk = interior_strides(params)
    ny, nz = params.Ny - 1, max(params.Nz - 1, 1)
    flat = np.arange(n)
    ii, jj, kk = flat // si, (flat // sj) % ny, flat % nz
    ok = (ii >= lo) & (jj >= lo)
    if params.is_3d:
        ok &= kk >= lo
    return flat[ok]


# ---------------------------------------------------------------------------
# Exact discrete identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid",
    [dict(Nx=8, Ny=7, Nz=1), dict(Nx=6, Ny=7, Nz=8)],
    ids=lambda g: f"{g['Nx']}x{g['Ny']}x{g['Nz']}",
)
def test_divergence_of_discrete_curl_is_exactly_zero(grid, phys_log):
    """∇·B = 0 to round-off when the divergence is dual to the plaquette curl.

    ``B`` is a *forward* plaquette curl, so its exact discrete divergence is the
    *forward* difference.  The mixed forward-curl/backward-divergence pairing
    used previously leaves an O(1) residual that has nothing to do with physics.
    """
    params, idx = make_grid(kappa=2.0, **grid)
    rng = np.random.default_rng(3)
    n_full = params.dim_x
    phi = [rng.normal(size=n_full).astype(np.complex128) * 0.3 for _ in range(3)]

    bx, by, bz = eval_bfield_full(phi[0], phi[1], phi[2], params, idx)
    b_full = [np.zeros(n_full) for _ in range(3)]
    m = idx.interior_to_full
    for arr, val in zip(b_full, (bx, by, bz)):
        arr[m] = val

    mj, mk = params.mj, params.mk
    ii, jj = m % mj, (m // mj) % (params.Ny + 1)
    kk = m // mk if params.is_3d else np.zeros_like(m)
    ok = (ii <= params.Nx - 2) & (jj <= params.Ny - 2)
    if params.is_3d:
        ok &= kk <= params.Nz - 2
    mb = m[ok]

    div = (b_full[0][mb + 1] - b_full[0][mb]) / params.hx
    div = div + (b_full[1][mb + mj] - b_full[1][mb]) / params.hy
    if params.is_3d:
        div = div + (b_full[2][mb + mk] - b_full[2][mb]) / params.hz

    b_scale = float(np.max(np.abs(np.stack([bx, by, bz]))))
    name = f"test_divergence_of_discrete_curl_is_exactly_zero[{params.Nx}x{params.Ny}x{params.Nz}]"
    with phys_log.test(
        name, dict(grid),
        "∇·B must vanish identically, not merely to discretisation order",
    ) as log:
        log["n_bulk_nodes"] = int(mb.size)
        log["B_scale"] = b_scale
        log.check_above("bulk nodes tested", float(mb.size), 1.0)
        log.check_below(
            "max|∇·B| / max|B|", float(np.max(np.abs(div))) / b_scale, 1e-13,
            detail="forward-difference divergence of the forward plaquette curl",
        )


@pytest.mark.parametrize(
    "grid",
    [dict(Nx=9, Ny=8, Nz=1), dict(Nx=7, Ny=6, Nz=6)],
    ids=lambda g: f"{g['Nx']}x{g['Ny']}x{g['Nz']}",
)
def test_curl_curl_operator_is_divergence_free(grid, phys_log):
    """∇·(∇×∇×A) = 0 identically — the basis of discrete charge conservation."""
    params, idx = make_grid(kappa=2.0, **grid)
    n = params.n_interior
    state = random_state(params, seed=5)
    state[:n] = 0.0  # remove the supercurrent source, leaving only curl-curl

    rhs = eval_f(state, params, idx, zero_boundary(params))
    s_x = rhs[n : 2 * n] / params.hx
    s_y = rhs[2 * n : 3 * n] / params.hy
    s_z = rhs[3 * n : 4 * n] / params.hz if params.is_3d else np.zeros(n)

    si, sj, sk = interior_strides(params)
    mb = _bulk_interior_mask(params)
    div = (s_x[mb] - s_x[mb - si]) / params.hx + (s_y[mb] - s_y[mb - sj]) / params.hy
    if params.is_3d:
        div = div + (s_z[mb] - s_z[mb - sk]) / params.hz

    scale = float(np.max(np.abs(np.concatenate([s_x, s_y, s_z]))))
    name = f"test_curl_curl_operator_is_divergence_free[{params.Nx}x{params.Ny}x{params.Nz}]"
    with phys_log.test(
        name, dict(grid),
        "the discrete curl-curl operator must annihilate gradients exactly",
    ) as log:
        log["operator_scale"] = scale
        log.check_below(
            "max|∇·(∇×∇×A)| / scale",
            float(np.max(np.abs(div))) / scale, 1e-13,
        )


def test_uniform_state_is_an_exact_fixed_point(phys_log):
    """|ψ|=1, φ=0 at zero applied field is a stationary solution to round-off."""
    params, idx = make_grid(Nx=7, Ny=6, Nz=5, kappa=2.0)
    state = StateVector.uniform_superconducting(params)
    rhs = eval_f(state.data, params, idx, zero_boundary(params))
    energy = gl_free_energy_terms(state.data, params, idx)

    with phys_log.test(
        "test_uniform_state_is_an_exact_fixed_point",
        {"Nx": 7, "Ny": 6, "Nz": 5},
        "the Meissner ground state must not drift",
    ) as log:
        log["free_energy_terms"] = {k: round(v, 10) for k, v in energy.items()}
        log.check_below("max|dX/dt|", float(np.max(np.abs(rhs))), 1e-13)
        log.check_below(
            "kinetic + magnetic energy", energy["kinetic"] + energy["magnetic"], 1e-13,
            detail="a uniform state carries no gradient or field energy",
        )
        log.check_close(
            "condensation energy per unit volume",
            energy["condensation"] / (params.n_interior * params.hx * params.hy * params.hz),
            -0.5, atol=1e-12,
            detail="-|ψ|² + ½|ψ|⁴ = -½ at |ψ| = 1",
        )


# ---------------------------------------------------------------------------
# Lyapunov decay of the free energy
# ---------------------------------------------------------------------------


def test_free_energy_decreases_monotonically_at_zero_field(phys_log):
    """Strict Lyapunov decay: F must not increase on a single step.

    No adaptive tolerance is used — the bound is fixed at round-off relative to
    the total energy released, so the check is falsifiable.
    """
    params = SimulationParameters(Nx=12, Ny=11, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    dt = 0.5 * cfl_limit(params)
    n_steps = 400
    times, states, device, idx = run_euler(
        params, applied_bz=0.0, n_steps=n_steps, dt=dt, noise_amplitude=0.25, seed=7,
    )
    energies = np.array(
        [gl_free_energy(states[:, s], params, idx) for s in range(states.shape[1])]
    )
    increments = np.diff(energies)
    released = float(energies[0] - energies[-1])
    worst_increase = float(increments.max())

    with phys_log.test(
        "test_free_energy_decreases_monotonically_at_zero_field",
        {"Nx": 12, "Ny": 11, "h": 0.5, "kappa": 2.0, "dt": round(dt, 6), "n_steps": n_steps},
        "TDGL is a gradient flow of the GL free energy, so F(t) is non-increasing",
    ) as log:
        log["F_initial"] = float(energies[0])
        log["F_final"] = float(energies[-1])
        log["energy_released"] = released
        log["n_steps_increasing"] = int((increments > 0).sum())
        log.check_above(
            "energy released (test would be vacuous otherwise)", released, 1.0,
        )
        log.check_below(
            "steps on which F increased", float((increments > 0).sum()), 0.0,
            detail=f"out of {increments.size} steps",
        )
        log.check_below(
            "worst single-step ΔF / energy released",
            worst_increase / released, 1e-9,
        )


def test_free_energy_decreases_while_relaxing_in_a_field(phys_log):
    """With the applied field held fixed, relaxation still lowers F.

    The boundary conditions pin ``B`` on the outer plaquettes, so the boundary
    ring exchanges energy with the source and the decay is monotone only up to
    that boundary term; the tolerance below is a fixed fraction of the energy
    released, never derived from the observed violation.
    """
    params = SimulationParameters(Nx=12, Ny=12, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    bz = 0.15
    dt = 0.5 * cfl_limit(params)
    device = Device(params, applied_field=AppliedField(Bz=bz, t_on_fraction=1.0))
    idx = device.idx
    boundary = applied_boundary(params, idx, bz=bz)

    # Relax to the Meissner state at this field, then perturb ψ and relax again:
    # the field is already switched on, so no work is injected through it.
    _, warm, _, _ = run_euler(params, bz, n_steps=600, dt=dt, noise_amplitude=0.0)
    rng = np.random.default_rng(9)
    start = warm[:, -1].copy()
    n = params.n_interior
    start[:n] *= 1.0 + 0.3 * rng.normal(size=n)

    _, states, _, _ = run_euler(params, bz, n_steps=400, dt=dt, x0=start)
    energies = np.array(
        [gl_free_energy(states[:, s], params, idx, boundary) for s in range(states.shape[1])]
    )
    increments = np.diff(energies)
    released = float(energies[0] - energies[-1])

    with phys_log.test(
        "test_free_energy_decreases_while_relaxing_in_a_field",
        {"Nx": 12, "h": 0.5, "kappa": 2.0, "Bz": bz},
        "relaxation at fixed applied field lowers the free energy",
    ) as log:
        log["F_initial"] = float(energies[0])
        log["F_final"] = float(energies[-1])
        log["energy_released"] = released
        log.check_above("energy released", released, 0.5)
        log.check_below(
            "worst single-step ΔF / energy released",
            float(increments.max()) / released, 1e-6,
        )


# ---------------------------------------------------------------------------
# Currents
# ---------------------------------------------------------------------------


def test_supercurrent_is_divergence_free_in_steady_state(phys_log):
    """∇·J_s → 0 once the link variables stop evolving (charge conservation)."""
    params = SimulationParameters(Nx=14, Ny=14, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    bz = 0.12
    dt = 0.5 * cfl_limit(params)
    _, states, device, idx = run_euler(
        params, bz, n_steps=1500, dt=dt, noise_amplitude=0.0, save_every=100,
    )
    boundary = applied_boundary(params, idx, bz=bz)
    psi, px, py, pz = expand_state(states[:, -1], params, idx, boundary)
    jx, jy, _ = eval_supercurrent_density(psi, px, py, pz, params, idx)

    si, sj, _ = interior_strides(params)
    mb = _bulk_interior_mask(params, lo=2)
    div = (jx[mb] - jx[mb - si]) / params.hx + (jy[mb] - jy[mb - sj]) / params.hy

    j_scale = float(np.max(np.abs(np.stack([jx, jy]))))
    drift = float(np.max(np.abs(states[:, -1] - states[:, -2])))

    with phys_log.test(
        "test_supercurrent_is_divergence_free_in_steady_state",
        {"Nx": 14, "h": 0.5, "kappa": 2.0, "Bz": bz},
        "∂(∇·A)/∂t = ∇·J_s, so a stationary gauge field forces a solenoidal current",
    ) as log:
        log["J_scale"] = j_scale
        log["state_drift_between_saves"] = drift
        log.check_below("state drift between saved steps", drift, 1e-6)
        log.check_below(
            "max|∇·J_s| · h / max|J_s|",
            float(np.max(np.abs(div))) * params.hx / j_scale, 1e-6,
        )


def test_normal_supercurrent_vanishes_on_external_boundaries(phys_log):
    """J_n = 0 on every outer face — the physical no-current-leaks condition.

    This measures the gauge-invariant current on the boundary links rather than
    just asserting that the link variables were zeroed there, so it also covers
    the high face, where the link variable is a live degree of freedom and the
    condition is carried by the ψ boundary condition instead.
    """
    params = SimulationParameters(Nx=8, Ny=7, Nz=6, kappa=2.0)
    bz = 0.4
    dt = 0.5 * cfl_limit(params)
    _, states, device, idx = run_euler(params, bz, n_steps=200, dt=dt, seed=13)
    boundary = applied_boundary(params, idx, bz=bz)
    psi, px, py, pz = expand_state(states[:, -1], params, idx, boundary)

    mj, mk = params.mj, params.mk

    def link_current(node, phi, offset):
        return np.abs(np.imag(np.conj(psi[node]) * np.exp(-1j * phi[node]) * psi[node + offset]))

    faces = {
        "x_lo": link_current(idx.x_face_lo_inner, px, 1),
        "x_hi": link_current(idx.x_last_inner, px, 1),
        "y_lo": link_current(idx.y_face_lo_inner, py, mj),
        "y_hi": link_current(idx.y_last_inner, py, mj),
        "z_lo": link_current(idx.z_face_lo_inner, pz, mk),
        "z_hi": link_current(idx.z_last_inner, pz, mk),
    }
    bulk = float(np.max(np.abs(np.imag(
        np.conj(psi[idx.interior_to_full]) * np.exp(-1j * px[idx.interior_to_full])
        * psi[idx.interior_to_full + 1]
    ))))

    with phys_log.test(
        "test_normal_supercurrent_vanishes_on_external_boundaries",
        {"Nx": 8, "Ny": 7, "Nz": 6, "Bz": bz},
        "no supercurrent may cross the superconductor/vacuum interface",
    ) as log:
        log["bulk_current_scale"] = bulk
        log.check_above("bulk current scale (non-trivial state)", bulk, 1e-4)
        for face, values in faces.items():
            log.check_below(
                f"max|J_n| on {face} face",
                float(np.max(values)) if values.size else 0.0,
                1e-12,
            )


# ---------------------------------------------------------------------------
# Time-step stability
# ---------------------------------------------------------------------------


def test_forward_euler_is_stable_below_the_cfl_limit(phys_log):
    """Below h²/(4κ²) the uniform state stays put; above it, it blows up."""
    params = SimulationParameters(Nx=6, Ny=6, Nz=1, kappa=2.0)
    limit = cfl_limit(params)

    _, stable, _, _ = run_euler(params, 0.0, n_steps=200, dt=0.9 * limit, noise_amplitude=0.05, seed=2)
    _, unstable, _, _ = run_euler(params, 0.5, n_steps=200, dt=3.0 * limit, noise_amplitude=0.05, seed=2)

    n = params.n_interior
    psi2_stable = np.abs(stable[:n, -1]) ** 2
    final_unstable = unstable[:, -1]
    diverged = (not np.all(np.isfinite(final_unstable))) or float(
        np.max(np.abs(final_unstable))
    ) > 1e3 or float(np.mean(np.abs(final_unstable[:n]))) < 0.5

    with phys_log.test(
        "test_forward_euler_is_stable_below_the_cfl_limit",
        {"Nx": 6, "kappa": 2.0, "cfl_limit": limit},
        "the explicit step size limit is set by the κ²∇×∇× term",
    ) as log:
        log["cfl_limit"] = float(limit)
        log["max_psi2_stable"] = float(np.max(psi2_stable))
        log["min_psi2_stable"] = float(np.min(psi2_stable))
        log["unstable_run_diverged"] = bool(diverged)
        log.check_close(
            "max|ψ|² at dt = 0.9 dt_CFL", float(np.max(psi2_stable)), 1.0, atol=0.05,
            detail="relaxes to the uniform state without amplification",
        )
        log.check_close(
            "min|ψ|² at dt = 0.9 dt_CFL", float(np.min(psi2_stable)), 1.0, atol=0.05,
        )
        log.check_above(
            "run at dt = 3 dt_CFL loses the superconducting state",
            float(diverged), 1.0,
        )
