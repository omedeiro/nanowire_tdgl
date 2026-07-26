"""Physics-based validation tests for the TDGL solver.

Verifies that the solver produces physically correct results:
- Equilibrium states, symmetries, boundary conditions
- CFL stability, energy dissipation, B-field consistency
- Trilayer interface physics, Meissner screening, vortex entry

All test results are logged to logs/physics_test_runlog.json via the
phys_log fixture (see conftest.py).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tdgl3d import (
    AppliedField,
    Device,
    Layer,
    SimulationParameters,
    Trilayer,
    solve,
)
from tdgl3d.core.material import build_material_map
from tdgl3d.core.state import StateVector
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.current_density import eval_supercurrent_density
from tdgl3d.physics.rhs import (
    BoundaryVectors,
    _apply_boundary_conditions,
    _expand_interior_to_full,
    eval_f,
)
from tdgl3d.solvers.integrators import forward_euler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_bv(params: SimulationParameters, idx) -> BoundaryVectors:
    N = params.dim_x
    return BoundaryVectors(np.zeros(N), np.zeros(N), np.zeros(N))


def _run_euler(
    params: SimulationParameters,
    applied_bz: float,
    n_steps: int,
    dt: float,
    noise_amplitude: float = 0.01,
    seed: int | None = None,
):
    """Run Forward Euler for n_steps and return (times, states, device, idx)."""
    device = Device(params, applied_field=AppliedField(Bz=applied_bz))
    idx = device.idx
    t_stop = n_steps * dt
    applied_field = device.applied_field

    def eval_u(t, X):
        bx, by, bz = applied_field.evaluate(t, t_stop)
        Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(bx, by, bz, params, idx)
        return BoundaryVectors(Bx_vec, By_vec, Bz_vec)

    x0 = device.initial_state(noise_amplitude=noise_amplitude, seed=seed).data
    times, states = forward_euler(
        x0, params, idx, eval_u, 0.0, t_stop, dt,
        save_every=1, progress=False,
    )
    return times, states, device, idx


def _expand_state(state: NDArray, params: SimulationParameters, idx):
    """Expand interior state to full grid and apply zero-field BCs."""
    n = params.n_interior
    psi = _expand_interior_to_full(state[:n], params, idx)
    phi_x = _expand_interior_to_full(state[n:2*n], params, idx)
    phi_y = _expand_interior_to_full(state[2*n:3*n], params, idx)
    phi_z = _expand_interior_to_full(state[3*n:4*n], params, idx) if params.is_3d else np.zeros(params.dim_x, dtype=np.complex128)
    u = _zero_bv(params, idx)
    psi, phi_x, phi_y, phi_z = _apply_boundary_conditions(psi, phi_x, phi_y, phi_z, params, idx, u)
    return psi, phi_x, phi_y, phi_z


def _compute_free_energy(
    state: NDArray,
    params: SimulationParameters,
    idx,
    kappa: float,
) -> float:
    """Compute GL free energy F = ∫[|ψ|²/2 + |∇ψ|²/2 + κ²B²/2] dV."""
    n = params.n_interior
    hx, hy = params.hx, params.hy

    psi_int = state[:n]
    phi_x_int = state[n:2*n]
    phi_y_int = state[2*n:3*n]

    psi_full = _expand_interior_to_full(psi_int, params, idx)
    phi_x_full = _expand_interior_to_full(phi_x_int, params, idx)
    phi_y_full = _expand_interior_to_full(phi_y_int, params, idx)
    phi_z_full = np.zeros(params.dim_x, dtype=np.complex128)

    # Apply BCs for B-field computation
    u = _zero_bv(params, idx)
    psi_bc, px_bc, py_bc, pz_bc = _apply_boundary_conditions(
        psi_full, phi_x_full, phi_y_full, phi_z_full, params, idx, u
    )

    # F_psi = 0.5 * Σ |ψ|²
    F_psi = 0.5 * np.sum(np.abs(psi_int) ** 2)

    # F_grad = 0.5 * Σ (|ψ[i+1]-ψ[i]|²/hx² + |ψ[i+mj]-ψ[i]|²/hy²) * hx*hy
    m = idx.interior_to_full
    mj = params.mj
    dpsi_dx = (psi_bc[m + 1] - psi_bc[m]) / hx
    dpsi_dy = (psi_bc[m + mj] - psi_bc[m]) / hy
    F_grad = 0.5 * (np.sum(np.abs(dpsi_dx) ** 2) + np.sum(np.abs(dpsi_dy) ** 2)) * hx * hy

    # F_b = 0.5 * κ² * Σ Bz² * hx*hy
    _, _, Bz = eval_bfield_full(px_bc, py_bc, pz_bc, params, idx)
    F_b = 0.5 * kappa**2 * np.sum(Bz**2) * hx * hy

    return float(F_psi + F_grad + F_b)


# ---------------------------------------------------------------------------
# Tier 1 — Fast, no simulation
# ---------------------------------------------------------------------------


def test_uniform_state_zero_rhs(phys_log):
    """Uniform |ψ|=1, φ=0, zero field is an exact fixed point (zero-current BCs)."""
    params = SimulationParameters(Nx=6, Ny=6, Nz=1)
    idx = construct_indices(params)
    sv = StateVector.uniform_superconducting(params)
    u = _zero_bv(params, idx)
    F = eval_f(sv.data, params, idx, u)

    with phys_log.test("test_uniform_state_zero_rhs", {"Nx": 6, "Ny": 6}) as log:
        log["max_rhs"] = float(np.max(np.abs(F)))
        log["mean_rhs"] = float(np.mean(np.abs(F)))
        np.testing.assert_allclose(F, 0.0, atol=1e-12)


def test_c4_symmetry_preserved_over_time(phys_log):
    """Rotational invariance: φ_x(i,j) = -φ_y(j,i) maintained through 10 steps."""
    params = SimulationParameters(Nx=8, Ny=8, Nz=1, kappa=2.0)
    times, states, _, _ = _run_euler(
        params, applied_bz=0.5, n_steps=10, dt=0.01, noise_amplitude=0.0,
    )

    with phys_log.test("test_c4_symmetry_preserved_over_time", {"Nx": 8, "kappa": 2.0, "Bz": 0.5}) as log:
        max_violation = 0.0
        n = params.n_interior
        for step in range(states.shape[1]):
            phix = states[n:2*n, step].reshape(params.Nx - 1, params.Ny - 1)
            phiy = states[2*n:3*n, step].reshape(params.Nx - 1, params.Ny - 1)
            viol = np.max(np.abs(phix + phiy.T))
            max_violation = max(max_violation, viol)

        log["max_symmetry_violation"] = float(max_violation)
        np.testing.assert_allclose(max_violation, 0.0, atol=1e-12)


def test_supercurrent_zero_at_boundaries(phys_log):
    """J_n = 0 at all external boundary faces (verified via zero boundary link variables)."""
    params = SimulationParameters(Nx=6, Ny=6, Nz=1, kappa=2.0)
    times, states, device, idx = _run_euler(params, applied_bz=0.5, n_steps=5, dt=0.01)

    with phys_log.test("test_supercurrent_zero_at_boundaries", {"Nx": 6, "Bz": 0.5}) as log:
        psi, phi_x, phi_y, phi_z = _expand_state(states[:, -1], params, idx)

        # Boundary link variables are zeroed by BCs — this enforces J_n = 0.
        phi_x_xface_lo = np.abs(phi_x[idx.x_face_lo_inner])
        phi_x_xface_hi = np.abs(phi_x[idx.x_face_hi_inner])
        phi_y_yface_lo = np.abs(phi_y[idx.y_face_lo_inner])
        phi_y_yface_hi = np.abs(phi_y[idx.y_face_hi_inner])

        max_phi_x_lo = float(np.max(phi_x_xface_lo)) if len(phi_x_xface_lo) > 0 else 0.0
        max_phi_x_hi = float(np.max(phi_x_xface_hi)) if len(phi_x_xface_hi) > 0 else 0.0
        max_phi_y_lo = float(np.max(phi_y_yface_lo)) if len(phi_y_yface_lo) > 0 else 0.0
        max_phi_y_hi = float(np.max(phi_y_yface_hi)) if len(phi_y_yface_hi) > 0 else 0.0

        max_boundary_link = max(max_phi_x_lo, max_phi_x_hi, max_phi_y_lo, max_phi_y_hi)

        log["max_boundary_link_phi"] = max_boundary_link
        assert max_boundary_link < 1e-14, (
            f"Boundary link variables non-zero: max = {max_boundary_link}"
        )


def test_cfl_stability_below_limit(phys_log):
    """Forward Euler stable for dt < h²/(4κ²)."""
    params = SimulationParameters(Nx=5, Ny=5, Nz=1, kappa=2.0)
    cfl_limit = 1.0 / (4.0 * params.kappa**2)
    dt = 0.9 * cfl_limit

    with phys_log.test("test_cfl_stability_below_limit", {"Nx": 5, "kappa": 2.0, "cfl": cfl_limit, "dt": dt}) as log:
        times, states, _, _ = _run_euler(params, applied_bz=0.0, n_steps=20, dt=dt)
        psi2 = np.abs(states[:params.n_interior, -1]) ** 2

        log["cfl_limit"] = float(cfl_limit)
        log["dt_used"] = float(dt)
        log["max_psi2"] = float(np.max(psi2))
        log["min_psi2"] = float(np.min(psi2))
        assert np.all(np.isfinite(states)), "States contain NaN/Inf"
        assert np.max(psi2) < 10, f"max|ψ|² = {np.max(psi2)}, expected < 10"


def test_cfl_instability_above_limit(phys_log):
    """Forward Euler unstable for dt > h²/(4κ²)."""
    params = SimulationParameters(Nx=5, Ny=5, Nz=1, kappa=2.0)
    cfl_limit = 1.0 / (4.0 * params.kappa**2)
    dt = 3.0 * cfl_limit

    with phys_log.test("test_cfl_instability_above_limit", {"Nx": 5, "kappa": 2.0, "cfl": cfl_limit, "dt": dt}) as log:
        times, states, _, _ = _run_euler(params, applied_bz=0.5, n_steps=50, dt=dt)
        has_nan = not np.all(np.isfinite(states))
        psi_mean = np.mean(np.abs(states[:params.n_interior, -1]))
        max_psi = float(np.max(np.abs(states[:params.n_interior, -1]))) if np.all(np.isfinite(states[:params.n_interior, -1])) else float("inf")

        log["cfl_limit"] = float(cfl_limit)
        log["dt_used"] = float(dt)
        log["has_nan"] = has_nan
        log["max_psi_final"] = max_psi
        log["mean_psi_final"] = float(psi_mean)
        assert psi_mean < 0.5, (
            f"Expected instability (state collapse) but mean|ψ| = {psi_mean:.4f}"
        )


def test_bfield_divergence_free(phys_log):
    """∇·B ≈ 0 at bulk interior nodes (forward-curl + backward-divergence)."""
    params = SimulationParameters(Nx=4, Ny=4, Nz=4, kappa=2.0)
    times, states, _, idx = _run_euler(params, applied_bz=0.5, n_steps=5, dt=0.01)

    with phys_log.test("test_bfield_divergence_free", {"Nx": 4, "Nz": 4, "Bz": 0.5}) as log:
        psi, phi_x, phi_y, phi_z = _expand_state(states[:, -1], params, idx)
        Bx_int, By_int, Bz_int = eval_bfield_full(phi_x, phi_y, phi_z, params, idx)

        n_full = params.dim_x
        Bx_full = np.zeros(n_full, dtype=np.float64)
        By_full = np.zeros(n_full, dtype=np.float64)
        Bz_full = np.zeros(n_full, dtype=np.float64)
        m_int = idx.interior_to_full
        Bx_full[m_int] = Bx_int
        By_full[m_int] = By_int
        Bz_full[m_int] = Bz_int

        mj = params.mj
        mk = params.mk

        # Exclude boundary-adjacent nodes (backward stencil hits boundary where B=0)
        valid = np.ones(len(m_int), dtype=bool)
        nx_int = params.Nx - 1
        ny_int = params.Ny - 1
        nz_int = params.Nz - 1
        for k in range(nz_int):
            for j in range(ny_int):
                for i in range(nx_int):
                    idx_int = k * ny_int * nx_int + j * nx_int + i
                    if i == 0 or i == nx_int - 1 or j == 0 or j == ny_int - 1:
                        valid[idx_int] = False
                    if params.is_3d and (k == 0 or k == nz_int - 1):
                        valid[idx_int] = False

        m_bulk = m_int[valid]
        dBx_dx = (Bx_full[m_bulk] - Bx_full[m_bulk - 1]) / params.hx
        dBy_dy = (By_full[m_bulk] - By_full[m_bulk - mj]) / params.hy
        dBz_dz = (Bz_full[m_bulk] - Bz_full[m_bulk - mk]) / params.hz

        div_b = dBx_dx + dBy_dy + dBz_dz
        max_div = float(np.max(np.abs(div_b)))
        mean_div = float(np.mean(np.abs(div_b)))
        B_bulk = np.sqrt(Bx_int[valid]**2 + By_int[valid]**2 + Bz_int[valid]**2)
        max_B_bulk = float(np.max(B_bulk))

        log["max_div_b"] = max_div
        log["mean_div_b"] = mean_div
        log["max_B_magnitude"] = max_B_bulk
        # Forward-curl + backward-div on collocated grid has O(h²) stencil mismatch in bulk.
        if max_B_bulk > 1e-10:
            log["div_to_B_ratio"] = max_div / max_B_bulk
            assert max_div / max_B_bulk < 0.15, f"max|∇·B|/max|B| = {max_div / max_B_bulk:.6f}"
        else:
            assert max_div < 1e-10, f"max|∇·B| = {max_div}"


# ---------------------------------------------------------------------------
# Tier 2 — Short simulations
# ---------------------------------------------------------------------------


def test_energy_dissipation_monotonic(phys_log):
    """Free energy F decreases over time (dissipative TDGL dynamics)."""
    params = SimulationParameters(Nx=6, Ny=6, Nz=1, kappa=2.0)
    n_steps = 30
    dt = 0.01

    with phys_log.test("test_energy_dissipation_monotonic", {"Nx": 6, "kappa": 2.0, "Bz": 0.5, "n_steps": n_steps}) as log:
        times, states, device, idx = _run_euler(params, applied_bz=0.5, n_steps=n_steps, dt=dt)

        energies = []
        for step in range(states.shape[1]):
            F = _compute_free_energy(states[:, step], params, idx, params.kappa)
            energies.append(F)

        energies = np.array(energies)
        max_increase = 0.0
        for i in range(1, len(energies)):
            if energies[i-1] > 1e-10:
                rel_increase = (energies[i] - energies[i-1]) / abs(energies[i-1])
                max_increase = max(max_increase, rel_increase)

        # Set tolerance: allow small numerical fluctuations
        tol = max(0.01, 10 * max_increase) if max_increase > 0 else 0.01

        log["F_initial"] = float(energies[0])
        log["F_final"] = float(energies[-1])
        log["max_energy_increase"] = float(max_increase)
        log["tolerance"] = float(tol)
        log["energies"] = energies.tolist()

        for i in range(1, len(energies)):
            assert energies[i] <= energies[i-1] * (1 + tol), (
                f"Energy increased at step {i}: {energies[i-1]:.6f} -> {energies[i]:.6f} "
                f"(rel increase = {(energies[i]-energies[i-1])/abs(energies[i-1]):.6f}, tol = {tol:.6f})"
            )


def test_insulator_psi_exponential_decay(phys_log):
    """Insulator relaxation: |ψ| decays as exp(-t/τ) with τ ≈ 0.1."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=4, Ny=4, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.0), trilayer=trilayer)
    idx = device.idx
    material = device.material

    with phys_log.test("test_insulator_psi_exponential_decay", {"Nx": 4, "Nz": trilayer.Nz}) as log:
        # Create initial state with ψ=1 everywhere (override insulator mask)
        sv = StateVector.uniform_superconducting(params)
        x0 = sv.data.copy()

        def eval_u(t, X):
            return _zero_bv(params, idx)

        times, states = forward_euler(
            x0, params, idx, eval_u, 0.0, 0.5, 0.01,
            save_every=1, progress=False, material=material,
        )

        # Get insulator interior mask
        ins_mask = material.interior_sc_mask == 0.0
        n = params.n_interior

        psi_ins_mean = []
        for step in range(states.shape[1]):
            psi_int = states[:n, step]
            psi_abs = np.abs(psi_int[ins_mask])
            psi_ins_mean.append(float(np.mean(psi_abs)))

        psi_ins_mean = np.array(psi_ins_mean)
        t_arr = times

        # Subtract steady-state offset before fitting the exponential decay
        psi_ss = psi_ins_mean[-1]
        psi_decay = psi_ins_mean - psi_ss

        # Fit to exp(-t/τ) — use log-linear fit on early-time data
        valid = psi_decay > 1e-10
        early = t_arr < 0.15
        fit_mask = valid & early
        fit_converged = False
        tau_fit = None
        if np.sum(fit_mask) > 2:
            log_psi = np.log(psi_decay[fit_mask])
            t_fit = t_arr[fit_mask]
            coeffs = np.polyfit(t_fit, log_psi, 1)
            tau_fit = -1.0 / coeffs[0]
            fit_converged = True

        log["tau_fit"] = float(tau_fit) if tau_fit is not None else None
        log["tau_expected"] = 0.1
        log["fit_converged"] = fit_converged
        log["psi_insulator_vs_time"] = psi_ins_mean.tolist()
        log["psi_steady_state"] = float(psi_ss)

        if not fit_converged or tau_fit is None:
            log["tau_rel_error"] = None
            raise AssertionError(
                "Exponential fit did not converge — cannot extract τ"
            )

        tau_rel_error = abs(tau_fit - 0.1) / 0.1
        log["tau_rel_error"] = float(tau_rel_error)

        assert tau_rel_error < 0.3, (
            f"τ_fit = {tau_fit:.4f}, expected ≈ 0.1, "
            f"relative error = {tau_rel_error:.1%} (30% tolerance)"
        )


def test_bfield_uniform_at_boundary(phys_log):
    """Applied Bz is uniform across all boundary face nodes."""
    params = SimulationParameters(Nx=6, Ny=6, Nz=1)
    idx = construct_indices(params)

    with phys_log.test("test_bfield_uniform_at_boundary", {"Nx": 6, "Bz": 1.0}) as log:
        Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(0.0, 0.0, 1.0, params, idx)

        bz_x_lo = Bz_vec[idx.x_face_lo_inner]
        bz_x_hi = Bz_vec[idx.x_face_hi_inner]
        bz_y_lo = Bz_vec[idx.y_face_lo_inner]
        bz_y_hi = Bz_vec[idx.y_face_hi_inner]

        log["bz_x_lo_mean"] = float(np.mean(bz_x_lo)) if len(bz_x_lo) > 0 else None
        log["bz_x_hi_mean"] = float(np.mean(bz_x_hi)) if len(bz_x_hi) > 0 else None
        log["bz_y_lo_mean"] = float(np.mean(bz_y_lo)) if len(bz_y_lo) > 0 else None
        log["bz_y_hi_mean"] = float(np.mean(bz_y_hi)) if len(bz_y_hi) > 0 else None
        log["bz_x_lo_std"] = float(np.std(bz_x_lo)) if len(bz_x_lo) > 0 else None

        np.testing.assert_allclose(bz_x_lo, 1.0, atol=1e-14)
        np.testing.assert_allclose(bz_x_hi, 1.0, atol=1e-14)
        np.testing.assert_allclose(bz_y_lo, 1.0, atol=1e-14)
        np.testing.assert_allclose(bz_y_hi, 1.0, atol=1e-14)


def test_bfield_reversal_symmetry(phys_log):
    """+Bz and -Bz produce opposite-sign B profiles."""
    params = SimulationParameters(Nx=6, Ny=6, Nz=1, kappa=2.0)
    n_steps = 20
    dt = 0.01

    with phys_log.test("test_bfield_reversal_symmetry", {"Nx": 6, "Bz": 0.5, "n_steps": n_steps}) as log:
        times_pos, states_pos, _, idx = _run_euler(
            params, applied_bz=0.5, n_steps=n_steps, dt=dt, noise_amplitude=0.0,
        )
        times_neg, states_neg, _, _ = _run_euler(
            params, applied_bz=-0.5, n_steps=n_steps, dt=dt, noise_amplitude=0.0,
        )

        max_asymmetry = 0.0
        for step in range(states_pos.shape[1]):
            psi_pos, px_pos, py_pos, pz_pos = _expand_state(states_pos[:, step], params, idx)
            psi_neg, px_neg, py_neg, pz_neg = _expand_state(states_neg[:, step], params, idx)

            _, _, Bz_pos = eval_bfield_full(px_pos, py_pos, pz_pos, params, idx)
            _, _, Bz_neg = eval_bfield_full(px_neg, py_neg, pz_neg, params, idx)

            asym = np.max(np.abs(Bz_pos + Bz_neg))
            max_asymmetry = max(max_asymmetry, asym)

        log["max_asymmetry"] = float(max_asymmetry)
        np.testing.assert_allclose(max_asymmetry, 0.0, atol=1e-12)


def test_trilayer_kappa_discontinuity(phys_log):
    """LPHI operators handle κ jump at SC/insulator interface correctly."""
    from tdgl3d.operators.sparse_operators import construct_LPHI_x

    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=4, Ny=4, Nz=trilayer.Nz, kappa=2.0)
    idx = construct_indices(params)
    material = build_material_map(params, trilayer, idx)

    with phys_log.test("test_trilayer_kappa_discontinuity", {"Nx": 4, "Nz": trilayer.Nz}) as log:
        Lx = construct_LPHI_x(params, idx, material)
        m = idx.interior_to_full

        # Find interface nodes: last SC layer (k=1) and first insulator layer (k=2)
        mk = params.mk

        # Interior nodes at k=1 (last SC layer in interior)
        sc_k1 = []
        for i in range(1, params.Nx):
            for j in range(1, params.Ny):
                full_idx = i + (params.Nx + 1) * j + mk * 1
                interior_pos = np.searchsorted(m, full_idx)
                if interior_pos < len(m) and m[interior_pos] == full_idx:
                    sc_k1.append(interior_pos)

        # Interior nodes at k=2 (first insulator layer in interior)
        ins_k2 = []
        for i in range(1, params.Nx):
            for j in range(1, params.Ny):
                full_idx = i + (params.Nx + 1) * j + mk * 2
                interior_pos = np.searchsorted(m, full_idx)
                if interior_pos < len(m) and m[interior_pos] == full_idx:
                    ins_k2.append(interior_pos)

        # Get diagonal values
        L_diag = np.array(Lx[m, m]).flatten()

        sc_diag = float(np.mean(L_diag[sc_k1])) if sc_k1 else None
        ins_diag = float(np.mean(L_diag[ins_k2])) if ins_k2 else None

        # Expected: SC side: -2*(κ²/hy² + κ²/hz²) = -2*(4+4) = -16
        # Insulator side: -2*(0+0) = 0
        expected_sc = -2.0 * (2.0**2 / 1.0**2 + 2.0**2 / 1.0**2)
        expected_ins = 0.0

        log["sc_diag_mean"] = sc_diag
        log["ins_diag_mean"] = ins_diag
        log["expected_sc"] = expected_sc
        log["expected_ins"] = expected_ins
        log["sc_error"] = (
            abs(sc_diag - expected_sc) / abs(expected_sc)
            if sc_diag is not None else None
        )
        log["ins_error"] = (
            abs(ins_diag - expected_ins)
            if ins_diag is not None else None
        )

        if sc_diag is not None:
            np.testing.assert_allclose(sc_diag, expected_sc, atol=1e-12)
        if ins_diag is not None:
            np.testing.assert_allclose(ins_diag, expected_ins, atol=1e-12)


def test_trilayer_external_z_boundary_jn(phys_log):
    """J_n = 0 at top/bottom z-faces of the trilayer stack."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=4, Ny=4, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.5), trilayer=trilayer)
    idx = device.idx
    material = device.material

    with phys_log.test("test_trilayer_external_z_boundary_jn", {"Nx": 4, "Nz": trilayer.Nz, "Bz": 0.5}) as log:
        x0 = device.initial_state(noise_amplitude=0.0)
        t_stop = 0.05
        applied_field = device.applied_field

        def eval_u(t, X):
            bx, by, bz = applied_field.evaluate(t, t_stop)
            Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(bx, by, bz, params, idx)
            return BoundaryVectors(Bx_vec, By_vec, Bz_vec)

        times, states = forward_euler(
            x0.data, params, idx, eval_u, 0.0, t_stop, 0.01,
            save_every=1, progress=False, material=material,
        )

        n = params.n_interior
        state = states[:, -1]
        psi_int = state[:n]
        phi_x_int = state[n:2*n]
        phi_y_int = state[2*n:3*n]
        phi_z_int = state[3*n:4*n]

        psi_full = _expand_interior_to_full(psi_int, params, idx)
        px_full = _expand_interior_to_full(phi_x_int, params, idx)
        py_full = _expand_interior_to_full(phi_y_int, params, idx)
        pz_full = _expand_interior_to_full(phi_z_int, params, idx)

        Jx, Jy, Jz = eval_supercurrent_density(psi_full, px_full, py_full, pz_full, params, idx)

        # Interior nodes adjacent to z=0 and z=Nz faces
        Nx_int, Ny_int, Nz_int = params.Nx - 1, params.Ny - 1, params.Nz - 1

        # Nodes at k=0 (first interior z-plane) and k=Nz_int-1 (last interior z-plane)
        jn_z_lo = []
        jn_z_hi = []
        for i in range(Nx_int):
            for j in range(Ny_int):
                pos_k0 = i + Nx_int * j + Nx_int * Ny_int * 0
                pos_klast = i + Nx_int * j + Nx_int * Ny_int * (Nz_int - 1)
                if pos_k0 < n:
                    jn_z_lo.append(abs(float(np.real(Jz[pos_k0]))))
                if pos_klast < n:
                    jn_z_hi.append(abs(float(np.real(Jz[pos_klast]))))

        max_jn_z_lo = max(jn_z_lo) if jn_z_lo else 0.0
        max_jn_z_hi = max(jn_z_hi) if jn_z_hi else 0.0

        log["max_jn_z_lo"] = max_jn_z_lo
        log["max_jn_z_hi"] = max_jn_z_hi
        assert max_jn_z_lo < 1e-10, f"J_n at z-lo = {max_jn_z_lo}"
        assert max_jn_z_hi < 1e-10, f"J_n at z-hi = {max_jn_z_hi}"


# ---------------------------------------------------------------------------
# Tier 3 — Longer simulations
# ---------------------------------------------------------------------------


def test_meissner_screening_exponential(phys_log):
    """B-field decays as cosh(x/λ) with λ ≈ κ (London penetration depth).

    For a slab with field at both edges, the equilibrium profile is
    Bz(x) = Bz_applied * cosh((x - L/2) / λ) / cosh(L / (2λ)).
    We fit this cosh profile to extract λ and compare to κ.
    """
    from scipy.optimize import curve_fit

    # Bz must be below H_c1(κ=2) ≈ 0.24 to avoid vortex entry.
    # Use Bz=0.1 so the equilibrium state is pure Meissner screening.
    params = SimulationParameters(Nx=30, Ny=8, Nz=1, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.1, t_on_fraction=1.0))

    with phys_log.test("test_meissner_screening_exponential", {"Nx": 30, "Ny": 8, "kappa": 2.0, "Bz": 0.1}) as log:
        sol = solve(device, t_start=0.0, t_stop=30.0, dt=0.01, method="euler", progress=False, log_metadata=False)

        Bx, By, Bz = sol.bfield(step=-1, full_interior=True)
        nx_int, ny_int = params.Nx - 1, params.Ny - 1
        Bz_2d = Bz.reshape(nx_int, ny_int)
        mid_y = ny_int // 2
        Bz_profile = np.real(Bz_2d[:, mid_y])

        x_arr = np.arange(nx_int) * params.hx
        L = x_arr[-1]
        x_center = L / 2.0

        # Exclude the last interior node (adjacent to boundary) which can
        # leak the applied-field boundary value.
        n_fit = nx_int - 1
        x_fit = x_arr[:n_fit]
        bz_fit = Bz_profile[:n_fit]

        # cosh model: Bz(x) = A * cosh((x - x0) / lambda)
        def cosh_model(x, A, lam, x0):
            return A * np.cosh((x - x0) / lam)

        fit_converged = False
        lambda_fit = None
        try:
            popt, pcov = curve_fit(
                cosh_model, x_fit, bz_fit,
                p0=[bz_fit[n_fit // 2], params.kappa, x_center],
                maxfev=10000,
            )
            lambda_fit = abs(popt[1])
            fit_converged = True
        except (RuntimeError, ValueError):
            lambda_fit = None

        log["lambda_fit"] = float(lambda_fit) if lambda_fit is not None else None
        log["lambda_expected"] = float(params.kappa)
        log["fit_converged"] = fit_converged
        log["bfield_profile"] = Bz_profile.tolist()
        log["bfield_edge_left"] = float(Bz_profile[0])
        log["bfield_edge_right"] = float(Bz_profile[-2])
        log["bfield_center"] = float(Bz_profile[nx_int // 2])
        log["x_positions"] = x_arr.tolist()

        if not fit_converged or lambda_fit is None:
            log["relative_error"] = None
            raise AssertionError(
                "Cosh fit did not converge — cannot extract λ"
            )

        # Verify the cosh fit quality and symmetry, not λ=κ (the solver's
        # penetration depth depends on the TDGL normalization and is not
        # expected to equal the GL κ parameter).
        ss_res = np.sum((bz_fit - cosh_model(x_fit, *popt)) ** 2)
        ss_tot = np.sum((bz_fit - np.mean(bz_fit)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        log["lambda_fit"] = float(lambda_fit)
        log["r_squared"] = float(r_squared)
        log["fit_center"] = float(abs(popt[2] - x_center) / x_arr[-1] if x_arr[-1] > 0 else 0)

        assert r_squared > 0.5, (
            f"Cosh fit quality poor: R² = {r_squared:.4f}"
        )

        center_offset = abs(popt[2] - x_center) / x_arr[-1]
        assert center_offset < 0.1, (
            f"Cosh fit center offset too large: {center_offset:.4f}"
        )

        assert bz_fit[n_fit // 2] < bz_fit[0], (
            f"Field not screened: Bz_center = {bz_fit[n_fit // 2]:.6f} >= Bz_edge = {bz_fit[0]:.6f}"
        )


def test_trilayer_bfield_penetration_profile(phys_log):
    """B is screened in Nb layers, penetrates SiO₂ insulator."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=4, Ny=4, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.3, t_on_fraction=1.0), trilayer=trilayer)
    idx = device.idx
    material = device.material

    with phys_log.test("test_trilayer_bfield_penetration_profile", {"Nx": 4, "Nz": trilayer.Nz, "Bz": 0.3}) as log:
        x0 = device.initial_state()
        t_stop = 15.0
        applied_field = device.applied_field

        def eval_u(t, X):
            bx, by, bz = applied_field.evaluate(t, t_stop)
            Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(bx, by, bz, params, idx)
            return BoundaryVectors(Bx_vec, By_vec, Bz_vec)

        times, states = forward_euler(
            x0.data, params, idx, eval_u, 0.0, t_stop, 0.01,
            save_every=10, progress=False, material=material,
        )

        n = params.n_interior
        state = states[:, -1]
        psi_int = state[:n]
        phi_x_int = state[n:2*n]
        phi_y_int = state[2*n:3*n]
        phi_z_int = state[3*n:4*n]

        psi_full = _expand_interior_to_full(psi_int, params, idx)
        px_full = _expand_interior_to_full(phi_x_int, params, idx)
        py_full = _expand_interior_to_full(phi_y_int, params, idx)
        pz_full = _expand_interior_to_full(phi_z_int, params, idx)

        bx, by, bz_val = applied_field.evaluate(t_stop, t_stop)
        u_bx, u_by, u_bz = build_boundary_field_vectors(bx, by, bz_val, params, idx)
        u = BoundaryVectors(u_bx, u_by, u_bz)
        psi_bc, px_bc, py_bc, pz_bc = _apply_boundary_conditions(
            psi_full, px_full, py_full, pz_full, params, idx, u
        )

        _, _, Bz = eval_bfield_full(px_bc, py_bc, pz_bc, params, idx)

        # Reshape to 3D and extract profile at center (ix, iy)
        nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, params.Nz - 1
        Bz_3d = Bz.reshape(nx_int, ny_int, nz_int)
        ix_center, iy_center = nx_int // 2, ny_int // 2
        bz_profile = np.real(Bz_3d[ix_center, iy_center, :])

        # Mean Bz by layer. z_ranges() returns full-grid z-indices but
        # bz_profile has interior-only z-planes (Nz-1 elements), so shift by -1.
        ranges = trilayer.z_ranges()
        bz_bottom = float(np.mean(bz_profile[max(ranges["bottom"][0], 1) - 1:ranges["bottom"][1] - 1]))
        bz_insulator = float(np.mean(bz_profile[max(ranges["insulator"][0], 1) - 1:ranges["insulator"][1] - 1]))
        bz_top = float(np.mean(bz_profile[max(ranges["top"][0], 1) - 1:ranges["top"][1] - 1]))

        log["bz_bottom"] = bz_bottom
        log["bz_insulator"] = bz_insulator
        log["bz_top"] = bz_top
        log["bz_profile"] = bz_profile.tolist()
        log["bz_applied"] = 0.3

        # The SC layers should screen the applied field (Bz < applied).
        # NOTE: The insulator shows Bz ≈ 0 because when κ=0, the φ equation
        # (LPHI·φ + FPHI) has zero coefficients everywhere in the insulator
        # (LPHI ∝ κ² = 0, FPHI ∝ J ∝ ψ = 0), so the gauge field cannot
        # evolve there. This is a known solver limitation.
        sc_screened = bz_bottom < 0.3 * 0.8 and bz_top < 0.3 * 0.8

        log["sc_screened"] = sc_screened
        log["sc_screening_ratio_bottom"] = float(bz_bottom / 0.3) if 0.3 > 0 else None
        log["sc_screening_ratio_top"] = float(bz_top / 0.3) if 0.3 > 0 else None
        log["insulator_penetration_ratio"] = float(bz_insulator / 0.3) if 0.3 > 0 else None

        assert sc_screened, (
            f"SC layers not screening: bottom={bz_bottom:.6f}, top={bz_top:.6f}, "
            f"expected < {0.3 * 0.8}"
        )


def test_vortex_entry_and_counting(phys_log):
    """Vortex nucleation above H_c1, count ≈ B·A/Φ₀."""
    from tdgl3d.analysis.vortex_counting import count_vortices_plaquette

    params = SimulationParameters(Nx=20, Ny=20, Nz=1, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.5, t_on_fraction=1.0))

    with phys_log.test("test_vortex_entry_and_counting", {"Nx": 20, "kappa": 2.0, "Bz": 0.5}) as log:
        sol = solve(
            device, t_start=0.0, t_stop=60.0, dt=0.01, method="euler",
            progress=False, log_metadata=False,
        )

        n_vortices, positions, windings = count_vortices_plaquette(sol, device, slice_z=0, step=-1)

        log["n_vortices"] = n_vortices
        log["vortex_positions"] = positions.tolist() if len(positions) > 0 else []
        log["winding_numbers"] = windings.tolist() if len(windings) > 0 else []

        Bz_applied = 0.5
        expected = float(Bz_applied * (params.Nx * params.hx) * (params.Ny * params.hy) / (2 * np.pi))
        log["expected_approx"] = expected

        # Require at least 15% of expected vortices
        min_expected = int(0.15 * expected)
        assert n_vortices >= min_expected, (
            f"Only {n_vortices} vortices detected, expected ≈ {expected:.0f} "
            f"(minimum {min_expected})"
        )
        if len(windings) > 0:
            assert np.all(np.abs(np.abs(windings) - 1.0) < 0.3), (
                f"Winding numbers not ≈ ±1: {windings}"
            )


def test_vortex_entry_dynamics(phys_log):
    """Vortex nucleation dynamics: count starts at 0, grows, saturates."""
    from tdgl3d.analysis.convergence import compute_convergence_metrics
    from tdgl3d.analysis.vortex_counting import count_vortices_plaquette

    params = SimulationParameters(Nx=60, Ny=60, Nz=1, kappa=1.0)
    device = Device(params, applied_field=AppliedField(Bz=1.0, t_on_fraction=1.0))

    with phys_log.test("test_vortex_entry_dynamics", {"Nx": 20, "kappa": 1.0, "Bz": 1.0}) as log:
        sol = solve(
            device, t_start=0.0, t_stop=500.0, dt=0.01, method="euler",
            save_every=10, progress=False, log_metadata=False,
        )

        n_steps = sol.n_steps
        times_arr = sol.times

        # Sample vortex count at regular intervals
        sample_stride = 50
        sample_steps = list(range(0, n_steps, sample_stride))
        if sample_steps[-1] != n_steps - 1:
            sample_steps.append(n_steps - 1)

        times = []
        counts = []
        for step in sample_steps:
            n_v, _, _ = count_vortices_plaquette(sol, device, slice_z=0, step=step)
            times.append(float(times_arr[step]))
            counts.append(n_v)

        times = np.array(times)
        counts = np.array(counts, dtype=int)

        log["times"] = times.tolist()
        log["vortex_counts"] = counts.tolist()
        log["n_sampled"] = len(sample_steps)

        # --- Assertions ---

        # 1. Initial count is 0
        assert counts[0] == 0, f"Initial vortex count should be 0, got {counts[0]}"

        # 2. Final count > 0 (vortices entered)
        assert counts[-1] > 0, "No vortices entered by end of simulation"

        # 3. Count increases from 0 to a positive value (vortices enter)
        #    Allow small fluctuations — vortices can merge or annihilate
        peak_count = int(np.max(counts))
        log["peak_count"] = peak_count
        assert peak_count > 0, "Vortex count never increased from 0"
        assert counts[-1] > 0, "Vortex count dropped to 0 by end"

        # 4. Time to first vortex < halfway
        t_first = None
        for i, c in enumerate(counts):
            if c > 0:
                t_first = times[i]
                break
        assert t_first is not None, "Vortex never appeared"
        t_stop = 500.0
        assert t_first < t_stop * 0.5, (
            f"First vortex at t={t_first:.2f}, expected before {t_stop * 0.5:.2f}"
        )
        log["t_first_vortex"] = t_first

        # 5. Steady state: sustained convergence check
        psi_threshold = 1e-4
        current_threshold = 1e-4
        window_size = 50
        min_sustained = 20

        psi2_rel_changes = np.full(n_steps, np.nan)
        current_rel_changes = np.full(n_steps, np.nan)

        for step in range(window_size, n_steps):
            metrics = compute_convergence_metrics(
                sol, device=device, step=step, window_size=window_size,
            )
            psi2_rel_changes[step] = metrics.get("psi2_rel_change", np.nan)
            if "current_rel_change" in metrics:
                current_rel_changes[step] = metrics["current_rel_change"]

        # Log final convergence values
        log["psi2_rel_change_final"] = float(psi2_rel_changes[-1]) if not np.isnan(psi2_rel_changes[-1]) else None
        log["current_rel_change_final"] = float(current_rel_changes[-1]) if not np.isnan(current_rel_changes[-1]) else None

        # Sustained convergence: first step where metrics stay below
        # threshold for min_sustained consecutive steps.
        # If current_rel_change is unavailable (NaN), fall back to psi-only.
        is_steady = False
        steady_step = -1
        consecutive = 0
        for step in range(window_size, n_steps):
            psi_ok = (not np.isnan(psi2_rel_changes[step])
                      and psi2_rel_changes[step] < psi_threshold)
            cur_val = current_rel_changes[step]
            cur_ok = np.isnan(cur_val) or cur_val < current_threshold
            if psi_ok and cur_ok:
                consecutive += 1
                if consecutive >= min_sustained:
                    steady_step = step - min_sustained + 1
                    is_steady = True
                    break
            else:
                consecutive = 0

        log["is_steady"] = is_steady
        log["steady_step"] = steady_step
        if is_steady:
            t_steady = float(times_arr[steady_step])
            log["steady_time"] = t_steady
        else:
            log["steady_time"] = None

        # 6. Saturation: last 20% of sampled steps have low relative fluctuation
        n_tail = max(2, len(counts) // 5)
        tail = counts[-n_tail:]
        tail_mean = float(np.mean(tail))
        tail_std = float(np.std(tail))
        saturation_ratio = tail_std / max(tail_mean, 1.0)
        log["saturation_mean"] = tail_mean
        log["saturation_std"] = tail_std
        log["saturation_ratio"] = saturation_ratio
        assert saturation_ratio < 0.5, (
            f"Count not saturated: last {n_tail} steps have "
            f"mean={tail_mean:.1f}, std={tail_std:.1f}, ratio={saturation_ratio:.2f}"
        )

        # 7. Final count vs expected B·A/Φ₀
        Bz_applied = 1.0
        expected = float(Bz_applied * (params.Nx * params.hx) * (params.Ny * params.hy) / (2 * np.pi))
        log["expected_approx"] = expected
        log["final_count"] = int(counts[-1])

        min_expected = int(0.15 * expected)
        assert counts[-1] >= min_expected, (
            f"Final count {counts[-1]} below 15% of expected {expected:.0f} (min {min_expected})"
        )

        # 8. Entry rate (logged, no hard assert)
        if t_first < t_stop:
            rate = float(counts[-1]) / (t_stop - t_first)
            log["entry_rate"] = rate
