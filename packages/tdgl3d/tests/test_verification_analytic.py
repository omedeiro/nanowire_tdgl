"""Verification against analytically known Ginzburg-Landau results.

Units: lengths in ξ, fields in Φ₀/(2πξ²).  In this normalisation

* the London penetration depth is ``λ = κ`` (so the screened field decays as
  ``e^{-x/κ}`` into the bulk),
* the flux quantum is ``Φ₀ = 2π``,
* the lowest Landau level of the covariant Laplacian is ``E₀ = B``, which places
  the bulk nucleation field at ``B_c2 = 1``.

Each test drives one of these to a *number* rather than to a shape.  Where a
discretisation error is unavoidable it is bounded by an explicit tolerance and,
where cheap, its convergence with ``h`` or ``dt`` is measured as well: an
observed order of accuracy is a much stronger statement than a single loose
comparison, because a wrong operator with the right magnitude still gets the
order wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d.core.state import StateVector
from tdgl3d.operators.sparse_operators import (
    construct_LPSI_x,
    construct_LPSI_y,
)
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.rhs import eval_f
from tdgl3d.solvers.integrators import forward_euler

from .physics_helpers import (
    applied_boundary,
    cfl_limit,
    expand_state,
    landau_gauge_links,
    make_grid,
    run_euler,
    zero_boundary,
)


def _covariant_laplacian(params, idx, phi_x, phi_y):
    """Interior block of ``-(∇ - iA)²`` assembled from the solver's own operators."""
    operator = (
        construct_LPSI_x(phi_x, params, idx) / params.hx**2
        + construct_LPSI_y(phi_y, params, idx) / params.hy**2
    ).tocsr()
    m = idx.interior_to_full
    return (-operator[m, :][:, m]).tocsc()


def _edge_decay_length(profile, h):
    """Fit ``ln B`` against distance over the outer half of a screening profile."""
    n_fit = max(4, len(profile) // 2)
    y = np.log(np.abs(profile[:n_fit]))
    x = np.arange(n_fit) * h
    slope = np.polyfit(x, y, 1)[0]
    return -1.0 / slope


# ---------------------------------------------------------------------------
# Meissner screening: λ = κ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [1.5, 3.0])
def test_london_penetration_depth_equals_kappa(kappa, phys_log):
    """The screened field decays with λ = κ, the London depth in ξ units.

    The sample must be several λ across in *both* in-plane directions and the
    grid must resolve λ, otherwise the measured decay is dominated by the finite
    size and the discrete stencil rather than by the physics.  The applied field
    is kept below H_c1 so the equilibrium state is pure Meissner screening.
    """
    h = 0.5
    n_cells = int(round(8.0 * kappa / h))
    params = SimulationParameters(
        Nx=n_cells, Ny=n_cells, Nz=1, hx=h, hy=h, kappa=kappa,
    )
    bz = 0.02
    dt = 0.8 * cfl_limit(params)
    n_steps = int(round(12.0 / dt))
    _, states, _, idx = run_euler(
        params, bz, n_steps=n_steps, dt=dt, noise_amplitude=0.0, save_every=n_steps // 4,
    )

    _, px, py, pz = expand_state(states[:, -1], params, idx, applied_boundary(params, idx, bz=bz))
    nx_int, ny_int = params.Nx - 1, params.Ny - 1
    field = eval_bfield_full(px, py, pz, params, idx)[2].reshape(nx_int, ny_int)

    # Profile inward from the low-x edge along the mid-line, skipping the pinned
    # boundary plaquette, over a window of two penetration depths.
    window = int(round(2.0 * kappa / h))
    profile = field[:window, ny_int // 2]
    lam = _edge_decay_length(profile, h)

    drift = float(np.max(np.abs(states[:, -1] - states[:, -2])))
    psi_min = float(np.min(np.abs(states[: params.n_interior, -1])))

    with phys_log.test(
        f"test_london_penetration_depth_equals_kappa[kappa={kappa}]",
        {"Nx": n_cells, "h": h, "kappa": kappa, "Bz": bz, "L_over_lambda": 8.0},
        "λ = κ in these units — the field decays as exp(-x/κ) into the bulk",
    ) as log:
        log["lambda_fit"] = float(lam)
        log["profile"] = [float(v) for v in profile]
        log["state_drift"] = drift
        log.check_below("state drift (equilibrium reached)", drift, 1e-6)
        log.check_above(
            "min |ψ| (still Meissner, no vortices)", psi_min, 0.9,
        )
        log.check_close(
            "λ from the screening profile", lam, kappa, rtol=0.10, units="ξ",
            detail="London penetration depth must equal the GL parameter κ",
        )


def test_penetration_depth_converges_with_grid_refinement(phys_log):
    """Refining h moves the measured λ towards κ, confirming it is the physics.

    A screening length set by the discretisation rather than by the physics would
    not improve when the same geometry is resolved more finely.
    """
    kappa, length = 2.0, 16.0
    results = {}
    for h in (1.0, 0.5):
        n_cells = int(round(length / h))
        params = SimulationParameters(Nx=n_cells, Ny=n_cells, Nz=1, hx=h, hy=h, kappa=kappa)
        bz = 0.02
        dt = 0.8 * cfl_limit(params)
        n_steps = int(round(12.0 / dt))
        _, states, _, idx = run_euler(
            params, bz, n_steps=n_steps, dt=dt, noise_amplitude=0.0, save_every=n_steps,
        )
        _, px, py, pz = expand_state(
            states[:, -1], params, idx, applied_boundary(params, idx, bz=bz)
        )
        nx_int, ny_int = params.Nx - 1, params.Ny - 1
        field = eval_bfield_full(px, py, pz, params, idx)[2].reshape(nx_int, ny_int)
        window = int(round(2.0 * kappa / h))
        results[h] = _edge_decay_length(field[:window, ny_int // 2], h)

    err_coarse = abs(results[1.0] - kappa)
    err_fine = abs(results[0.5] - kappa)

    with phys_log.test(
        "test_penetration_depth_converges_with_grid_refinement",
        {"L": length, "kappa": kappa, "h_values": [1.0, 0.5]},
        "the measured λ must approach κ as the grid is refined",
    ) as log:
        log["lambda_h1.0"] = float(results[1.0])
        log["lambda_h0.5"] = float(results[0.5])
        log.check_below("|λ − κ| at h = 1.0", err_coarse, 0.35 * kappa)
        log.check_below("|λ − κ| at h = 0.5", err_fine, 0.10 * kappa)
        log.check_below(
            "error ratio fine/coarse", err_fine / max(err_coarse, 1e-12), 0.75,
            detail="refinement must reduce the discretisation error",
        )


# ---------------------------------------------------------------------------
# Landau levels and H_c2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bz", [0.1, 0.2])
def test_lowest_landau_level_of_covariant_laplacian(bz, phys_log):
    """The Peierls phases reproduce the lowest Landau level E₀ = B.

    ``-(∇ - iA)²`` in a uniform field has spectrum ``(2n+1)B``.  Because the
    linearised GL equation is ``∂ψ/∂t = [1 - (∇ - iA)²]ψ``, ``E₀ = B`` is exactly
    the statement that bulk superconductivity survives up to ``B_c2 = 1`` in
    these units.  The check is a direct eigenvalue computation on the solver's
    own operator: it needs no time integration and cannot be satisfied by a
    stencil that encodes the wrong flux.
    """
    h, length = 0.5, 24.0
    n_cells = int(round(length / h))
    params, idx = make_grid(Nx=n_cells, Ny=n_cells, Nz=1, hx=h, hy=h, kappa=2.0)
    phi_x, phi_y = landau_gauge_links(params, bz)

    curl = eval_bfield_full(phi_x, phi_y, np.zeros(params.dim_x, np.complex128), params, idx)[2]
    operator = _covariant_laplacian(params, idx, phi_x, phi_y)
    hermiticity = float(abs(operator - operator.getH()).max())
    e0 = float(eigsh(operator, k=1, which="SA", return_eigenvectors=False, tol=1e-10)[0])

    with phys_log.test(
        f"test_lowest_landau_level_of_covariant_laplacian[Bz={bz}]",
        {"L": length, "h": h, "Bz": bz},
        "the covariant Laplacian's ground-state energy is the lowest Landau level",
    ) as log:
        log["E0"] = e0
        log["B_c2_implied"] = float(bz / e0) if e0 else None
        log.check_below(
            "max|∇×A − B| for the Landau-gauge links",
            float(np.max(np.abs(curl - bz))), 1e-13,
            detail="the test field must really be uniform before the spectrum means anything",
        )
        log.check_below("non-Hermiticity of −(∇ − iA)²", hermiticity, 1e-13)
        log.check_close("lowest eigenvalue E₀", e0, bz, rtol=0.03,
                        detail="E₀ = B places H_c2 at B = 1")


def test_covariant_laplacian_reduces_to_the_standard_laplacian(phys_log):
    """At A = 0 the covariant stencil is the ordinary 5-point Laplacian."""
    params, idx = make_grid(Nx=9, Ny=7, Nz=1, hx=0.4, hy=0.6, kappa=2.0)
    zero = np.zeros(params.dim_x, dtype=np.complex128)
    covariant = _covariant_laplacian(params, idx, zero, zero)

    n_int, m = params.n_interior, idx.interior_to_full
    lookup = {int(node): pos for pos, node in enumerate(m)}
    rows, cols, vals = [], [], []
    for pos, node in enumerate(m):
        rows.append(pos)
        cols.append(pos)
        vals.append(2.0 / params.hx**2 + 2.0 / params.hy**2)
        for offset, hh in ((1, params.hx), (-1, params.hx),
                           (params.mj, params.hy), (-params.mj, params.hy)):
            neighbour = lookup.get(int(node) + offset)
            if neighbour is not None:
                rows.append(pos)
                cols.append(neighbour)
                vals.append(-1.0 / hh**2)
    reference = sp.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))

    difference = float(abs(covariant - reference).max())

    with phys_log.test(
        "test_covariant_laplacian_reduces_to_the_standard_laplacian",
        {"Nx": 9, "Ny": 7, "hx": 0.4, "hy": 0.6},
        "the Peierls factors must be the only difference from the plain Laplacian",
    ) as log:
        log["operator_scale"] = float(abs(reference).max())
        log.check_below("max|L_covariant(A=0) − L_standard|", difference, 1e-13)


def test_covariant_laplacian_is_second_order_accurate(phys_log):
    """Manufactured solution: the truncation error falls as h².

    ``ψ = exp(i k·x)`` with ``A = 0`` has ``∇²ψ = -|k|²ψ`` exactly, so the
    discrete operator's error against that value measures its order.
    """
    k = np.array([0.7, 0.4])
    errors = {}
    for h in (0.4, 0.2, 0.1):
        n_cells = int(round(4.0 / h))
        params, idx = make_grid(Nx=n_cells, Ny=n_cells, Nz=1, hx=h, hy=h, kappa=2.0)
        zero = np.zeros(params.dim_x, dtype=np.complex128)
        m = idx.interior_to_full
        xs = (m % params.mj) * h
        ys = (m // params.mj) * h
        psi = np.exp(1j * (k[0] * xs + k[1] * ys))

        operator = _covariant_laplacian(params, idx, zero, zero)
        # Interior nodes whose neighbours are also interior (no boundary truncation).
        lookup = {int(node): pos for pos, node in enumerate(m)}
        bulk = [
            pos for pos, node in enumerate(m)
            if all(int(node) + off in lookup for off in (1, -1, params.mj, -params.mj))
        ]
        residual = -(operator @ psi) + (k @ k) * psi
        errors[h] = float(np.max(np.abs(residual[bulk])))

    hs = np.array(sorted(errors))
    order = float(
        np.polyfit(np.log(hs), np.log([errors[h] for h in hs]), 1)[0]
    )

    with phys_log.test(
        "test_covariant_laplacian_is_second_order_accurate",
        {"k": k.tolist(), "h_values": [0.4, 0.2, 0.1]},
        "the discrete Laplacian must converge at second order",
    ) as log:
        log["errors"] = {str(h): errors[h] for h in hs}
        log.check_close("observed order of accuracy", order, 2.0, atol=0.15)
        log.check_below("error at h = 0.1", errors[0.1], 1e-3)


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------


def test_forward_euler_is_first_order_in_dt(phys_log):
    """Halving dt halves the error: the scheme is O(dt), as advertised.

    The error is estimated by Richardson differencing, ``e(dt) = |X(dt) −
    X(dt/2)|``, rather than against a fixed fine reference: with a reference of
    comparable size the ``dt − dt_ref`` offset biases the fitted order upwards.
    """
    params, idx = make_grid(Nx=8, Ny=7, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.3, t_on_fraction=1.0))
    boundary = applied_boundary(params, idx, bz=0.3)
    t_stop = 0.5
    x0 = device.initial_state(noise_amplitude=0.05, seed=6).data

    def integrate(dt):
        _, states = forward_euler(
            x0, params, idx, lambda t, X: boundary, 0.0, t_stop, dt,
            save_every=10**9, progress=False,
        )
        return states[:, -1]

    base = cfl_limit(params) / 2.0
    steps = [base / 2**k for k in range(5)]
    finals = {dt: integrate(dt) for dt in steps}
    errors = {
        steps[k]: float(np.max(np.abs(finals[steps[k]] - finals[steps[k + 1]])))
        for k in range(len(steps) - 1)
    }

    dts = np.array(sorted(errors))
    order = float(np.polyfit(np.log(dts), np.log([errors[d] for d in dts]), 1)[0])

    with phys_log.test(
        "test_forward_euler_is_first_order_in_dt",
        {"Nx": 8, "Ny": 7, "t_stop": t_stop, "dt_max": base},
        "explicit Euler must show first-order global convergence",
    ) as log:
        log["richardson_errors"] = {f"{d:.3e}": errors[d] for d in dts}
        log.check_close("observed order in dt", order, 1.0, atol=0.1)
        log.check_below("Richardson error at the smallest dt", errors[dts[0]], 1e-3)


def test_trapezoidal_agrees_with_euler_in_the_small_dt_limit(phys_log):
    """The implicit integrator must reproduce the explicit one's trajectory.

    Two integrators of the same right-hand side agreeing to five digits is a
    strong statement that the right-hand side, its Jacobian-vector products and
    the Newton-GCR solve are all consistent with each other.
    """
    from tdgl3d.solvers.integrators import trapezoidal

    params, idx = make_grid(Nx=5, Ny=4, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.25, t_on_fraction=1.0))
    boundary = applied_boundary(params, idx, bz=0.25)
    t_stop = 0.1
    x0 = device.initial_state(noise_amplitude=0.05, seed=8).data
    limit = cfl_limit(params)

    _, euler_states = forward_euler(
        x0, params, idx, lambda t, X: boundary, 0.0, t_stop, limit / 64.0,
        save_every=10**9, progress=False,
    )
    # The Jacobian-vector products are finite differences, so the Newton-GCR
    # tolerances cannot usefully be pushed below the differencing noise floor.
    _, trap_states = trapezoidal(
        x0, params, idx, lambda t, X: boundary, 0.0, t_stop, limit / 2.0,
        newton_tol_f=1e-5, newton_tol_dx=1e-5, tol_gcr=1e-5, eps_mf=1e-6,
        save_every=10**9, progress=False,
    )
    scale = float(np.max(np.abs(euler_states[:, -1])))
    difference = float(np.max(np.abs(trap_states[:, -1] - euler_states[:, -1])))

    with phys_log.test(
        "test_trapezoidal_agrees_with_euler_in_the_small_dt_limit",
        {"Nx": 5, "Ny": 4, "t_stop": t_stop, "Bz": 0.25},
        "two independent integrators of the same right-hand side must agree",
    ) as log:
        log["state_scale"] = scale
        log["relative_difference"] = difference / scale
        log.check_below("max|X_trapezoidal − X_euler| / |X|", difference / scale, 1e-3)


# ---------------------------------------------------------------------------
# Material response
# ---------------------------------------------------------------------------


def test_insulator_order_parameter_decays_with_the_stated_time_constant(phys_log):
    """In an insulator layer ψ relaxes as exp(−t/τ) with the τ used by FPSI."""
    from tdgl3d.operators.sparse_operators import INSULATOR_RELAXATION_TIME

    tau_expected = INSULATOR_RELAXATION_TIME
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=4, Ny=4, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.0), trilayer=trilayer)
    idx, material = device.idx, device.material

    x0 = StateVector.uniform_superconducting(params).data
    times, states = forward_euler(
        x0, params, idx, lambda t, X: zero_boundary(params), 0.0, 0.5, 0.005,
        save_every=1, progress=False, material=material,
    )

    insulator = material.interior_sc_mask == 0.0
    n = params.n_interior
    trace = np.array(
        [float(np.mean(np.abs(states[:n, s][insulator]))) for s in range(states.shape[1])]
    )
    decaying = trace - trace[-1]
    window = (times < 0.15) & (decaying > 1e-9)
    tau_fit = -1.0 / np.polyfit(times[window], np.log(decaying[window]), 1)[0]

    with phys_log.test(
        "test_insulator_order_parameter_decays_with_the_stated_time_constant",
        {"Nz": trilayer.Nz, "tau_expected": tau_expected},
        "the insulator relaxation term must act on its documented time scale",
    ) as log:
        log["tau_fit"] = float(tau_fit)
        log["psi_steady_state"] = float(trace[-1])
        log.check_above("points used in the fit", float(window.sum()), 5.0)
        log.check_close("fitted τ", float(tau_fit), tau_expected, rtol=0.2, units="τ_GL")
        log.check_below(
            "residual |ψ| in the insulator", float(trace[-1]), 0.15,
            detail="proximity leakage from the neighbouring superconductors",
        )


def test_zero_field_ground_state_is_the_uniform_condensate(phys_log):
    """From noise at B = 0 the solver relaxes to |ψ| = 1 with no residual field."""
    params = SimulationParameters(Nx=10, Ny=8, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    dt = 0.8 * cfl_limit(params)
    _, states, _, idx = run_euler(
        params, 0.0, n_steps=int(round(40.0 / dt)), dt=dt, noise_amplitude=0.3, seed=15,
        save_every=10**9,
    )
    n = params.n_interior
    psi_abs = np.abs(states[:n, -1])
    _, px, py, pz = expand_state(states[:, -1], params, idx)
    field = eval_bfield_full(px, py, pz, params, idx)[2]
    residual = float(np.max(np.abs(eval_f(states[:, -1], params, idx, zero_boundary(params)))))

    with phys_log.test(
        "test_zero_field_ground_state_is_the_uniform_condensate",
        {"Nx": 10, "Ny": 8, "h": 0.5, "kappa": 2.0},
        "|ψ| = 1 minimises −|ψ|² + ½|ψ|⁴; the ground state must reach it",
    ) as log:
        log["psi_min"] = float(psi_abs.min())
        log["psi_max"] = float(psi_abs.max())
        log.check_close("min |ψ|", float(psi_abs.min()), 1.0, atol=1e-4)
        log.check_close("max |ψ|", float(psi_abs.max()), 1.0, atol=1e-4)
        log.check_below("max |B| in the relaxed state", float(np.max(np.abs(field))), 1e-6)
        log.check_below("max |dX/dt| at the fixed point", residual, 1e-4)
