"""Gauge-invariance verification for the TDGL solver.

Local U(1) gauge symmetry is the structural backbone of Ginzburg-Landau theory:
the physics may not depend on the choice of ``χ`` in

    ψ → ψ e^{iχ},   A → A + ∇χ,

which on the lattice reads ``φ_μ → φ_μ + (χ_{m+e_μ} - χ_m)`` for the link
variables.  Two consequences are checked here:

* **Covariance of the dynamics.**  ``eval_f`` must satisfy
  ``F(G·X) = G·F(X)``: the ψ-component rotates by ``e^{iχ}`` and the link
  components are unchanged.
* **Invariance of observables.**  ``|ψ|``, ``B = ∇×A``, the supercurrent, the
  free energy and the vortex count must not move at all.

These are algebraic identities, so the tolerances are at round-off level.  They
are the sharpest tests in the suite: a mismatched Peierls phase between the
covariant Laplacian and the supercurrent source — which still screens correctly
and still nucleates vortices — shows up here as an O(1) violation.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer, solve
from tdgl3d.analysis.vortex_counting import count_vortices_plaquette, plaquette_vorticity
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.current_density import eval_supercurrent_density
from tdgl3d.physics.free_energy import gl_free_energy
from tdgl3d.physics.rhs import eval_f

from .physics_helpers import (
    applied_boundary,
    expand_state,
    gauge_transform,
    make_grid,
    random_state,
    smooth_gauge_field,
    zero_boundary,
)

GRIDS_2D = [dict(Nx=10, Ny=10, Nz=1, kappa=2.0), dict(Nx=9, Ny=7, Nz=1, kappa=3.0)]
GRIDS_3D = [dict(Nx=6, Ny=7, Nz=5, kappa=2.0)]


# ---------------------------------------------------------------------------
# Covariance of the right-hand side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid", GRIDS_2D + GRIDS_3D, ids=lambda g: f"{g['Nx']}x{g['Ny']}x{g['Nz']}")
def test_rhs_is_gauge_covariant(grid, phys_log):
    """eval_f(G·X) = G·eval_f(X) for a local gauge transformation G."""
    params, idx = make_grid(**grid)
    n = params.n_interior
    state = random_state(params, seed=11)
    chi = smooth_gauge_field(params, seed=12)
    state_g = gauge_transform(state, chi, params, idx)
    u = zero_boundary(params)

    f0 = eval_f(state, params, idx, u)
    fg = eval_f(state_g, params, idx, u)

    phase = np.exp(1j * chi[idx.interior_to_full])
    psi_err = float(np.max(np.abs(fg[:n] - f0[:n] * phase)))
    phi_err = float(np.max(np.abs(fg[n:] - f0[n:])))
    scale = float(max(np.max(np.abs(f0)), 1.0))

    name = f"test_rhs_is_gauge_covariant[{params.Nx}x{params.Ny}x{params.Nz}]"
    with phys_log.test(
        name, dict(grid),
        "dψ/dt must rotate with the gauge phase and dφ/dt must be invariant",
    ) as log:
        log["rhs_scale"] = scale
        log["gauge_amplitude"] = float(np.max(np.abs(chi)))
        log.check_below(
            "max|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)|", psi_err, 1e-11 * scale,
            detail="ψ-equation must be covariant under ψ→ψe^{iχ}, φ→φ+Δχ",
        )
        log.check_below(
            "max|dφ/dt(GX) − dφ/dt(X)|", phi_err, 1e-11 * scale,
            detail="the supercurrent source and curl-curl term must be gauge invariant",
        )


def test_rhs_covariant_with_material_map(phys_log):
    """Gauge covariance survives a trilayer material map (κ jump, insulator)."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=6, Ny=5, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.0), trilayer=trilayer)
    idx, material = device.idx, device.material

    n = params.n_interior
    state = random_state(params, seed=21)
    chi = smooth_gauge_field(params, seed=22)
    state_g = gauge_transform(state, chi, params, idx)
    u = zero_boundary(params)

    f0 = eval_f(state, params, idx, u, material=material)
    fg = eval_f(state_g, params, idx, u, material=material)
    phase = np.exp(1j * chi[idx.interior_to_full])
    err = max(
        float(np.max(np.abs(fg[:n] - f0[:n] * phase))),
        float(np.max(np.abs(fg[n:] - f0[n:]))),
    )
    scale = float(max(np.max(np.abs(f0)), 1.0))

    with phys_log.test(
        "test_rhs_covariant_with_material_map",
        {"Nz": trilayer.Nz, "kappa_sc": 2.0},
        "a spatially varying κ and an insulator mask must not break gauge covariance",
    ) as log:
        log["rhs_scale"] = scale
        log.check_below("max covariance violation", err, 1e-11 * scale)


def test_global_phase_rotation_is_exact_symmetry(phys_log):
    """A constant phase ψ → ψ e^{iα} leaves the link equations untouched."""
    params, idx = make_grid(Nx=8, Ny=8, Nz=1, kappa=2.0)
    n = params.n_interior
    state = random_state(params, seed=31)
    u = applied_boundary(params, idx, bz=0.4)

    alpha = 0.7
    rotated = np.array(state, dtype=np.complex128, copy=True)
    rotated[:n] *= np.exp(1j * alpha)

    f0 = eval_f(state, params, idx, u)
    fr = eval_f(rotated, params, idx, u)
    psi_err = float(np.max(np.abs(fr[:n] - f0[:n] * np.exp(1j * alpha))))
    phi_err = float(np.max(np.abs(fr[n:] - f0[n:])))

    with phys_log.test(
        "test_global_phase_rotation_is_exact_symmetry",
        {"Nx": 8, "alpha": alpha, "Bz": 0.4},
        "global U(1) symmetry holds even with an applied field on the boundary",
    ) as log:
        log.check_below("max|dψ/dt rotation error|", psi_err, 1e-12)
        log.check_below("max|dφ/dt change|", phi_err, 1e-12)


# ---------------------------------------------------------------------------
# Invariance of observables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid", GRIDS_2D + GRIDS_3D, ids=lambda g: f"{g['Nx']}x{g['Ny']}x{g['Nz']}")
def test_observables_are_gauge_invariant(grid, phys_log):
    """|ψ|, B = ∇×A, J_s and the free energy do not move under a gauge change."""
    params, idx = make_grid(**grid)
    n = params.n_interior
    state = random_state(params, seed=41)
    chi = smooth_gauge_field(params, seed=42)
    state_g = gauge_transform(state, chi, params, idx)

    psi0, px0, py0, pz0 = expand_state(state, params, idx)
    psi1, px1, py1, pz1 = expand_state(state_g, params, idx)

    b0 = np.stack(eval_bfield_full(px0, py0, pz0, params, idx))
    b1 = np.stack(eval_bfield_full(px1, py1, pz1, params, idx))
    j0 = np.stack(eval_supercurrent_density(psi0, px0, py0, pz0, params, idx))
    j1 = np.stack(eval_supercurrent_density(psi1, px1, py1, pz1, params, idx))

    dpsi = float(np.max(np.abs(np.abs(state_g[:n]) - np.abs(state[:n]))))
    db = float(np.max(np.abs(b1 - b0)))
    dj = float(np.max(np.abs(j1 - j0)))
    f0 = gl_free_energy(state, params, idx)
    f1 = gl_free_energy(state_g, params, idx)

    name = f"test_observables_are_gauge_invariant[{params.Nx}x{params.Ny}x{params.Nz}]"
    with phys_log.test(
        name, dict(grid),
        "every measurable quantity must be independent of the gauge",
    ) as log:
        log["B_scale"] = float(np.max(np.abs(b0)))
        log["J_scale"] = float(np.max(np.abs(j0)))
        log["free_energy"] = float(f0)
        log.check_below("max Δ|ψ|", dpsi, 1e-12)
        log.check_below("max ΔB", db, 1e-11 * max(float(np.max(np.abs(b0))), 1.0))
        log.check_below("max ΔJ_s", dj, 1e-11 * max(float(np.max(np.abs(j0))), 1.0))
        log.check_below("Δ free energy", abs(f1 - f0), 1e-9 * max(abs(f0), 1.0))


def test_vortex_count_is_gauge_invariant(phys_log):
    """Vorticity is topological: a gauge change must not create or destroy it."""
    params = SimulationParameters(Nx=14, Ny=14, Nz=1, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.6, t_on_fraction=1.0))
    sol = solve(
        device, t_start=0.0, t_stop=25.0, dt=0.01, method="euler",
        progress=False, log_metadata=False,
    )

    n_before, _, w_before = count_vortices_plaquette(sol, device, step=-1)
    vort_before, _ = plaquette_vorticity(sol, step=-1)

    chi = smooth_gauge_field(params, seed=51, amplitude=0.9)
    sol.states[:, -1] = gauge_transform(sol.states[:, -1], chi, params, sol.idx)

    n_after, _, w_after = count_vortices_plaquette(sol, device, step=-1)
    vort_after, _ = plaquette_vorticity(sol, step=-1)

    with phys_log.test(
        "test_vortex_count_is_gauge_invariant",
        {"Nx": 14, "kappa": 2.0, "Bz": 0.6},
        "plaquette vorticity is a topological invariant of the gauge-field configuration",
    ) as log:
        log["n_vortices"] = int(n_before)
        log["windings"] = np.rint(np.real(w_before)).astype(int).tolist()
        log.check_above(
            "vortices present (test would be vacuous otherwise)", n_before, 1.0,
        )
        log.check_close("vortex count after gauge change", n_after, n_before, atol=0.0)
        log.check_below(
            "max Δ(plaquette vorticity)",
            float(np.max(np.abs(vort_after - vort_before))), 1e-9,
        )
        if n_before:
            log.check_below(
                "max |winding change|",
                float(np.max(np.abs(np.sort(np.real(w_after)) - np.sort(np.real(w_before))))),
                1e-9,
            )
