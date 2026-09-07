"""Symmetry and boundary-condition verification for the TDGL solver.

A discretisation inherits the symmetries of the continuum problem only if the
index arithmetic is right.  Most of the checks here are deliberately run on
**non-cubic** grids: the interior numbering is i-slowest while the full grid is
i-fastest, so a routine that applies one set of strides to the other is an exact
transposition and is completely invisible on an ``Nx == Ny == Nz`` grid.

Covered here:

* the applied field really appears on the boundary plaquettes, and only once —
  the corner where two "hi" faces meet is closed by two ghost links and must not
  be given the applied flux twice;
* time reversal ``B → -B``;
* the C4 rotation of a square device and the mirror symmetry of a rectangular
  one;
* agreement between the two B-field evaluators and the reshape helpers on grids
  where an index-ordering mistake cannot hide.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d import AppliedField, Device, SimulationParameters
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.bfield import eval_bfield, eval_bfield_full
from tdgl3d.physics.rhs import _expand_interior_to_full

from .physics_helpers import (
    applied_boundary,
    cfl_limit,
    expand_state,
    interior_strides,
    make_grid,
    run_euler,
)


def _plaquette_flux_grid(state, params, idx, bz):
    """All plaquette fluxes ``B[i, j]`` on the full grid, anchors 0…N-1."""
    _, phi_x, phi_y, _ = expand_state(state, params, idx, applied_boundary(params, idx, bz=bz))
    shape = (params.Ny + 1, params.Nx + 1)
    px = np.real(phi_x).reshape(shape).T
    py = np.real(phi_y).reshape(shape).T
    return (px[:-1, :-1] - px[:-1, 1:] - py[:-1, :-1] + py[1:, :-1]) / (params.hx * params.hy)


# ---------------------------------------------------------------------------
# Applied field on the boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid", [dict(Nx=9, Ny=7), dict(Nx=8, Ny=8)], ids=lambda g: f"{g['Nx']}x{g['Ny']}"
)
def test_applied_flux_on_boundary_plaquettes(grid, phys_log):
    """Every boundary plaquette carries exactly the applied flux, once.

    Each boundary plaquette is closed by one ghost link that the boundary
    condition offsets by ``B·h_x h_y``.  At the corner where the two *hi* faces
    meet, the plaquette is closed by two such links; giving it a full offset on
    both puts twice the applied field there and leaves a permanent unbalanced
    curl-curl force on the corner links.
    """
    bz = 0.12
    params = SimulationParameters(Nz=1, hx=0.5, hy=0.5, kappa=2.0, **grid)
    dt = 0.5 * cfl_limit(params)
    _, states, _, idx = run_euler(
        params, bz, n_steps=1200, dt=dt, noise_amplitude=0.0, save_every=100,
    )
    flux = _plaquette_flux_grid(states[:, -1], params, idx, bz)

    ring = np.concatenate([flux[-1, :], flux[:, -1], flux[0, 1:], flux[1:, 0]])
    drift = float(np.max(np.abs(states[:, -1] - states[:, -2])))

    name = f"test_applied_flux_on_boundary_plaquettes[{params.Nx}x{params.Ny}]"
    with phys_log.test(
        name, {"Nx": params.Nx, "Ny": params.Ny, "h": 0.5, "Bz": bz},
        "the boundary condition must impose B_applied on each boundary plaquette exactly once",
    ) as log:
        log["corner_hi_hi"] = float(flux[-1, -1])
        log["boundary_ring_min"] = float(ring.min())
        log["boundary_ring_max"] = float(ring.max())
        log["interior_min"] = float(flux[1:-1, 1:-1].min())
        log.check_close(
            "flux on the hi/hi corner plaquette", float(flux[-1, -1]), bz, rtol=1e-10,
            detail="must be B_applied, not 2 B_applied",
        )
        log.check_below(
            "max deviation of boundary ring from B_applied",
            float(np.max(np.abs(ring - bz))), 1e-10 * bz + 1e-14,
        )
        log.check_below(
            "state drift once relaxed", drift, 1e-8,
            detail="an over-counted corner drives an unbounded drift of the corner links",
        )
        log.check_below(
            "screened interior field / applied", float(flux[1:-1, 1:-1].max()) / bz, 0.99,
            detail="the interior must be screened below the applied field",
        )


def test_applied_field_vectors_are_uniform_on_each_face(phys_log):
    """The applied-field source vector is constant over every boundary face."""
    params, idx = make_grid(Nx=7, Ny=6, Nz=5, kappa=2.0)
    bx_vec, by_vec, bz_vec = build_boundary_field_vectors(0.3, -0.2, 1.0, params, idx)

    faces = {
        "Bz on x_lo": bz_vec[idx.x_face_lo_inner],
        "Bz on x_hi": bz_vec[idx.x_face_hi_inner],
        "Bz on y_lo": bz_vec[idx.y_face_lo_inner],
        "Bz on y_hi": bz_vec[idx.y_face_hi_inner],
        "Bx on y_lo": bx_vec[idx.y_face_lo_inner],
        "Bx on z_hi": bx_vec[idx.z_face_hi_inner],
        "By on x_hi": by_vec[idx.x_face_hi_inner],
        "By on z_lo": by_vec[idx.z_face_lo_inner],
    }
    expected = {"Bz": 1.0, "Bx": 0.3, "By": -0.2}

    with phys_log.test(
        "test_applied_field_vectors_are_uniform_on_each_face",
        {"Nx": 7, "Ny": 6, "Nz": 5, "Bx": 0.3, "By": -0.2, "Bz": 1.0},
        "a uniform applied field must be uniform on every face it is imposed on",
    ) as log:
        for label, values in faces.items():
            target = expected[label.split()[0]]
            log.check_below(
                f"max deviation of {label}",
                float(np.max(np.abs(values - target))) if values.size else 0.0,
                1e-14,
            )


# ---------------------------------------------------------------------------
# Discrete symmetries of the dynamics
# ---------------------------------------------------------------------------


def test_field_reversal_flips_b_and_preserves_psi(phys_log):
    """Time reversal: B → −B, J → −J, |ψ| unchanged, to round-off."""
    params = SimulationParameters(Nx=9, Ny=7, Nz=1, kappa=2.0)
    n_steps, dt = 60, 0.5 * cfl_limit(params)

    _, pos, _, idx = run_euler(params, 0.4, n_steps, dt, noise_amplitude=0.0)
    _, neg, _, _ = run_euler(params, -0.4, n_steps, dt, noise_amplitude=0.0)

    worst_b, worst_psi, b_scale = 0.0, 0.0, 0.0
    for step in range(pos.shape[1]):
        _, px_p, py_p, pz_p = expand_state(pos[:, step], params, idx)
        _, px_n, py_n, pz_n = expand_state(neg[:, step], params, idx)
        bz_p = eval_bfield_full(px_p, py_p, pz_p, params, idx)[2]
        bz_n = eval_bfield_full(px_n, py_n, pz_n, params, idx)[2]
        worst_b = max(worst_b, float(np.max(np.abs(bz_p + bz_n))))
        b_scale = max(b_scale, float(np.max(np.abs(bz_p))))
        n = params.n_interior
        worst_psi = max(
            worst_psi,
            float(np.max(np.abs(np.abs(pos[:n, step]) - np.abs(neg[:n, step])))),
        )

    with phys_log.test(
        "test_field_reversal_flips_b_and_preserves_psi",
        {"Nx": 9, "Ny": 7, "Bz": 0.4, "n_steps": n_steps},
        "the GL equations are invariant under B → −B combined with ψ → ψ*",
    ) as log:
        log["B_scale"] = b_scale
        log.check_above("B scale (non-trivial state)", b_scale, 1e-3)
        log.check_below("max|Bz(+B) + Bz(−B)|", worst_b, 1e-12)
        log.check_below("max| |ψ(+B)| − |ψ(−B)| |", worst_psi, 1e-12)


def test_c4_symmetry_of_a_square_device(phys_log):
    """A square device in a uniform Bz has C4-symmetric |ψ| and Bz.

    Checked on the physical fields under an actual 90° rotation of the plaquette
    grid rather than on a component identity, so a transposed reshape cannot
    satisfy it accidentally.
    """
    params = SimulationParameters(Nx=10, Ny=10, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    bz = 0.15
    dt = 0.5 * cfl_limit(params)
    _, states, _, idx = run_euler(params, bz, n_steps=1200, dt=dt, noise_amplitude=0.0)

    n = params.n_interior
    nx_int, ny_int = params.Nx - 1, params.Ny - 1
    psi_abs = np.abs(states[:n, -1]).reshape(nx_int, ny_int)
    _, px, py, pz = expand_state(states[:, -1], params, idx, applied_boundary(params, idx, bz=bz))
    bz_field = eval_bfield_full(px, py, pz, params, idx)[2].reshape(nx_int, ny_int)

    # ψ lives on the interior *nodes* 1…Nx-1, a set the reflection i → Nx-i maps
    # onto itself, so the whole array is compared.  Bz lives on *plaquettes*,
    # anchored at nodes 1…Nx-1 but spanning 1…Nx; the reflection of the anchor-1
    # plaquette is the ghost anchor-0 plaquette, which the interior array does
    # not carry, so the last anchor is dropped before comparing.
    psi_core, bz_core = psi_abs, bz_field[:-1, :-1]
    psi_rot = np.rot90(psi_core)
    bz_rot = np.rot90(bz_core)

    with phys_log.test(
        "test_c4_symmetry_of_a_square_device",
        {"Nx": 10, "Ny": 10, "h": 0.5, "Bz": bz},
        "a square sample in a uniform out-of-plane field is invariant under 90° rotation",
    ) as log:
        log["psi_contrast"] = float(psi_core.max() - psi_core.min())
        log["bz_contrast"] = float(bz_core.max() - bz_core.min())
        log.check_above(
            "Bz contrast (screening present, so the test is non-trivial)",
            float(bz_core.max() - bz_core.min()), 1e-3,
        )
        log.check_below("max|ψ| − R₉₀|ψ||", float(np.max(np.abs(psi_core - psi_rot))), 1e-12)
        log.check_below("max|Bz − R₉₀Bz|", float(np.max(np.abs(bz_core - bz_rot))), 1e-12)


def test_mirror_symmetry_of_a_rectangular_device(phys_log):
    """|ψ| and Bz are mirror-symmetric about both mid-planes on a Nx ≠ Ny grid."""
    params = SimulationParameters(Nx=12, Ny=8, Nz=1, hx=0.5, hy=0.5, kappa=2.0)
    bz = 0.15
    dt = 0.5 * cfl_limit(params)
    _, states, _, idx = run_euler(params, bz, n_steps=1200, dt=dt, noise_amplitude=0.0)

    n = params.n_interior
    nx_int, ny_int = params.Nx - 1, params.Ny - 1
    # Node-centred ψ reflects onto itself; plaquette-centred Bz needs its last
    # anchor dropped first (see test_c4_symmetry_of_a_square_device).
    psi_abs = np.abs(states[:n, -1]).reshape(nx_int, ny_int)
    _, px, py, pz = expand_state(states[:, -1], params, idx, applied_boundary(params, idx, bz=bz))
    bz_field = eval_bfield_full(px, py, pz, params, idx)[2].reshape(nx_int, ny_int)[:-1, :-1]

    with phys_log.test(
        "test_mirror_symmetry_of_a_rectangular_device",
        {"Nx": 12, "Ny": 8, "h": 0.5, "Bz": bz},
        "reflection symmetry on a non-square grid — a transposed index would break it",
    ) as log:
        log["psi_contrast"] = float(psi_abs.max() - psi_abs.min())
        log.check_above("Bz contrast", float(bz_field.max() - bz_field.min()), 1e-3)
        log.check_below("max|ψ(x) − ψ(−x)|", float(np.max(np.abs(psi_abs - psi_abs[::-1, :]))), 1e-12)
        log.check_below("max|ψ(y) − ψ(−y)|", float(np.max(np.abs(psi_abs - psi_abs[:, ::-1]))), 1e-12)
        log.check_below("max|Bz(x) − Bz(−x)|", float(np.max(np.abs(bz_field - bz_field[::-1, :]))), 1e-12)
        log.check_below("max|Bz(y) − Bz(−y)|", float(np.max(np.abs(bz_field - bz_field[:, ::-1]))), 1e-12)


# ---------------------------------------------------------------------------
# Index-ordering consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid",
    [dict(Nx=6, Ny=6, Nz=1), dict(Nx=9, Ny=5, Nz=1), dict(Nx=5, Ny=7, Nz=6)],
    ids=lambda g: f"{g['Nx']}x{g['Ny']}x{g['Nz']}",
)
def test_interior_numbering_matches_documented_strides(grid, phys_log):
    """``interior_to_full`` is i-slowest / k-fastest, opposite to the full grid."""
    params, idx = make_grid(kappa=2.0, **grid)
    si, sj, sk = interior_strides(params)
    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)
    mj, mk = params.mj, params.mk

    expected = np.empty(params.n_interior, dtype=np.intp)
    for i in range(nx_int):
        for j in range(ny_int):
            for k in range(nz_int):
                expected[i * si + j * sj + k * sk] = (
                    (i + 1) + mj * (j + 1) + (mk * (k + 1) if params.is_3d else 0)
                )

    # A reshape must recover (i, j, k) with the same convention.
    flat = np.arange(params.n_interior)
    cube = flat.reshape(nx_int, ny_int, nz_int)
    reshape_ok = int(np.max(np.abs(cube[1, 0, 0] - si) + np.abs(cube[0, 1, 0] - sj)))

    name = f"test_interior_numbering_matches_documented_strides[{params.Nx}x{params.Ny}x{params.Nz}]"
    with phys_log.test(
        name, dict(grid),
        "interior arrays are C-ordered over (Nx-1, Ny-1, Nz-1)",
    ) as log:
        log["strides"] = [int(si), int(sj), int(sk)]
        log.check_below(
            "mismatched entries of interior_to_full",
            float(np.count_nonzero(idx.interior_to_full != expected)), 0.0,
        )
        log.check_below("reshape stride mismatch", float(reshape_ok), 0.0)


@pytest.mark.parametrize(
    "grid",
    [dict(Nx=6, Ny=6, Nz=1), dict(Nx=9, Ny=5, Nz=1), dict(Nx=5, Ny=7, Nz=6)],
    ids=lambda g: f"{g['Nx']}x{g['Ny']}x{g['Nz']}",
)
def test_bfield_evaluators_agree(grid, phys_log):
    """``eval_bfield`` and ``eval_bfield_full`` must return the same field.

    They used different index strides, which agree only on a cubic grid.
    """
    params, idx = make_grid(kappa=2.0, **grid)
    rng = np.random.default_rng(17)
    n = params.n_interior
    n_components = 4 if params.is_3d else 3
    state = (rng.normal(size=n_components * n) * 0.25).astype(np.complex128)

    subset = eval_bfield(state, params, idx, full_interior=False)
    everywhere = eval_bfield(state, params, idx, full_interior=True)

    n_links = 3 if params.is_3d else 2
    phi = [
        _expand_interior_to_full(state[(k + 1) * n : (k + 2) * n], params, idx)
        for k in range(n_links)
    ]
    while len(phi) < 3:
        phi.append(np.zeros(params.dim_x, dtype=np.complex128))
    reference = eval_bfield_full(phi[0], phi[1], phi[2], params, idx)

    subset_err = max(
        float(np.max(np.abs(a - b[idx.bfield_interior]))) for a, b in zip(subset, reference)
    )
    full_err = max(float(np.max(np.abs(a - b))) for a, b in zip(everywhere, reference))

    nz_layers = max(params.Nz - 2, 1) if params.is_3d else 1
    expected_subset = (params.Nx - 2) * (params.Ny - 2) * nz_layers

    name = f"test_bfield_evaluators_agree[{params.Nx}x{params.Ny}x{params.Nz}]"
    with phys_log.test(
        name, dict(grid), "one curl stencil, one answer, on any grid shape",
    ) as log:
        log["b_scale"] = float(np.max(np.abs(np.stack(reference))))
        log.check_below("max|eval_bfield(subset) − reference|", subset_err, 1e-14)
        log.check_below("max|eval_bfield(all interior) − reference|", full_err, 1e-14)
        log.check_close(
            "len(bfield_interior)", float(idx.bfield_interior.size), float(expected_subset),
            atol=0.0,
        )


def test_solution_reshape_helpers_are_consistent(phys_log):
    """``Solution`` slicing helpers index ``[i, j]`` on a non-square grid."""
    params = SimulationParameters(Nx=11, Ny=6, Nz=1, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.2, t_on_fraction=1.0))
    _, states, _, idx = run_euler(params, 0.2, n_steps=10, dt=0.5 * cfl_limit(params), seed=4)

    from tdgl3d.core.solution import Solution

    sol = Solution(
        times=np.linspace(0.0, 1.0, states.shape[1]),
        states=states,
        params=params,
        idx=idx,
        device=device,
    )
    psi2_2d = sol.psi_squared_2d(step=-1)
    psi2_flat = np.abs(sol.psi(step=-1)) ** 2
    si, sj, _ = interior_strides(params)

    with phys_log.test(
        "test_solution_reshape_helpers_are_consistent",
        {"Nx": 11, "Ny": 6},
        "the 2-D view must be indexed [i, j] with the interior strides",
    ) as log:
        log["shape"] = list(psi2_2d.shape)
        log.check_close("shape[0]", float(psi2_2d.shape[0]), float(params.Nx - 1), atol=0.0)
        log.check_close("shape[1]", float(psi2_2d.shape[1]), float(params.Ny - 1), atol=0.0)
        log.check_below(
            "max|reshape − stride-indexed|",
            float(np.max(np.abs(psi2_2d[2, 3] - psi2_flat[2 * si + 3 * sj]))), 1e-15,
        )


def test_indices_are_within_bounds_on_ragged_grids(phys_log):
    """No index array may reach outside the full grid on an anisotropic grid."""
    checked = 0
    worst = -1
    for nx, ny, nz in [(4, 9, 3), (9, 4, 3), (3, 4, 9), (5, 5, 2)]:
        params = SimulationParameters(Nx=nx, Ny=ny, Nz=nz, kappa=2.0)
        idx = construct_indices(params)
        for name in (
            "interior_to_full", "x_face_lo_inner", "x_face_hi_inner", "x_first_inner",
            "x_last_inner", "y_face_lo_inner", "y_face_hi_inner", "y_first_inner",
            "y_last_inner", "z_face_lo_inner", "z_face_hi_inner", "z_first_inner",
            "z_last_inner", "x_normal_bc_mask", "y_normal_bc_mask", "z_normal_bc_mask",
        ):
            arr = getattr(idx, name)
            if arr.size:
                worst = max(worst, int(arr.max()) - (params.dim_x - 1))
                checked += 1
        if idx.bfield_interior.size:
            worst = max(worst, int(idx.bfield_interior.max()) - (params.n_interior - 1))
            checked += 1

    with phys_log.test(
        "test_indices_are_within_bounds_on_ragged_grids",
        {"grids": "4x9x3, 9x4x3, 3x4x9, 5x5x2"},
        "index arrays must stay in range for every grid aspect ratio",
    ) as log:
        log["index_arrays_checked"] = checked
        log.check_above("index arrays checked", float(checked), 40.0)
        log.check_below("worst overshoot past the last valid index", float(worst), -1.0)
