"""Physics validation for heterostructures (trilayer / material maps).

The general-purpose physics verification lives in four dedicated suites, each
organised around a principle rather than around a feature:

===================================== =======================================
Module                                Verifies
===================================== =======================================
``test_verification_gauge.py``        local U(1) covariance of the right-hand
                                      side; gauge invariance of |ψ|, B, J_s,
                                      the free energy and the vortex count
``test_verification_conservation.py`` ∇·B = 0 and ∇·(∇×∇×A) = 0 as exact
                                      identities; free energy as a Lyapunov
                                      functional; ∇·J_s = 0 in steady state;
                                      J_n = 0 on every external face; CFL
``test_verification_symmetry.py``     applied flux on the boundary
                                      plaquettes; B → −B; C4 and mirror
                                      symmetry; index-ordering consistency on
                                      non-cubic grids
``test_verification_analytic.py``     λ = κ; the lowest Landau level E₀ = B
                                      (hence H_c2 = 1); order of accuracy in
                                      h and in dt; insulator relaxation time
``test_verification_vortex.py``       exact fluxoid quantisation; winding sign
                                      versus field sign; lattice Stokes;
                                      Meissner state free of vortices
===================================== =======================================

What remains here are the checks specific to a stacked
superconductor/insulator/superconductor device, where a spatially varying κ and
the insulator mask enter the operators.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d import (
    AppliedField,
    Device,
    Layer,
    SimulationParameters,
    Trilayer,
)
from tdgl3d.core.material import build_material_map
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.operators.sparse_operators import construct_LPHI_x
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.current_density import eval_supercurrent_density
from tdgl3d.solvers.integrators import forward_euler

from .physics_helpers import (
    applied_boundary,
    cfl_limit,
    expand_state,
    zero_boundary,
)


def _trilayer(thickness: int = 2, kappa: float = 2.0) -> Trilayer:
    return Trilayer(
        bottom=Layer(thickness_z=thickness, kappa=kappa),
        insulator=Layer(thickness_z=thickness, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=thickness, kappa=kappa),
    )


def _interior_position(params, idx) -> dict[int, int]:
    """Map full-grid index → interior index.

    ``interior_to_full`` is *not* sorted (it runs i-slowest over a grid whose
    linear index runs i-fastest), so a bisection lookup on it silently returns
    the wrong node.
    """
    return {int(node): pos for pos, node in enumerate(idx.interior_to_full)}


def test_trilayer_kappa_discontinuity(phys_log):
    """The φ-Laplacian picks up κ from the material map, node by node.

    In the superconducting layers the diagonal must be ``-2κ²(1/h_y² + 1/h_z²)``;
    in the insulator, where ``κ = 0``, it must vanish exactly.
    """
    kappa = 2.0
    trilayer = _trilayer(kappa=kappa)
    params = SimulationParameters(Nx=4, Ny=5, Nz=trilayer.Nz, kappa=kappa)
    idx = construct_indices(params)
    material = build_material_map(params, trilayer, idx)

    operator = construct_LPHI_x(params, idx, material)
    m = idx.interior_to_full
    diagonal = np.real(np.asarray(operator[m, m]).ravel())
    position = _interior_position(params, idx)

    def layer_mean(k_full: int) -> float:
        values = [
            diagonal[position[i + params.mj * j + params.mk * k_full]]
            for i in range(1, params.Nx)
            for j in range(1, params.Ny)
            if (i + params.mj * j + params.mk * k_full) in position
        ]
        return float(np.mean(values))

    expected_sc = -2.0 * (kappa**2 / params.hy**2 + kappa**2 / params.hz**2)
    ranges = trilayer.z_ranges()
    k_sc = max(ranges["bottom"][0], 1)
    k_ins = ranges["insulator"][0]

    with phys_log.test(
        "test_trilayer_kappa_discontinuity",
        {"Nx": 4, "Ny": 5, "Nz": trilayer.Nz, "kappa": kappa},
        "a spatially varying κ must appear in the operator coefficients exactly",
    ) as log:
        log["k_superconductor"] = int(k_sc)
        log["k_insulator"] = int(k_ins)
        log.check_close(
            "LPHI_x diagonal in the superconductor", layer_mean(k_sc), expected_sc,
            atol=1e-12,
        )
        log.check_close(
            "LPHI_x diagonal in the insulator", layer_mean(k_ins), 0.0, atol=1e-12,
            detail="κ = 0 there, so the curl-curl term must switch off entirely",
        )


def test_trilayer_external_z_boundary_jn(phys_log):
    """No supercurrent leaves the stack through the top or bottom face.

    Indexed with the interior strides (i-slowest / k-fastest); applying the
    full-grid strides here samples an x-slab instead of the z-faces, which is
    exactly where the current is *not* expected to vanish.
    """
    trilayer = _trilayer()
    params = SimulationParameters(Nx=5, Ny=4, Nz=trilayer.Nz, kappa=2.0)
    bz = 0.5
    device = Device(params, applied_field=AppliedField(Bz=bz, t_on_fraction=1.0), trilayer=trilayer)
    idx, material = device.idx, device.material

    boundary = applied_boundary(params, idx, bz=bz)
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data, params, idx,
        lambda t, X: boundary, 0.0, 0.2, 0.01,
        save_every=10**9, progress=False, material=material,
    )
    psi, px, py, pz = expand_state(states[:, -1], params, idx, boundary)
    _, _, jz_interior = eval_supercurrent_density(psi, px, py, pz, params, idx)

    def face_current(nodes):
        """|J_z| on the z-links anchored at *nodes*, in full-grid numbering."""
        if not nodes.size:
            return 0.0
        return float(np.max(np.abs(np.imag(
            np.conj(psi[nodes]) * np.exp(-1j * pz[nodes]) * psi[nodes + params.mk]
        ))))

    # The external z-links are the one closing onto the k=0 ghost plane and the
    # one leaving the last interior plane — not the links at the first and last
    # *interior* nodes, which are ordinary bulk links.
    lo_current = face_current(idx.z_face_lo_inner)
    hi_current = face_current(idx.z_last_inner)

    with phys_log.test(
        "test_trilayer_external_z_boundary_jn",
        {"Nx": 5, "Ny": 4, "Nz": trilayer.Nz, "Bz": bz},
        "the z-faces of the stack are superconductor/vacuum interfaces",
    ) as log:
        log["bulk_Jz_scale"] = float(np.max(np.abs(jz_interior)))
        log.check_below("max |J_z| on the bottom face", lo_current, 1e-12)
        log.check_below("max |J_z| on the top face", hi_current, 1e-12)


def _relax_stack(params, idx, material, boundary, t_stop=15.0, device=None):
    """Integrate a trilayer device at a fixed applied field and return the state."""
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data, params, idx,
        lambda t, X: boundary, 0.0, t_stop, 0.5 * cfl_limit(params),
        save_every=10**9, progress=False, material=material,
    )
    return states[:, -1]


def _layer_field_profile(state, params, idx, boundary):
    """Bz along z at the centre of the stack, one entry per interior z-plane."""
    _, px, py, pz = expand_state(state, params, idx, boundary)
    field = eval_bfield_full(px, py, pz, params, idx)[2]
    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, params.Nz - 1
    return np.real(field.reshape(nx_int, ny_int, nz_int)[nx_int // 2, ny_int // 2, :])


@pytest.mark.parametrize("kappa_insulator", [0.0, 2.0])
def test_insulator_kappa_controls_field_transmission(kappa_insulator, phys_log):
    """κ in a non-superconducting layer decides whether it can carry a field.

    The φ-equation is ``∂A/∂t = J_s − κ²∇×∇×A``.  In a layer with ``ψ = 0`` the
    supercurrent term vanishes, so the layer relaxes towards the magnetostatic
    solution ``∇×∇×A = 0`` at a rate set by ``κ²`` — *any* positive κ gives the
    same steady state, only faster or slower.  Setting ``κ = 0`` removes the
    only remaining term: the gauge field there is frozen at its initial value
    and the layer transmits nothing.

    ``Layer(kappa=0.0, is_superconductor=False)`` is therefore a modelling
    choice with real consequences, not a neutral way of saying "not a
    superconductor". Give an oxide the same κ as the metal unless the field is
    meant to be blocked.
    """
    kappa_sc, bz = 2.0, 0.1
    trilayer = Trilayer(
        bottom=Layer(thickness_z=4, kappa=kappa_sc),
        insulator=Layer(thickness_z=4, kappa=kappa_insulator, is_superconductor=False),
        top=Layer(thickness_z=4, kappa=kappa_sc),
    )
    params = SimulationParameters(
        Nx=12, Ny=12, Nz=trilayer.Nz, hx=0.5, hy=0.5, hz=0.5, kappa=kappa_sc
    )
    device = Device(
        params, applied_field=AppliedField(Bz=bz, t_on_fraction=1.0), trilayer=trilayer
    )
    idx = device.idx
    boundary = applied_boundary(params, idx, bz=bz)
    state = _relax_stack(params, idx, device.material, boundary, device=device)
    profile = _layer_field_profile(state, params, idx, boundary) / bz

    ranges = trilayer.z_ranges()
    start, stop = ranges["insulator"]
    insulator_mean = float(np.mean(profile[max(start, 1) - 1 : stop - 1]))

    name = f"test_insulator_kappa_controls_field_transmission[kappa={kappa_insulator}]"
    with phys_log.test(
        name,
        {"Nz": trilayer.Nz, "kappa_sc": kappa_sc, "kappa_insulator": kappa_insulator, "Bz": bz},
        "a κ = 0 layer freezes the gauge field; a κ > 0 layer transmits the field",
    ) as log:
        log["bz_profile_over_applied"] = [float(v) for v in profile]
        log["insulator_mean_over_applied"] = insulator_mean
        if kappa_insulator == 0.0:
            log.check_below(
                "Bz in the insulator / applied", abs(insulator_mean), 1e-6,
                detail="κ = 0 leaves no term able to evolve A there",
            )
        else:
            log.check_above(
                "Bz in the insulator / applied", insulator_mean, 0.5,
                detail="a non-superconducting layer with κ > 0 lets the field through",
            )


def test_trilayer_superconducting_layers_screen(phys_log):
    """With a transmitting oxide, both Nb layers still screen the applied field.

    Run with ``κ_insulator = κ_SC`` so the stack is magnetically continuous;
    the screening then comes from the superconducting layers rather than from a
    frozen gauge field in the middle.
    """
    kappa, bz = 2.0, 0.1
    trilayer = Trilayer(
        bottom=Layer(thickness_z=4, kappa=kappa),
        insulator=Layer(thickness_z=4, kappa=kappa, is_superconductor=False),
        top=Layer(thickness_z=4, kappa=kappa),
    )
    params = SimulationParameters(
        Nx=16, Ny=16, Nz=trilayer.Nz, hx=0.5, hy=0.5, hz=0.5, kappa=kappa
    )
    device = Device(
        params, applied_field=AppliedField(Bz=bz, t_on_fraction=1.0), trilayer=trilayer
    )
    idx = device.idx
    boundary = applied_boundary(params, idx, bz=bz)
    state = _relax_stack(params, idx, device.material, boundary, t_stop=20.0, device=device)
    profile = _layer_field_profile(state, params, idx, boundary) / bz

    ranges = trilayer.z_ranges()

    def layer_mean(name: str) -> float:
        start, stop = ranges[name]
        return float(np.mean(profile[max(start, 1) - 1 : stop - 1]))

    bottom, top = layer_mean("bottom"), layer_mean("top")

    with phys_log.test(
        "test_trilayer_superconducting_layers_screen",
        {"Nx": 16, "Nz": trilayer.Nz, "kappa": kappa, "Bz": bz},
        "both superconducting layers of a magnetically continuous stack screen",
    ) as log:
        log["bz_profile_over_applied"] = [float(v) for v in profile]
        log["bottom_over_applied"] = bottom
        log["top_over_applied"] = top
        log.check_below("Bz in the bottom Nb layer / applied", bottom, 0.98)
        log.check_below("Bz in the top Nb layer / applied", top, 0.98)
        log.check_above("Bz in the bottom Nb layer / applied", bottom, 0.0)
        log.check_close(
            "top/bottom screening asymmetry", top / bottom, 1.0, atol=0.15,
            detail="the stack is symmetric about its mid-plane",
        )


def test_insulator_mask_suppresses_the_order_parameter(phys_log):
    """ψ is driven to zero inside the insulator and stays finite outside it."""
    # Four cells per superconducting layer: with only two, proximity suppression
    # from the insulator reaches all the way through and there is no bulk-like
    # interior left to check.
    trilayer = _trilayer(thickness=4)
    params = SimulationParameters(Nx=4, Ny=5, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.0), trilayer=trilayer)
    idx, material = device.idx, device.material

    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data, params, idx,
        lambda t, X: zero_boundary(params), 0.0, 8.0, 0.01,
        save_every=10**9, progress=False, material=material,
    )
    n = params.n_interior
    psi = np.abs(states[:n, -1])
    insulator = material.interior_sc_mask == 0.0

    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, params.Nz - 1
    profile = psi.reshape(nx_int, ny_int, nz_int)[nx_int // 2, ny_int // 2, :]
    mask_profile = material.interior_sc_mask.reshape(nx_int, ny_int, nz_int)[
        nx_int // 2, ny_int // 2, :
    ]

    with phys_log.test(
        "test_insulator_mask_suppresses_the_order_parameter",
        {"Nx": 4, "Ny": 5, "Nz": trilayer.Nz, "sc_thickness": 4},
        "the material mask must separate superconducting from insulating nodes",
    ) as log:
        log["n_insulator_nodes"] = int(insulator.sum())
        log["psi_z_profile"] = [float(v) for v in profile]
        log["sc_mask_z_profile"] = [float(v) for v in mask_profile]
        log.check_above("insulator nodes present", float(insulator.sum()), 1.0)
        log.check_below(
            "mean |ψ| in the insulator", float(np.mean(psi[insulator])), 0.15,
            detail="residual value is proximity leakage from the adjacent layers",
        )
        log.check_above(
            "max |ψ| in the superconductor", float(np.max(psi[~insulator])), 0.95,
            detail="the middle of a 4-cell layer must recover the bulk condensate",
        )
        log.check_above(
            "mean |ψ| in the superconductor", float(np.mean(psi[~insulator])), 0.75,
        )

# ---------------------------------------------------------------------------
# Geometric symmetry of the stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc_cells,insulator_cells",
    [(4, 2), (3, 1), (5, 2)],
    ids=lambda v: str(v),
)
def test_stack_is_mirror_symmetric_about_its_midplane(sc_cells, insulator_cells, phys_log):
    """A stack with equal S layers must be symmetric under z → Nz − z.

    Layer thicknesses are given in cells but material properties live on nodes,
    and the two interface nodes are shared.  Assigning each node to the cell
    range containing it hands the lower interface to the oxide and the upper one
    to the top layer, leaving the superconducting layers with different node
    counts.  Both interfaces belong to the oxide instead.
    """
    trilayer = Trilayer(
        bottom=Layer(thickness_z=sc_cells, kappa=2.0),
        insulator=Layer(thickness_z=insulator_cells, kappa=2.0, is_superconductor=False),
        top=Layer(thickness_z=sc_cells, kappa=2.0),
    )
    params = SimulationParameters(Nx=6, Ny=5, Nz=trilayer.Nz, kappa=2.0)
    device = Device(params, applied_field=AppliedField(Bz=0.0), trilayer=trilayer)

    profile = device.material.sc_mask.reshape(
        params.Nz + 1, params.Ny + 1, params.Nx + 1
    )[:, 0, 0]
    kappa_profile = device.material.kappa.reshape(
        params.Nz + 1, params.Ny + 1, params.Nx + 1
    )[:, 0, 0]
    n_bottom = int(profile[: len(profile) // 2].sum())
    n_top = int(profile[len(profile) // 2 + len(profile) % 2 :].sum())

    name = f"test_stack_is_mirror_symmetric_about_its_midplane[({sc_cells}, {insulator_cells})]"
    with phys_log.test(
        name, {"sc_cells": sc_cells, "insulator_cells": insulator_cells, "Nz": trilayer.Nz},
        "equal superconducting layers must give an exactly symmetric material map",
    ) as log:
        log["sc_mask_z_profile"] = [int(v) for v in profile]
        log.check_below(
            "sc_mask asymmetry under z → Nz − z",
            float(np.max(np.abs(profile - profile[::-1]))), 0.0,
        )
        log.check_below(
            "κ asymmetry under z → Nz − z",
            float(np.max(np.abs(kappa_profile - kappa_profile[::-1]))), 1e-15,
        )
        log.check_close(
            "superconducting nodes below vs above the mid-plane",
            float(n_bottom), float(n_top), atol=0.0,
        )


@pytest.mark.parametrize(
    "length,hole,h",
    [(10.0, 4.0, 1.0), (10.0, 4.0, 0.5), (12.0, 6.0, 1.0), (12.0, 5.0, 0.5)],
    ids=lambda v: str(v),
)
def test_centred_hole_is_centred(length, hole, h, phys_log):
    """A hole centred in the film comes out centred on the grid.

    Bare ray casting is half-open — points on the low-x/low-y edges fall outside
    and those on the high edges inside — so a polygon whose edges land on grid
    nodes is carved half a cell off centre, and every symmetry of the device is
    broken by that much.  ``identify_hole_nodes`` takes the closed region by
    default for this reason.
    """
    trilayer = _trilayer(thickness=4)
    n_cells = int(round(length / h))
    params = SimulationParameters(
        Nx=n_cells, Ny=n_cells, Nz=trilayer.Nz, hx=h, hy=h, kappa=2.0
    )
    device = Device(params, applied_field=AppliedField(Bz=0.0), trilayer=trilayer)
    lo, hi = 0.5 * (length - hole), 0.5 * (length + hole)
    square = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    z_ranges = trilayer.z_ranges()
    device.add_hole(square, z_range=z_ranges["bottom"])
    device.add_hole(square, z_range=z_ranges["top"])

    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, params.Nz - 1
    mask = device.material.interior_sc_mask.reshape(nx_int, ny_int, nz_int)
    plane = mask[:, :, 0]
    carved = plane == 0.0
    ii, jj = np.nonzero(carved)

    centre_x = 0.5 * (ii.min() + ii.max() + 2) * h  # +1 for the ghost offset, twice
    centre_y = 0.5 * (jj.min() + jj.max() + 2) * h
    width = (ii.max() - ii.min()) * h

    name = f"test_centred_hole_is_centred[({length}, {hole}, {h})]"
    with phys_log.test(
        name, {"length": length, "hole": hole, "h": h},
        "the carved geometry must inherit the symmetry of the polygon it was given",
    ) as log:
        log["carved_nodes"] = int(carved.sum())
        log["hole_centre"] = [centre_x, centre_y]
        log["hole_width"] = width
        log.check_above("nodes carved out", float(carved.sum()), 1.0)
        log.check_close("hole centre x", centre_x, length / 2, atol=1e-12, units="ξ")
        log.check_close("hole centre y", centre_y, length / 2, atol=1e-12, units="ξ")
        log.check_close("carved width", width, hole, atol=1e-12, units="ξ")
        log.check_below(
            "material map asymmetry under x → −x",
            float(np.max(np.abs(mask - mask[::-1, :, :]))), 0.0,
        )
        log.check_below(
            "material map asymmetry under y → −y",
            float(np.max(np.abs(mask - mask[:, ::-1, :]))), 0.0,
        )
        log.check_below(
            "material map asymmetry under z → −z",
            float(np.max(np.abs(mask - mask[:, :, ::-1]))), 0.0,
        )
