"""Verification: the field outside the stack, and the interfaces it crosses.

The applied field enters this solver as Peierls offsets on the *ghost*
links that close the boundary plaquettes, which pins the flux through
those plaquettes to the applied value.  That is the right statement only
where the boundary is far-field vacuum.  When the superconductor runs
all the way to the wall of the box, the same condition instead pins the
field on the metal's own surface, and the film is given no room to expel
flux into.

These checks pin down what happens outside the stack:

* an empty box must reproduce the applied field exactly, so any error
  seen with a device in the box belongs to the device and not to the
  boundary treatment;
* a κ contrast in a region carrying no current must change nothing,
  because the Maxwell coefficient is the vacuum field energy;
* with vacuum padding the field near the film must be free to differ
  from the applied field, while the far field returns to it, and the
  residual must fall as the padding grows;
* flux is conserved: the film redistributes the applied flux, it does
  not destroy it.

Each field solve here freezes ψ, which makes the φ-equation linear, and
solves for its steady state directly rather than time-stepping to it.
The steady state is what the checks are about, and a least-squares solve
reaches it exactly instead of approaching it — the reported residual says
whether a steady state exists at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d.core.material import MaterialMap
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.rhs import (
    _apply_boundary_conditions,
    _expand_interior_to_full,
    eval_f,
)
from tdgl3d.solvers.integrators import forward_euler

from .physics_helpers import applied_boundary, cfl_limit

B0 = 0.02
KAPPA = 2.0


# ────────────────────────────────────────────────────────────
# Frozen-ψ steady state
# ────────────────────────────────────────────────────────────

def _vacuum_steady_field(params, idx, material, *, bx=0.0, by=0.0, bz=0.0):
    """Exact steady state of the φ-equation in a box where **ψ = 0**.

    With ψ = 0 the supercurrent term ``Im[e^{-iφ} ψ* ψ']`` vanishes
    identically, so the φ-block of ``eval_f`` is *exactly* affine —
    ``dφ/dt = M φ + b`` — and its steady state can be solved for
    rather than relaxed towards.  Build ``M`` a column at a time and
    solve ``M φ = -b``.

    This is only valid at ψ = 0.  With a condensate present the
    supercurrent depends on φ through ``e^{-iφ}`` and a unit-column
    probe measures a secant, not the operator; those cases relax with
    :func:`_relax` instead.

    The curl-curl operator annihilates pure gauge, so ``M`` is singular
    by construction and the minimum-norm solution is taken; everything
    checked here is gauge invariant, so which representative comes back
    does not matter.

    Returns ``(Bx, By, Bz, residual)`` with the components shaped
    ``(Nx-1, Ny-1, Nz-1)`` and *residual* the relative least-squares
    residual — non-zero would mean no steady state exists at all.
    """
    n = params.n_interior
    n_phi = 3 * n if params.is_3d else 2 * n
    boundary = applied_boundary(params, idx, bx=bx, by=by, bz=bz)

    base = np.zeros(params.n_state, dtype=np.complex128)   # ψ = 0
    offset = np.real(eval_f(base, params, idx, boundary, material)[n:])

    matrix = np.zeros((n_phi, n_phi))
    for column in range(n_phi):
        probe = base.copy()
        probe[n + column] = 1.0
        matrix[:, column] = (
            np.real(eval_f(probe, params, idx, boundary, material)[n:]) - offset
        )

    phi, *_ = np.linalg.lstsq(matrix, -offset, rcond=None)
    residual = float(
        np.abs(matrix @ phi + offset).max() / max(np.abs(offset).max(), 1e-300)
    )

    state = base.copy()
    state[n:] = phi
    return (*_bfield_of(state, params, idx, boundary), residual)


def _bfield_of(state, params, idx, boundary):
    """``(Bx, By, Bz)`` of a state, shaped ``(Nx-1, Ny-1, Nz-1)``."""
    n = params.n_interior
    full = [
        _expand_interior_to_full(state[i * n : (i + 1) * n], params, idx)
        for i in range(4)
    ]
    _, phi_x, phi_y, phi_z = _apply_boundary_conditions(*full, params, idx, boundary)
    shape = (params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))
    return tuple(
        c.reshape(shape) for c in eval_bfield_full(phi_x, phi_y, phi_z, params, idx)
    )


def _relax(params, device, *, bz=B0, t_stop=40.0):
    """Relax a device to its steady state and return ``(Bz, residual)``.

    Used wherever a condensate is present, where the exact solve above
    does not apply.  The reported residual is ``max |dX/dt|`` at the
    final state, so a check can say how close to steady the answer
    it is reading actually is.
    """
    boundary = applied_boundary(params, device.idx, bz=bz)
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data,
        params, device.idx, lambda t, X: boundary,
        0.0, t_stop, 0.5 * cfl_limit(params),
        save_every=10**9, progress=False, material=device.material,
    )
    state = states[:, -1]
    residual = float(
        np.abs(eval_f(state, params, device.idx, boundary, device.material)).max()
    )
    return _bfield_of(state, params, device.idx, boundary)[2], residual


def _empty_box(params):
    """A ``MaterialMap`` in which nothing superconducts."""
    return MaterialMap(
        kappa=np.full(params.dim_x, params.kappa),
        sc_mask=np.zeros(params.dim_x),
        interior_sc_mask=np.zeros(params.n_interior),
    )


# ────────────────────────────────────────────────────────────
# The boundary condition on its own
# ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("component", ["Bx", "By", "Bz"])
def test_empty_box_reproduces_the_applied_field(component, phys_log):
    """With no superconductor anywhere, B must equal the applied field.

    ψ = 0 everywhere means no supercurrent anywhere, so the steady state
    solves ∇×∇×A = 0 subject to the applied flux on the boundary
    plaquettes — and the only such field is the uniform applied one.
    This is the calibration for every other check in this file: it
    isolates the boundary treatment from any device physics, so a
    failure here is a boundary-condition failure and nothing else.
    """
    params = SimulationParameters(Nx=8, Ny=8, Nz=8, kappa=KAPPA)
    device = Device(params, applied_field=AppliedField(**{component: B0}))
    fields = _vacuum_steady_field(
        params, device.idx, _empty_box(params), **{component.lower(): B0}
    )
    measured = fields[{"Bx": 0, "By": 1, "Bz": 2}[component]]

    with phys_log.test(
        f"test_empty_box_reproduces_the_applied_field[{component}]",
        {"Nx": 8, "Ny": 8, "Nz": 8, "applied": B0, "component": component},
        "the applied-field boundary condition is exact in vacuum",
    ) as log:
        log["steady_state_residual"] = fields[3]
        log.check_below(
            "least-squares residual of the steady state", fields[3], 1e-9,
            detail="a non-zero residual would mean no steady state exists",
        )
        log.check_below(
            f"max |{component} - applied| / applied",
            float(np.abs(measured - B0).max() / B0), 1e-12,
            detail="vacuum carries the applied field unchanged",
        )


@pytest.mark.parametrize(
    "axis,component",
    [("x", "Bz"), ("y", "Bz"), ("z", "Bx"), ("x", "By")],
)
def test_applied_field_on_a_periodic_axis_is_refused(axis, component):
    """A periodic axis cannot carry an applied field, and says so.

    The offset is written onto the ghost links closing the boundary
    plaquettes; a periodic axis has none, its boundary link being
    identified with the far-side one.  Left alone the solver runs
    happily and returns a field that is wrong by 50-100% — measured on
    the empty box above, where the right answer is known exactly.  A
    silent wrong answer is worse than a refusal, so it refuses.
    """
    params = SimulationParameters(
        Nx=6, Ny=6, Nz=6, kappa=KAPPA, **{f"periodic_{axis}": True}
    )
    with pytest.raises(ValueError, match="periodic"):
        Device(params, applied_field=AppliedField(**{component: B0}))


def test_periodic_boundaries_are_fine_without_an_applied_field():
    """The refusal above is about the *combination*, not about periodicity."""
    params = SimulationParameters(
        Nx=6, Ny=6, Nz=6, kappa=KAPPA,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    Device(params, applied_field=AppliedField())


# ────────────────────────────────────────────────────────────
# The Maxwell coefficient is the vacuum's
# ────────────────────────────────────────────────────────────

def test_kappa_contrast_without_current_changes_nothing(phys_log):
    """A κ contrast in a currentless region must not bend the field.

    ``κ²|∇×A|²`` is the field energy ``B²/2μ₀`` in the units set by the
    reference material.  It belongs to the field, so it is the same in
    the metal, in an oxide and in vacuum; a "magnetically different"
    slab of vacuum is not a thing that can be built.  Put a slab of
    declared κ ≠ κ_ref into an otherwise empty box and the field must
    come out uniform anyway.

    The one value that is not merely inert is κ = 0: it removes the last
    term from the φ-equation, freezes **A**, and used to drive Bz to
    -0.52·B₀ in this very configuration.  That is why the declared κ no
    longer reaches the Maxwell term.
    """
    params = SimulationParameters(Nx=8, Ny=8, Nz=12, kappa=KAPPA)
    device = Device(params, applied_field=AppliedField(Bz=B0))
    slab = slice(5 * params.mk, 8 * params.mk)

    worst = {}
    for declared in (0.0, 1.0, 4.0):
        material = _empty_box(params)
        material.kappa[slab] = declared
        *_, bz, residual = _vacuum_steady_field(
            params, device.idx, material, bz=B0
        )
        worst[declared] = float(np.abs(bz - B0).max() / B0)

    with phys_log.test(
        "test_kappa_contrast_without_current_changes_nothing",
        {"Nx": 8, "Ny": 8, "Nz": 12, "kappa_ref": KAPPA, "applied_Bz": B0,
         "slab_z_nodes": "5-7"},
        "the Maxwell coefficient is a property of the vacuum, not of the material",
    ) as log:
        log["max_relative_error_by_declared_kappa"] = {
            str(k): v for k, v in worst.items()
        }
        for declared, error in worst.items():
            log.check_below(
                f"max |Bz - applied| / applied, declared κ = {declared}",
                error, 1e-12,
                detail="no current anywhere, so the field must stay uniform",
            )


# ────────────────────────────────────────────────────────────
# Vacuum outside the stack
# ────────────────────────────────────────────────────────────

_RELAXED: dict[tuple, tuple] = {}


def _stack(pad, lateral, *, layer=3, oxide=2, n_lateral=16):
    """An S/I/S stack with *pad* cells of vacuum above and below."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=layer, kappa=KAPPA),
        insulator=Layer(thickness_z=oxide, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=layer, kappa=KAPPA),
        vacuum_below=pad,
        vacuum_above=pad,
        lateral_margin=lateral,
    )
    params = SimulationParameters(
        Nx=n_lateral, Ny=n_lateral, Nz=trilayer.Nz, kappa=KAPPA
    )
    device = Device(
        params, applied_field=AppliedField(Bz=B0, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    return params, device, trilayer


def _relaxed(pad, lateral, **kwargs):
    """Relax a stack once and reuse it — several checks read the same run."""
    key = (pad, lateral, tuple(sorted(kwargs.items())))
    if key not in _RELAXED:
        params, device, trilayer = _stack(pad, lateral, **kwargs)
        bz, residual = _relax(params, device)
        _RELAXED[key] = (params, trilayer, bz, residual)
    return _RELAXED[key]


def _midplane_line(bz, params, trilayer):
    """``Bz/B0`` along x through the middle of the bottom metal layer."""
    centre = (params.Ny - 1) // 2
    k_node = max(trilayer.z_ranges()["bottom"][0], 1)
    return bz[:, centre, k_node - 1] / B0        # interior arrays start at k = 1


def _is_superconducting(params, trilayer):
    """Which entries of that line are plaquettes lying *inside* the metal.

    ``Bz`` is a plaquette quantity: ``Bz[i]`` is the flux through the
    square spanning nodes ``i`` and ``i+1``, so it sits at ``i + ½``
    and is inside the metal only when *both* its corners are.  With a
    lateral margin of *m* the metal nodes are ``m … Nx - m``, so the
    metal plaquettes are ``m … Nx - m - 1`` — one fewer, and centred
    half a cell to the left of the nodes.  Taking the node range here
    instead picks up the edge plaquette, which is exactly where the
    flux crowds, and reads the crowding peak as if it were inside
    the film.
    """
    margin = trilayer.lateral_margin
    i = np.arange(1, params.Nx)                  # interior plaquettes
    if margin == 0:
        return np.ones(i.shape, dtype=bool)
    return (i >= margin) & (i <= params.Nx - margin - 1)


def test_lateral_vacuum_unpins_the_film_edge(phys_log):
    """Without a lateral margin the film's own edge carries the applied field.

    A perpendicular applied field is imposed as flux through the
    boundary plaquettes on the *x and y* walls of the box.  When the
    film runs all the way to those walls, that condition lands on
    superconducting nodes and prescribes the field inside the metal:
    the outermost metal node reads exactly the applied field, however
    well the film screens, because nothing was ever solved for there.

    Give the film a margin of vacuum and the same condition lands on
    vacuum instead, where it is the correct far-field statement, and
    the metal edge is free to be screened.
    """
    _, tri_bare, bz_bare, _ = _relaxed(0, 0)
    params, tri_pad, bz_pad, _ = _relaxed(5, 3)

    bare_line = _midplane_line(bz_bare, params, tri_bare)
    pad_line = _midplane_line(bz_pad, params, tri_pad)
    bare_edge = float(bare_line[_is_superconducting(params, tri_bare)].max())
    pad_edge = float(pad_line[_is_superconducting(params, tri_pad)].max())

    with phys_log.test(
        "test_lateral_vacuum_unpins_the_film_edge",
        {"Nx": params.Nx, "kappa": KAPPA, "applied_Bz": B0,
         "lateral_margin_bare": 0, "lateral_margin_padded": 3},
        "the applied-field condition must land on vacuum, not on the metal",
    ) as log:
        log["bz_line_no_margin"] = [float(v) for v in bare_line]
        log["bz_line_with_margin"] = [float(v) for v in pad_line]
        log.check_close(
            "max Bz over metal nodes / applied, no lateral margin",
            bare_edge, 1.0, atol=1e-9,
            detail="prescribed, not solved for — this is the artefact",
        )
        log.check_below(
            "max Bz over metal nodes / applied, with a lateral margin",
            pad_edge, 0.999,
            detail="every metal node is now screened below the applied field",
        )


def test_flux_crowds_into_the_vacuum_beside_the_film(phys_log):
    """Flux expelled from the film must pile up just outside its edge.

    The film pushes flux out of itself, and that flux has to go
    somewhere: it concentrates in the vacuum beside the edge, so the
    field there *exceeds* the applied field.  It is the field's
    response to the film having a boundary at all, and a film with no
    vacuum around it cannot show it — there is nowhere for the flux
    to crowd into, and the wall holds the field at the applied value.
    """
    params, trilayer, bz, residual = _relaxed(5, 3)
    line = _midplane_line(bz, params, trilayer)
    vacuum = ~_is_superconducting(params, trilayer)

    peak = float(line[vacuum].max())
    centre = float(line[len(line) // 2])

    with phys_log.test(
        "test_flux_crowds_into_the_vacuum_beside_the_film",
        {"Nx": params.Nx, "Nz": params.Nz, "kappa": KAPPA,
         "applied_Bz": B0, "lateral_margin": 3, "vacuum_cells": 5},
        "expelled flux crowds into the vacuum beside the film",
    ) as log:
        log["bz_line_over_applied"] = [float(v) for v in line]
        log["max_abs_dXdt"] = residual
        log.check_below(
            "max |dX/dt| at the final state", residual, 1e-3,
            detail="the line below is read at a steady state",
        )
        log.check_above(
            "peak Bz in the vacuum beside the film / applied", peak, 1.005,
            detail="flux pushed out of the film has to go somewhere",
        )
        log.check_below(
            "Bz at the film centre / applied", centre, 0.9,
            detail="and it came from here",
        )


def test_far_field_converges(phys_log):
    """The far-field error must fall as the vacuum padding grows.

    The boundary imposes the applied field at a finite distance, so
    the padding is an approximation to "infinitely far away" and its
    error has to be shown to shrink.  Measure how far the field
    directly above the film has been dragged from the applied value,
    at three paddings.
    """
    errors, residuals = {}, {}
    for pad in (2, 4, 8):
        params, _, bz, residual = _relaxed(pad, 3)
        centre = (params.Nx - 1) // 2
        errors[pad] = float(abs(bz[centre, centre, -1] / B0 - 1.0))
        residuals[pad] = residual

    with phys_log.test(
        "test_far_field_converges",
        {"paddings": [2, 4, 8], "kappa": KAPPA, "applied_Bz": B0,
         "Nx": 16, "lateral_margin": 3},
        "the far-field boundary condition converges as the vacuum grows",
    ) as log:
        log["far_field_error_by_padding"] = {str(k): v for k, v in errors.items()}
        log["max_abs_dXdt"] = {str(k): v for k, v in residuals.items()}
        log.check_below(
            "far-field error at 4 cells / error at 2 cells",
            errors[4] / max(errors[2], 1e-300), 1.0,
            detail="doubling the padding must not make the far field worse",
        )
        log.check_below(
            "far-field error at 8 cells / error at 4 cells",
            errors[8] / max(errors[4], 1e-300), 1.0,
        )


def test_padded_stack_is_mirror_symmetric(phys_log):
    """Equal padding and equal metal layers must give an exactly symmetric field.

    A statement the solver has no room to be approximately right
    about.  The stack is mirror-symmetric about its mid-plane and so
    is the padding, so the field must be too — to round-off, not to
    a tolerance.  It fails if the padding is laid down asymmetrically,
    or if the interface nodes between vacuum and metal are handed to
    one side.
    """
    params, trilayer, bz, _ = _relaxed(5, 3)
    centre = (params.Nx - 1) // 2
    profile = bz[centre, centre, :] / B0
    asymmetry = float(np.abs(profile - profile[::-1]).max())

    with phys_log.test(
        "test_padded_stack_is_mirror_symmetric",
        {"Nz": params.Nz, "vacuum_cells": 5, "layers": "3/2/3",
         "kappa": KAPPA, "applied_Bz": B0},
        "a mirror-symmetric stack in a symmetric box gives a symmetric field",
    ) as log:
        log["bz_profile_over_applied"] = [float(v) for v in profile]
        log.check_below(
            "max |Bz(z) - Bz(-z)| / applied", asymmetry, 1e-12,
            detail="exact symmetry, not an approximate one",
        )
