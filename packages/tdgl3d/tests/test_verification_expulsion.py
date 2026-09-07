"""Flux expulsion by an S/I/S ring, and the field at which it fails.

A square hole is carved through both superconducting layers of an S/I/S stack,
leaving the oxide continuous. The ring around the hole is multiply connected, so
the enclosed fluxoid is quantised: it holds ``n = 0`` while the screening
current in the arms can sustain it, and steps to a non-zero integer once it
cannot.

What is checked here:

* the ring is actually superconducting — a stack whose layers are thinner than a
  coherence length is fully pair-broken by the oxide, and a "flux expulsion"
  measured on such a device is a phase read off numerical noise;
* the enclosed fluxoid is an exact integer at every instant and does not depend
  on the contour used to measure it;
* below threshold the ring holds ``n = 0`` for the whole run *and* relaxes to a
  fixed point — a state that has merely not broken yet is not an expelled state;
* above threshold flux enters in whole quanta, and the entry time falls
  monotonically with field: that divergence as the threshold is approached from
  above is what distinguishes a stability boundary from a run that was too short;
* a larger hole expels less, with ``B_exp·A_hole`` of order Φ₀.

The threshold is a *dynamical* stability boundary, so it is only defined
relative to a hold time; every check states the hold time it used.  The grid is
deliberately coarse (h = ξ) to keep the scan affordable — the quantitative
number belongs to ``docs/figures/sis_hole_expulsion.py``, which resolves the
in-plane structure more finely; what is asserted here is the behaviour that must
survive refinement.
"""

from __future__ import annotations

import numpy as np
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d.analysis.expulsion import (
    expulsion_field,
    first_entry_time,
    fluxoid_history,
    rectangular_contour,
)
from tdgl3d.core.solution import Solution
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.rhs import eval_f
from tdgl3d.solvers.integrators import forward_euler

from .physics_helpers import applied_boundary, cfl_limit, expand_state

PHI0 = 2.0 * np.pi

# Geometry.  Arms 3 ξ wide either side of the hole; superconducting layers 4 ξ
# thick, which is the binding constraint: at 1 ξ the oxide's pair-breaking
# reaches straight through and |ψ| collapses to ~1e-4 everywhere (see
# test_the_ring_is_superconducting).  The oxide is written with the metal's κ,
# but that is cosmetic: κ on a non-superconducting layer does not reach the
# Maxwell term — see test_declared_oxide_kappa_does_not_change_the_field in
# test_physics_validation.
ARM = 3.0
KAPPA = 2.0
H = 1.0
SC_CELLS = 4
INSULATOR_CELLS = 2
HOLD_TIME = 30.0

_RUNS: dict[tuple, tuple] = {}


def _build(applied_bz: float, hole_size: float):
    length = hole_size + 2 * ARM
    n_cells = int(round(length / H))
    trilayer = Trilayer(
        bottom=Layer(thickness_z=SC_CELLS, kappa=KAPPA),
        insulator=Layer(thickness_z=INSULATOR_CELLS, kappa=KAPPA, is_superconductor=False),
        top=Layer(thickness_z=SC_CELLS, kappa=KAPPA),
    )
    params = SimulationParameters(
        Nx=n_cells, Ny=n_cells, Nz=trilayer.Nz, hx=H, hy=H, hz=H, kappa=KAPPA
    )
    device = Device(
        params,
        applied_field=AppliedField(Bz=applied_bz, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    lo, hi = 0.5 * (length - hole_size), 0.5 * (length + hole_size)
    square = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    z_ranges = trilayer.z_ranges()
    device.add_hole(square, z_range=z_ranges["bottom"])
    device.add_hole(square, z_range=z_ranges["top"])
    return params, device, trilayer, (lo, hi, lo, hi)


def ring_run(applied_bz: float, hole_size: float = 4.0, t_stop: float = HOLD_TIME):
    """Relax an S/I/S ring at a fixed field; memoised across checks."""
    key = (applied_bz, hole_size, t_stop)
    if key in _RUNS:
        return _RUNS[key]

    params, device, trilayer, hole_bounds = _build(applied_bz, hole_size)
    idx = device.idx
    boundary = applied_boundary(params, idx, bz=applied_bz)
    dt = 0.9 * cfl_limit(params)
    times, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data, params, idx,
        lambda t, X: boundary, 0.0, t_stop, dt,
        save_every=max(1, int(t_stop / dt / 15)), progress=False,
        material=device.material,
    )
    solution = Solution(times=times, states=states, params=params, idx=idx, device=device)
    slice_z = max(trilayer.z_ranges()["bottom"][1] // 2 - 1, 0)
    contour = rectangular_contour(hole_bounds, params, margin=1.5)
    history = fluxoid_history(solution, device, contour, slice_z=slice_z)

    _RUNS[key] = (params, device, trilayer, solution, history, hole_bounds, slice_z, boundary)
    return _RUNS[key]


def _scan(hole_size: float, fields: list[float]):
    finals, entries = [], []
    for applied_bz in fields:
        _, _, _, solution, history, _, _, _ = ring_run(applied_bz, hole_size)
        finals.append(float(history[-1]))
        entries.append(first_entry_time(solution.times, history))
    return expulsion_field(fields, finals, entries, hold_time=HOLD_TIME)


SCAN_4 = [0.05, 0.15, 0.22, 0.32, 0.45, 0.6]
SCAN_6 = [0.15, 0.22]


# ---------------------------------------------------------------------------
# The device has to be a superconductor first
# ---------------------------------------------------------------------------


def test_the_ring_is_superconducting(phys_log):
    """The arms sustain a condensate, so the phase they carry means something.

    Proximity to the oxide suppresses ψ over roughly a coherence length on each
    side of the interface. Layers thinner than that are suppressed all the way
    through, and everything downstream — fluxoid, expulsion field, vortex
    entry — is then read off the phase of a ~1e-4 order parameter. The check is
    cheap and it is what makes the rest of this file mean anything.
    """
    _, device, _, solution, _, _, _, _ = ring_run(0.05)
    psi = np.abs(solution.psi(step=-1))
    superconducting = device.material.interior_sc_mask > 0.0
    in_arms = psi[superconducting]

    with phys_log.test(
        "test_the_ring_is_superconducting",
        {"hole": 4.0, "arm": ARM, "sc_thickness": SC_CELLS * H, "kappa": KAPPA},
        "the superconducting layers must be thicker than the proximity length",
    ) as log:
        log["psi_max_in_arms"] = float(in_arms.max())
        log["psi_mean_in_arms"] = float(in_arms.mean())
        log["psi_max_in_insulator_and_hole"] = float(psi[~superconducting].max())
        log.check_above(
            "max |ψ| in the superconducting layers", float(in_arms.max()), 0.9,
            detail="the middle of a 4 ξ layer must recover the bulk condensate",
        )
        log.check_above(
            "mean |ψ| in the superconducting layers", float(in_arms.mean()), 0.5,
        )
        log.check_below(
            "max |ψ| in the oxide and the hole", float(psi[~superconducting].max()), 0.25,
        )


def test_the_relaxed_ring_is_symmetric(phys_log):
    """A centred square hole in a symmetric stack gives a symmetric solution.

    The device is invariant under x → −x, y → −y, z → −z and a 90° rotation, so
    |ψ| and Bz must be too — to round-off, since the run starts from a noiseless
    state.  Two geometric conventions used to break this at the 1e-3 level and
    both were invisible in the fields alone:

    * ray casting is half-open, so a hole given as ``[3, 7]`` carved nodes 4…7
      and sat half a cell off centre;
    * layer thicknesses are in cells but materials live on nodes, and assigning
      each node to the cell range containing it gave the top layer one more
      superconducting node than the bottom.

    Node-centred ``ψ`` reflects onto itself; plaquette-centred ``Bz`` needs its
    last anchor dropped first (see ``docs/notes/PHYSICS_CONVENTIONS.md``).
    """
    applied_bz = 0.05
    params, device, _, solution, _, _, _, boundary = ring_run(applied_bz)
    nx, ny, nz = params.Nx - 1, params.Ny - 1, params.Nz - 1

    psi = np.abs(solution.psi(step=-1)).reshape(nx, ny, nz)
    _, phi_x, phi_y, phi_z = expand_state(solution.states[:, -1], params, device.idx, boundary)
    field = eval_bfield_full(phi_x, phi_y, phi_z, params, device.idx)[2].reshape(nx, ny, nz)
    field_core = field[:-1, :-1, :]

    psi_scale = float(psi.max())
    field_scale = float(np.abs(field_core).max())

    with phys_log.test(
        "test_the_relaxed_ring_is_symmetric",
        {"hole": 4.0, "arm": ARM, "kappa": KAPPA, "Bz": applied_bz},
        "the solution must inherit every symmetry of the device",
    ) as log:
        log["psi_scale"] = psi_scale
        log["Bz_scale"] = field_scale
        log.check_above("|ψ| scale (non-trivial)", psi_scale, 0.5)
        log.check_above("Bz scale (non-trivial)", field_scale, 1e-3)
        for label, mirrored in (
            ("x → −x", psi[::-1]),
            ("y → −y", psi[:, ::-1]),
            ("z → −z", psi[:, :, ::-1]),
            ("90° rotation", np.rot90(psi, axes=(0, 1))),
        ):
            log.check_below(
                f"max |ψ| asymmetry under {label}",
                float(np.max(np.abs(psi - mirrored))), 1e-12,
            )
        for label, mirrored in (
            ("x → −x", field_core[::-1]),
            ("y → −y", field_core[:, ::-1]),
            ("z → −z", field_core[:, :, ::-1]),
            ("90° rotation", np.rot90(field_core, axes=(0, 1))),
        ):
            log.check_below(
                f"max Bz asymmetry under {label}",
                float(np.max(np.abs(field_core - mirrored))), 1e-12 * field_scale,
            )


# ---------------------------------------------------------------------------
# The expelled branch
# ---------------------------------------------------------------------------


def test_ring_expels_flux_below_threshold(phys_log):
    """At low field the ring holds n = 0 and settles into a genuine fixed point."""
    applied_bz = 0.05
    params, device, _, solution, history, _, _, boundary = ring_run(applied_bz)
    residual = float(
        np.max(np.abs(eval_f(solution.states[:, -1], params, device.idx, boundary,
                             material=device.material)))
    )

    with phys_log.test(
        "test_ring_expels_flux_below_threshold",
        {"hole": 4.0, "arm": ARM, "kappa": KAPPA, "Bz": applied_bz, "t_hold": HOLD_TIME},
        "below threshold the multiply-connected ring keeps the enclosed fluxoid at zero",
    ) as log:
        log["fluxoid_history"] = [float(v) for v in history]
        log["residual"] = residual
        log.check_below(
            "max |fluxoid| over the whole run", float(np.max(np.abs(history))), 1e-9,
            units="Φ₀", detail="not one quantum enters at any time",
        )
        log.check_below(
            "max |dX/dt| at the end", residual, 1e-3,
            detail="the expelled state is a fixed point, not a slow transient",
        )


def test_fluxoid_does_not_depend_on_the_contour(phys_log):
    """The enclosed fluxoid is topological: contour shape and size cannot matter."""
    applied_bz = 0.45
    params, device, _, solution, _, hole_bounds, slice_z, _ = ring_run(applied_bz)

    values = {}
    for margin in (1.0, 1.5, 2.0):
        contour = rectangular_contour(hole_bounds, params, margin=margin)
        values[margin] = float(
            fluxoid_history(solution, device, contour, slice_z=slice_z)[-1]
        )
    spread = max(values.values()) - min(values.values())
    reference = values[1.5]

    with phys_log.test(
        "test_fluxoid_does_not_depend_on_the_contour",
        {"hole": 4.0, "Bz": applied_bz, "margins": [1.0, 1.5, 2.0]},
        "the fluxoid counts what the contour encloses, not how it is drawn",
    ) as log:
        log["fluxoid_by_margin"] = {str(k): v for k, v in values.items()}
        log.check_above(
            "|fluxoid| (non-trivial, so the check has content)", abs(reference), 0.5,
        )
        log.check_below("spread across contour margins", spread, 1e-9, units="Φ₀")
        log.check_below(
            "|fluxoid − nearest integer|", abs(reference - round(reference)), 1e-9,
        )


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


def test_flux_enters_in_whole_quanta(phys_log):
    """Above threshold the fluxoid steps through integers, never fractions.

    The steps here come in twos: the device is C4-symmetric and the run starts
    from a noiseless state, so the possible entry points are degenerate and
    quanta arrive in symmetry-related pairs. That is a property of this
    idealised geometry, not of the quantisation — what is asserted is
    integrality, not a step of one.
    """
    applied_bz = 0.6
    _, _, _, solution, history, _, _, _ = ring_run(applied_bz)
    deviation = float(np.max(np.abs(history - np.rint(history))))

    with phys_log.test(
        "test_flux_enters_in_whole_quanta",
        {"hole": 4.0, "Bz": applied_bz, "t_hold": HOLD_TIME},
        "fluxoid quantisation holds instant by instant, including mid-entry",
    ) as log:
        log["fluxoid_history"] = [float(v) for v in history]
        log["entry_time"] = first_entry_time(solution.times, history)
        log.check_close("fluxoid at t = 0", float(history[0]), 0.0, atol=1e-9)
        log.check_above("final |fluxoid|", abs(float(history[-1])), 1.0, units="Φ₀")
        log.check_below("max |fluxoid − nearest integer|", deviation, 1e-9)
        log.check_below(
            "largest decrease along the history",
            float(-np.min(np.diff(np.abs(history)))), 1e-9,
            detail="flux accumulates; it does not leak back out at fixed field",
        )


def test_expulsion_threshold_is_bracketed(phys_log):
    """Scan the applied field and bracket the expulsion threshold."""
    hole_size = 4.0
    result = _scan(hole_size, SCAN_4)
    timed = [t for t in result.entry_times if t is not None]
    area = hole_size**2

    with phys_log.test(
        "test_expulsion_threshold_is_bracketed",
        {"hole": hole_size, "arm": ARM, "kappa": KAPPA, "h": H, "t_hold": HOLD_TIME},
        "the ring expels flux up to a definite field and admits quanta above it",
    ) as log:
        log["fields"] = result.fields
        log["final_fluxoid"] = result.final_fluxoid
        log["entry_times"] = [None if t is None else float(t) for t in result.entry_times]
        log["summary"] = result.summary()
        log.check_above("scan brackets the threshold", float(result.threshold is not None), 1.0)
        log.check_within(
            "expulsion field B_exp", result.threshold, 0.05, 0.9,
            units="Φ₀/2πξ²",
            detail="above the lowest field scanned and below H_c2 = 1",
        )
        log.check_within(
            "B_exp · A_hole / Φ₀", result.threshold * area / PHI0, 0.2, 3.0,
            detail="the threshold is set by the fluxoid scale, not by the grid",
        )
        log.check_above("fields that admitted flux", float(len(timed)), 3.0)
        log.check_below(
            "largest increase in entry time with field",
            float(np.max(np.diff(timed))), 0.0,
            units="τ_GL",
            detail=f"entry times {timed} must fall as the field rises",
        )


def test_a_larger_hole_expels_less(phys_log):
    """B_exp falls as the hole grows, at fixed arm width.

    Both rings have 3 ξ arms, so their arms carry the same critical current;
    what changes is the flux the hole gathers per unit applied field. The 6 ξ
    hole must therefore give way at a lower field than the 4 ξ one.

    The comparison holds while the hole is at least as wide as the arms. Shrink
    it much below that and the loop's enclosed flux is dominated by the arms
    rather than by the hole, and the area argument stops applying.
    """
    small = _scan(4.0, SCAN_4)
    large = _scan(6.0, SCAN_6)

    with phys_log.test(
        "test_a_larger_hole_expels_less",
        {"holes": [4.0, 6.0], "arm": ARM, "kappa": KAPPA, "t_hold": HOLD_TIME},
        "the expulsion field is set by the hole area at fixed arm width",
    ) as log:
        log["B_exp_hole4"] = small.threshold
        log["B_exp_hole6"] = large.threshold
        log["summary_hole4"] = small.summary()
        log["summary_hole6"] = large.summary()
        log.check_above(
            "both scans bracket a threshold",
            float(small.threshold is not None and large.threshold is not None), 1.0,
        )
        log.check_below(
            "B_exp(6 ξ hole) / B_exp(4 ξ hole)", large.threshold / small.threshold, 0.9,
            detail="a larger hole gathers more flux per unit field, so it gives way sooner",
        )
        for size, result in ((4.0, small), (6.0, large)):
            log.check_within(
                f"B_exp · A_hole / Φ₀ for the {size:g} ξ hole",
                result.threshold * size**2 / PHI0, 0.2, 3.0,
                detail="the threshold sits within a factor of a few of one flux quantum",
            )
