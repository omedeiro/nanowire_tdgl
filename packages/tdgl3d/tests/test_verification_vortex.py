"""Vortex and flux-quantisation verification.

On the lattice, fluxoid quantisation is not an approximate statement.  For any
closed loop of links,

    Σ wrap(Δθ − φ) + Σ φ = 2π n,  n ∈ ℤ

exactly, because the bare phases cancel pairwise around the loop.  The left-hand
side is the discrete ``∮(∇θ − A)·dl + ∮A·dl``, so ``n`` is the enclosed number
of flux quanta.  That gives three checks with essentially no tolerance:

* every plaquette's vorticity is an integer to round-off;
* the fluxoid around any contour equals the vorticity it encloses (lattice
  Stokes theorem), independent of the contour's shape;
* the sign of every vortex follows the sign of the applied field.

The last one is the cheapest way to notice a broken gauge coupling: a solver in
which the supercurrent and the covariant Laplacian disagree about the sign of
``A`` still nucleates vortices and still screens, but produces a mix of ``+1``
and ``−1`` windings in a uniform field, which no superconductor does.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d import AppliedField, Device, SimulationParameters, solve
from tdgl3d.analysis.vortex_counting import (
    count_vortices_plaquette,
    count_vortices_polygon,
    plaquette_vorticity,
)
from tdgl3d.physics.bfield import eval_bfield_full

from .physics_helpers import applied_boundary, expand_state

PHI0 = 2.0 * np.pi


_RELAXED: dict[tuple, tuple] = {}


def _vortex_state(bz: float, n_cells: int = 20, t_stop: float = 60.0, kappa: float = 2.0):
    """Relax a square device in a uniform field until vortices have settled.

    Results are memoised: several checks interrogate the same final state and
    re-running the integration for each would dominate the suite's runtime.
    """
    key = (bz, n_cells, t_stop, kappa)
    if key not in _RELAXED:
        params = SimulationParameters(Nx=n_cells, Ny=n_cells, Nz=1, kappa=kappa)
        device = Device(params, applied_field=AppliedField(Bz=bz, t_on_fraction=1.0))
        solution = solve(
            device, t_start=0.0, t_stop=t_stop, dt=0.01, method="euler",
            progress=False, log_metadata=False, save_every=100,
        )
        _RELAXED[key] = (params, device, solution)
    return _RELAXED[key]


# ---------------------------------------------------------------------------
# Exact quantisation
# ---------------------------------------------------------------------------


def test_plaquette_vorticity_is_an_exact_integer(phys_log):
    """Vorticity is a winding number, so it must be integral to round-off."""
    params, device, solution = _vortex_state(bz=0.5)
    vorticity, _ = plaquette_vorticity(solution, step=-1)
    deviation = float(np.max(np.abs(vorticity - np.rint(vorticity))))
    charged = int(np.count_nonzero(np.rint(vorticity)))

    with phys_log.test(
        "test_plaquette_vorticity_is_an_exact_integer",
        {"Nx": 20, "kappa": 2.0, "Bz": 0.5},
        "Σ wrap(Δθ − φ) + Φ_plaquette is exactly 2π × integer",
    ) as log:
        log["n_charged_plaquettes"] = charged
        log["vorticity_values"] = sorted(
            int(v) for v in np.unique(np.rint(vorticity))
        )
        log.check_above("plaquettes carrying vorticity", float(charged), 1.0)
        log.check_below("max |vorticity − nearest integer|", deviation, 1e-10)


@pytest.mark.parametrize("bz", [0.5, -0.5])
def test_vortex_winding_sign_follows_the_applied_field(bz, phys_log):
    """In a uniform field every vortex has the same sign, matching the field.

    Mixed ``±1`` windings in a uniform applied field are unphysical: they signal
    that the supercurrent source and the covariant Laplacian disagree about the
    sign of the vector potential.
    """
    params, device, solution = _vortex_state(bz=bz)
    n_vortices, _, windings = count_vortices_plaquette(solution, device, step=-1)
    windings = np.rint(np.real(windings)).astype(int)

    with phys_log.test(
        f"test_vortex_winding_sign_follows_the_applied_field[Bz={bz}]",
        {"Nx": 20, "kappa": 2.0, "Bz": bz},
        "vortices in a uniform field are all of the same chirality as the field",
    ) as log:
        log["n_vortices"] = int(n_vortices)
        log["winding_values"] = sorted(int(v) for v in np.unique(windings))
        log.check_above("vortices detected", float(n_vortices), 1.0)
        log.check_close(
            "distinct winding values", float(np.unique(windings).size), 1.0, atol=0.0,
            detail=f"found {sorted(set(int(w) for w in windings))}",
        )
        log.check_close(
            "common winding", float(windings[0]), float(np.sign(bz)), atol=0.0,
            detail="winding sign must match the sign of the applied field",
        )
        log.check_close(
            "max |winding|", float(np.max(np.abs(windings))), 1.0, atol=0.0,
            detail="singly quantised vortices at this field",
        )


def test_fluxoid_equals_enclosed_vorticity_for_any_contour(phys_log):
    """Lattice Stokes: ∮ around a contour = Σ of the plaquettes it encloses.

    Checked for nested square contours and for a non-convex staircase, so the
    result cannot depend on the contour shape.
    """
    params, device, solution = _vortex_state(bz=0.5)
    vorticity, _ = plaquette_vorticity(solution, step=-1)
    rounded = np.rint(vorticity)

    contours = {}
    for pad in (2, 4, 6):
        lo, hi = 1 + pad, params.Nx - 1 - pad
        contours[f"square_pad{pad}"] = (
            np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=float),
            (lo, hi, lo, hi),
        )

    worst_integer, worst_stokes, values = 0.0, 0.0, {}
    for name, (polygon, (i0, i1, j0, j1)) in contours.items():
        fluxoid = count_vortices_polygon(solution, device, polygon)
        enclosed = float(rounded[i0 - 1 : i1 - 1, j0 - 1 : j1 - 1].sum())
        values[name] = {"fluxoid": fluxoid, "enclosed_vorticity": enclosed}
        worst_integer = max(worst_integer, abs(fluxoid - round(fluxoid)))
        worst_stokes = max(worst_stokes, abs(fluxoid - enclosed))

    # An L-shaped (non-convex) contour enclosing the same region as square_pad2
    # minus one corner block must give the vorticity of that reduced region.
    lo, hi = 3, params.Nx - 3
    mid = (lo + hi) // 2
    staircase = np.array(
        [[lo, lo], [hi, lo], [hi, mid], [mid, mid], [mid, hi], [lo, hi]], dtype=float
    )
    staircase_fluxoid = count_vortices_polygon(solution, device, staircase)
    enclosed_l = float(
        rounded[lo - 1 : hi - 1, lo - 1 : mid - 1].sum()
        + rounded[lo - 1 : mid - 1, mid - 1 : hi - 1].sum()
    )

    with phys_log.test(
        "test_fluxoid_equals_enclosed_vorticity_for_any_contour",
        {"Nx": 20, "kappa": 2.0, "Bz": 0.5},
        "the fluxoid is a topological invariant of the region, not of the path",
    ) as log:
        log["contours"] = values
        log["staircase_fluxoid"] = staircase_fluxoid
        log["staircase_enclosed"] = enclosed_l
        log.check_below("max |fluxoid − nearest integer|", worst_integer, 1e-9)
        log.check_below("max |fluxoid − enclosed vorticity|", worst_stokes, 1e-9)
        log.check_below(
            "|staircase fluxoid − enclosed vorticity|",
            abs(staircase_fluxoid - enclosed_l), 1e-9,
        )


# ---------------------------------------------------------------------------
# Vortex thermodynamics
# ---------------------------------------------------------------------------


def test_no_vortices_in_the_meissner_state(phys_log):
    """Well below H_c1 the equilibrium state carries no vorticity at all."""
    params, device, solution = _vortex_state(bz=0.03, n_cells=16, t_stop=40.0)
    vorticity, _ = plaquette_vorticity(solution, step=-1)
    n_vortices, _, _ = count_vortices_plaquette(solution, device, step=-1)
    psi_min = float(np.min(np.abs(solution.psi(step=-1))))

    with phys_log.test(
        "test_no_vortices_in_the_meissner_state",
        {"Nx": 16, "kappa": 2.0, "Bz": 0.03},
        "below H_c1 flux is expelled and the order parameter stays uniform",
    ) as log:
        log["n_vortices"] = int(n_vortices)
        log["psi_min"] = psi_min
        log.check_close("vortex count", float(n_vortices), 0.0, atol=0.0)
        log.check_below("max |vorticity| anywhere", float(np.max(np.abs(vorticity))), 1e-9)
        log.check_above(
            "min |ψ|", psi_min, 0.95,
            detail="no cores means no suppression of the order parameter",
        )


def test_vortex_count_increases_with_the_applied_field(phys_log):
    """More field, more vortices — and never more than ``B·A/Φ₀`` of them.

    The equilibrium vortex density in the mixed state rises with ``B``, but
    screening keeps the interior field below the applied one, so the applied
    flux is only an upper bound on the count.  Note that the check deliberately
    does *not* equate the count with the interior magnetic flux: in a sample only
    a few λ across, a large part of that flux is carried by the Meissner
    currents near the edges rather than by vortex cores.
    """
    fields = (0.35, 0.5, 0.7)
    counts, screening, bounds = [], [], []
    for bz in fields:
        params, device, solution = _vortex_state(bz=bz, n_cells=16, t_stop=40.0)
        n_vortices, _, _ = count_vortices_plaquette(solution, device, step=-1)
        counts.append(int(n_vortices))

        _, px, py, pz = expand_state(
            solution.states[:, -1], params, solution.idx,
            applied_boundary(params, solution.idx, bz=bz),
        )
        nx_int, ny_int = params.Nx - 1, params.Ny - 1
        field = eval_bfield_full(px, py, pz, params, solution.idx)[2].reshape(nx_int, ny_int)
        # Drop the boundary ring, which is pinned to the applied field.
        screening.append(float(field[:-1, :-1].mean()) / bz)
        bounds.append(bz * (params.Nx * params.hx) * (params.Ny * params.hy) / PHI0)

    with phys_log.test(
        "test_vortex_count_increases_with_the_applied_field",
        {"Nx": 16, "kappa": 2.0, "Bz_values": list(fields)},
        "the mixed state admits more flux quanta as the applied field rises",
    ) as log:
        log["vortex_counts"] = counts
        log["applied_flux_quanta"] = [round(b, 2) for b in bounds]
        log["interior_field_over_applied"] = [round(s, 4) for s in screening]
        log.check_below(
            "largest decrease in count along the sweep",
            float(-np.min(np.diff(counts))), 0.0,
            detail=f"counts {counts} at Bz {list(fields)} must not decrease",
        )
        log.check_above(
            "increase from the lowest to the highest field",
            float(counts[-1] - counts[0]), 1.0,
        )
        for bz, count, bound in zip(fields, counts, bounds):
            log.check_below(
                f"count / (B·A/Φ₀) at Bz = {bz}", count / bound, 1.0,
                detail="screening keeps the interior field below the applied field",
            )
        for bz, ratio in zip(fields, screening):
            log.check_below(
                f"mean interior Bz / applied at Bz = {bz}", ratio, 1.0,
                detail="the sample still screens in the mixed state",
            )


def test_vortices_grow_from_zero_and_saturate(phys_log):
    """Nucleation dynamics: no vortices at t=0, entry, then a steady count."""
    bz = 0.5
    params, device, solution = _vortex_state(bz=bz)

    counts, times = [], []
    for step in range(0, solution.n_steps, max(1, solution.n_steps // 12)):
        n_v, _, _ = count_vortices_plaquette(solution, device, step=step)
        counts.append(n_v)
        times.append(float(solution.times[step]))
    counts = np.array(counts)

    tail = counts[-max(2, len(counts) // 4):]
    first_entry = next((t for t, c in zip(times, counts) if c > 0), None)

    with phys_log.test(
        "test_vortices_grow_from_zero_and_saturate",
        {"Nx": 20, "kappa": 2.0, "Bz": bz, "t_stop": 60.0},
        "vortices must nucleate from the uniform state and reach a steady number",
    ) as log:
        log["times"] = times
        log["vortex_counts"] = counts.tolist()
        log["t_first_vortex"] = first_entry
        log.check_close("vortex count at t = 0", float(counts[0]), 0.0, atol=0.0)
        log.check_above("final vortex count", float(counts[-1]), 1.0)
        log.check_below(
            "time of first vortex entry", float(first_entry), 30.0, units="τ_GL",
        )
        log.check_below(
            "std/mean of the count over the final quarter",
            float(np.std(tail)) / max(float(np.mean(tail)), 1.0), 0.25,
            detail="the vortex population must settle rather than keep growing",
        )
