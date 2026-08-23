"""Flux expulsion by a superconducting ring, and the field at which it fails.

A hole in a superconducting film is a multiply-connected region, so the fluxoid
threading it is quantised: for any contour drawn in the superconductor around
the hole,

.. math::
    \\oint (\\nabla\\theta - A)\\cdot dl \\; + \\; \\oint A\\cdot dl \\; = \\; 2\\pi n ,
    \\qquad n \\in \\mathbb{Z}.

Below a threshold applied field the ring holds ``n = 0``: it circulates a
screening current that keeps the *fluxoid* at zero even though magnetic flux
does thread the hole.  Above the threshold that current exceeds what the
superconducting arms can carry, a vortex crosses one of them, and ``n`` steps to
a non-zero integer.

The threshold is what these helpers measure.  It is a **dynamical stability
boundary, not a thermodynamic one**: in deterministic TDGL the ``n = 0`` branch
remains a linearly stable fixed point until the barrier protecting it vanishes,
and just above that point the time taken for flux to enter diverges (critical
slowing down).  A measurement is therefore only meaningful together with the
hold time it used, which :func:`expulsion_field` records.

See ``docs/figures/sis_hole_expulsion.py`` for a worked S/I/S example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .vortex_counting import count_vortices_polygon

if TYPE_CHECKING:  # pragma: no cover
    from ..core.device import Device
    from ..core.solution import Solution

__all__ = [
    "ExpulsionResult",
    "expulsion_field",
    "fluxoid_history",
    "rectangular_contour",
]


def rectangular_contour(
    hole_bounds: tuple[float, float, float, float],
    params,
    margin: float = 1.5,
) -> NDArray[np.float64]:
    """A rectangular contour in the superconductor enclosing a rectangular hole.

    Parameters
    ----------
    hole_bounds : (x_min, x_max, y_min, y_max)
        Hole extent in physical (ξ) coordinates — the same coordinates
        :meth:`tdgl3d.core.device.Device.add_hole` takes.
    params : SimulationParameters
        Used to convert to grid-node coordinates and to clip to the interior.
    margin : float, default 1.5
        Distance in ξ between the hole edge and the contour.  It should exceed a
        couple of coherence lengths so the contour sits in well-formed
        superconductor, and stay clear of the outer boundary.

    Returns
    -------
    ndarray, shape (4, 2)
        Contour vertices in full-grid node coordinates, ready for
        :func:`tdgl3d.analysis.count_vortices_polygon`.

    Notes
    -----
    The fluxoid is a topological invariant of the enclosed region, so its value
    does not depend on ``margin`` as long as the contour separates the hole from
    the outer boundary and encloses no other vortices.  Comparing two margins is
    a cheap way to confirm that.
    """
    x_min, x_max, y_min, y_max = hole_bounds
    i_lo = int(round((x_min - margin) / params.hx))
    i_hi = int(round((x_max + margin) / params.hx))
    j_lo = int(round((y_min - margin) / params.hy))
    j_hi = int(round((y_max + margin) / params.hy))

    i_lo, i_hi = max(i_lo, 1), min(i_hi, params.Nx - 1)
    j_lo, j_hi = max(j_lo, 1), min(j_hi, params.Ny - 1)
    if i_hi - i_lo < 2 or j_hi - j_lo < 2:
        raise ValueError(
            "contour collapsed: the hole plus margin does not leave room inside "
            "the interior grid"
        )
    return np.array(
        [[i_lo, j_lo], [i_hi, j_lo], [i_hi, j_hi], [i_lo, j_hi]], dtype=float
    )


def fluxoid_history(
    solution: Solution,
    device: Device,
    contour: NDArray[np.float64],
    slice_z: int = 0,
) -> NDArray[np.float64]:
    """Fluxoid enclosed by *contour* at every saved step, in units of Φ₀.

    Each entry is an integer to floating-point round-off (see
    :func:`tdgl3d.analysis.count_vortices_polygon`), so the history is a
    staircase whose steps are the individual flux quanta entering the hole.
    """
    return np.array(
        [
            count_vortices_polygon(solution, device, contour, slice_z=slice_z, step=step)
            for step in range(solution.states.shape[1])
        ],
        dtype=float,
    )


def first_entry_time(
    times: Sequence[float],
    fluxoid: Sequence[float],
    threshold: float = 0.5,
) -> Optional[float]:
    """Time at which the fluxoid first leaves zero, or ``None`` if it never does."""
    for time, value in zip(times, fluxoid):
        if abs(value) >= threshold:
            return float(time)
    return None


@dataclass
class ExpulsionResult:
    """Outcome of an applied-field scan for the flux-expulsion threshold.

    Attributes
    ----------
    fields : list of float
        Applied fields scanned, in ascending order.
    final_fluxoid : list of float
        Fluxoid enclosed by the contour at the end of each run.
    entry_times : list of float or None
        Time at which flux first entered at each field; ``None`` where it never
        did.
    hold_time : float
        Integration time each run was given.  The threshold is only defined
        relative to this: just above it, the entry time diverges.
    last_expelled, first_entered : float or None
        The bracket — the largest scanned field that held the fluxoid at zero,
        and the smallest that did not.
    """

    fields: list[float] = field(default_factory=list)
    final_fluxoid: list[float] = field(default_factory=list)
    entry_times: list[Optional[float]] = field(default_factory=list)
    hold_time: float = 0.0
    last_expelled: Optional[float] = None
    first_entered: Optional[float] = None

    @property
    def threshold(self) -> Optional[float]:
        """Midpoint of the bracket, or ``None`` if the scan did not bracket it."""
        if self.last_expelled is None or self.first_entered is None:
            return None
        return 0.5 * (self.last_expelled + self.first_entered)

    @property
    def uncertainty(self) -> Optional[float]:
        """Half-width of the bracket — the scan's resolution, not an error bar."""
        if self.last_expelled is None or self.first_entered is None:
            return None
        return 0.5 * (self.first_entered - self.last_expelled)

    def summary(self) -> str:
        if self.threshold is None:
            return (
                "expulsion field not bracketed: "
                f"{'all fields expelled' if self.first_entered is None else 'no field expelled'}"
            )
        return (
            f"B_exp = {self.threshold:.4f} ± {self.uncertainty:.4f} "
            f"(hold time {self.hold_time:g})"
        )


def expulsion_field(
    fields: Sequence[float],
    final_fluxoid: Sequence[float],
    entry_times: Sequence[Optional[float]],
    hold_time: float,
    threshold: float = 0.5,
) -> ExpulsionResult:
    """Bracket the flux-expulsion threshold from a completed field scan.

    Parameters
    ----------
    fields : sequence of float
        Applied fields, ascending.
    final_fluxoid : sequence of float
        Fluxoid at the end of each run.
    entry_times : sequence of float or None
        First-entry time for each run.
    hold_time : float
        Integration time used for every run.
    threshold : float, default 0.5
        Fluxoid magnitude above which the ring counts as having admitted flux.

    Returns
    -------
    ExpulsionResult

    Notes
    -----
    The scan is assumed monotone: a field that admits flux implies every larger
    field does.  The bracket is taken from the last expelled and first admitted
    entries, so a non-monotone scan (which would indicate the hold time is too
    short near the threshold) yields a bracket that is too wide rather than a
    confidently wrong number.
    """
    order = np.argsort(np.asarray(fields, dtype=float))
    sorted_fields = [float(fields[i]) for i in order]
    sorted_fluxoid = [float(final_fluxoid[i]) for i in order]
    sorted_entries = [entry_times[i] for i in order]

    expelled = [b for b, n in zip(sorted_fields, sorted_fluxoid) if abs(n) < threshold]
    entered = [b for b, n in zip(sorted_fields, sorted_fluxoid) if abs(n) >= threshold]

    return ExpulsionResult(
        fields=sorted_fields,
        final_fluxoid=sorted_fluxoid,
        entry_times=list(sorted_entries),
        hold_time=float(hold_time),
        last_expelled=max(expelled) if expelled else None,
        first_entered=min(entered) if entered else None,
    )
