r"""The order-parameter equation on its own, in the two TDGL codes.

SuperScreen has no ``ψ``, so this benchmark has two participants rather
than three — which is the point of running it: everything the Pearl-disk
benchmark measures is magnetostatics, and a code can get all of that
right with the wrong condensate.

The problem is a pair-breaking wall at zero applied field.  With no field
the gauge field drops out and Ginzburg-Landau reduces to

.. math::  \psi'' = -\psi + \psi^3,

whose first integral, fixed by ``ψ → 1``, ``ψ' → 0`` in the bulk, is

.. math::  \psi' = (1 - \psi^2)/\sqrt2 .

That is what is checked, rather than ``tanh((x - x_0)/\sqrt2)``, because
the two codes suppress ``ψ`` by different mechanisms and therefore put
the interface in different places: ``tdgl3d`` carves a hole, whose
non-superconducting nodes relax ``ψ`` towards zero on a fixed time
constant; pyTDGL is given ``ε = -1`` there, which is the physical
statement ``T > T_c``.  The first integral has no ``x`` in it, so it
compares the two without either having to agree on where zero is — and
the quantity it pins down is the one that matters, that the healing
length is ``√2 ξ``.

Both codes measure ``ψ'`` by central differences on a uniform sample of
the relaxed profile, so the differencing error is common to both.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .closed_form import gl_wall_first_integral

__all__ = ["WallRun", "run_all", "run_pytdgl_wall", "run_tdgl3d_wall"]

#: Where the profile is compared.  Below ``ψ = 0.05`` the answer is the
#: interface model rather than Ginzburg-Landau; above ``0.95`` both sides
#: of the first integral are approaching zero and the ratio stops being
#: informative.
PSI_WINDOW = (0.05, 0.98)


@dataclass
class WallRun:
    tool: str
    spacing: float
    psi: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    dpsi_dx: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    meta: dict = field(default_factory=dict)

    @property
    def exact(self) -> NDArray[np.float64]:
        return gl_wall_first_integral(self.psi)

    @property
    def rms(self) -> float:
        """rms of ``ψ' - (1-ψ²)/√2`` over the healing window."""
        return float(np.sqrt(np.mean((self.dpsi_dx - self.exact) ** 2)))

    @property
    def healing_length(self) -> float:
        r"""Best-fit ``ℓ`` in ``ψ' = (1 - ψ²)/ℓ``, which theory fixes at ``√2``.

        A single number for the whole profile, from a least-squares fit
        through the origin, so a code that healed over ``ξ`` rather than
        ``√2 ξ`` shows up as ``1.0`` here instead of ``1.414``.
        """
        rhs = 1.0 - self.psi**2
        return float(rhs @ rhs / (rhs @ self.dpsi_dx))

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "spacing": self.spacing,
            "psi": self.psi.tolist(),
            "dpsi_dx": self.dpsi_dx.tolist(),
            "exact": self.exact.tolist(),
            "rms": self.rms,
            "healing_length": self.healing_length,
            **self.meta,
        }


def _window(psi: NDArray, superconducting: NDArray, *arrays):
    """Keep the healing window, and only on the superconducting side.

    The window alone is not enough.  ``ψ`` passes through the same values
    on the other side of the interface, where it obeys the interface
    model rather than Ginzburg-Landau — in ``tdgl3d`` a decay over
    ``√(0.1) ξ``, in pyTDGL one over ``ξ/√|ε|``.  Those points have ``ψ``
    in range and a slope several times too steep, and including them
    pulls the fitted healing length well above ``√2`` while making the
    residual grow under refinement rather than shrink.
    """
    keep = (psi > PSI_WINDOW[0]) & (psi < PSI_WINDOW[1]) & superconducting
    return (psi[keep],) + tuple(a[keep] for a in arrays)


def _central_difference(x: NDArray, y: NDArray):
    """``(x, y, dy/dx)`` on the interior of a uniformly spaced sample."""
    dx = x[1] - x[0]
    return x[1:-1], y[1:-1], (y[2:] - y[:-2]) / (2.0 * dx)


# ---------------------------------------------------------------------------

def run_tdgl3d_wall(
    spacing: float = 0.25,
    *,
    length: float = 24.0,
    wall: float = 8.0,
    kappa: float = 2.0,
    t_stop: float = 40.0,
) -> WallRun:
    """Relax a half-plane hole against bulk metal and read ``|ψ|`` out of it.

    The geometry is the one ``docs/figures/analytic_cross_sections.py``
    uses, so the number here is comparable with the figure in the README.
    """
    from tdgl3d import AppliedField, Device, SimulationParameters
    from tdgl3d.physics.rhs import BoundaryVectors
    from tdgl3d.solvers.integrators import forward_euler

    n_cells = int(round(length / spacing))
    params = SimulationParameters(
        Nx=n_cells, Ny=6, Nz=1, hx=spacing, hy=spacing, kappa=kappa
    )
    device = Device(params, applied_field=AppliedField(Bz=0.0))
    device.add_hole(
        [(-1.0, -1.0), (wall, -1.0), (wall, length + 1.0), (-1.0, length + 1.0)]
    )
    zeros = np.zeros(params.dim_x, dtype=np.float64)
    boundary = BoundaryVectors(zeros, zeros.copy(), zeros.copy())
    dt = 0.9 * spacing**2 / (4.0 * kappa**2)
    start = time.perf_counter()
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data, params, device.idx,
        lambda t, X: boundary, 0.0, t_stop, dt,
        save_every=10**9, progress=False, material=device.material,
    )
    elapsed = time.perf_counter() - start

    nx_int, ny_int = params.Nx - 1, params.Ny - 1
    profile = np.abs(states[:, -1][:params.n_interior]).reshape(nx_int, ny_int)
    mask = device.material.interior_sc_mask.reshape(nx_int, ny_int)
    row = profile[:, ny_int // 2]
    x = np.arange(1, params.Nx) * spacing

    _, psi, dpsi = _central_difference(x, row)
    psi, dpsi = _window(psi, mask[1:-1, ny_int // 2] > 0, dpsi)
    return WallRun(
        tool="tdgl3d", spacing=spacing, psi=psi, dpsi_dx=dpsi,
        meta={"seconds": elapsed, "cells": n_cells, "kappa": kappa,
              "t_stop": t_stop, "samples": int(psi.size)},
    )


def run_pytdgl_wall(
    spacing: float = 0.25,
    *,
    xi: float = 1.0,
    length: float = 24.0,
    width: float = 6.0,
    wall: float = 8.0,
    solve_time: float = 60.0,
) -> WallRun:
    """Same wall in pyTDGL, made with ``ε = -1`` on one side.

    Lengths are given in units of ξ and handed to pyTDGL as µm with
    ``coherence_length = 1``, so the profile comes back in the same units
    the closed form is written in.  ``max_edge_length`` is set to
    *spacing* so the two codes are refined together.
    """
    import tdgl
    from tdgl.geometry import box

    layer = tdgl.Layer(coherence_length=xi, london_lambda=2.0 * xi,
                       thickness=0.1 * xi, gamma=1.0)
    film = tdgl.Polygon("film", points=box(length, width))
    device = tdgl.Device("wall", layer=layer, film=film, length_units="um")
    device.make_mesh(max_edge_length=spacing, smooth=20)

    # box() is centred on the origin, so the wall sits at ``wall - length/2``.
    x_wall = wall - 0.5 * length

    def epsilon(r):
        # pyTDGL calls this once per site with a single (x, y) pair.
        return -1.0 if r[0] < x_wall else 1.0

    workspace = tempfile.mkdtemp(prefix="tdgl3d-benchmark-")
    output = os.path.join(workspace, "wall.h5")
    options = tdgl.SolverOptions(
        solve_time=solve_time, output_file=output, field_units="mT",
        current_units="uA", save_every=10**6,
    )
    start = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()):
        solution = tdgl.solve(device, options, disorder_epsilon=epsilon)
    elapsed = time.perf_counter() - start

    x = np.arange(x_wall - 2.0, x_wall + 10.0 + 1e-9, spacing)
    positions = np.column_stack([x, np.zeros_like(x)])
    row = np.abs(solution.interp_order_parameter(positions))
    sites = int(device.points.shape[0])
    del solution
    shutil.rmtree(workspace, ignore_errors=True)

    xs, psi, dpsi = _central_difference(x, row)
    psi, dpsi = _window(psi, xs > x_wall, dpsi)
    return WallRun(
        tool="pytdgl", spacing=spacing, psi=psi, dpsi_dx=dpsi,
        meta={"seconds": elapsed, "sites": sites, "solve_time": solve_time,
              "samples": int(psi.size)},
    )


def run_all(spacings=(0.5, 0.25, 0.125)) -> list[dict]:
    """Both codes at three grid spacings, so the residual can be shown to shrink.

    Coarser than ``h = 0.5 ξ`` there are too few points inside the healing
    window to fit anything: ``ψ`` crosses from 0.05 to 0.98 in about
    ``3.2 ξ``, so the sample count is roughly ``3/h``.
    """
    runs = []
    for spacing in spacings:
        for runner in (run_tdgl3d_wall, run_pytdgl_wall):
            result = runner(spacing)
            print(
                f"{result.tool:8s} h={spacing:5.3g}  rms={result.rms:.3e}  "
                f"healing length={result.healing_length:.4f}  (exact 1.41421)"
            )
            runs.append(result.as_dict())
    return runs
