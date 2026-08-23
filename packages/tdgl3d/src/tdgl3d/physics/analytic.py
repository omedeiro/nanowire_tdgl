"""Closed-form Ginzburg-Landau solutions for verifying the solver.

Two limits of the coupled TDGL equations have exact solutions, and between them
they exercise both equations independently:

* **The London limit** — ``|ψ| = 1`` everywhere, so the ψ-equation drops out and
  the gauge field obeys ``∇²B = B/λ²``.  On a square with the field pinned on
  the boundary this has an exact Fourier solution
  (:func:`london_square_2d`).  It tests the ``κ²∇×∇×`` operator and the
  applied-field boundary condition.
* **The pair-breaking wall** — zero field, so the gauge field drops out and the
  ψ-equation reduces to ``ψ'' = -ψ + ψ³``.  Against an insulator this has the
  exact solution ``tanh((x - x₀)/√2)`` with the offset fixed by matching to the
  insulator's relaxation (:func:`gl_wall_profile`).  It tests the covariant
  Laplacian, the nonlinear term, and the material mask.

Both comparisons hinge on where the discrete quantities actually sit, which is
easy to get wrong by half a cell and shows up as a spurious first-order error.
:func:`interior_node_positions` and :func:`plaquette_positions` record the
answer; see their docstrings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..operators.sparse_operators import INSULATOR_RELAXATION_TIME

__all__ = [
    "gl_wall_interface_value",
    "gl_wall_profile",
    "interior_node_positions",
    "london_slab_1d",
    "london_square_2d",
    "plaquette_positions",
]

SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Where the discrete quantities live
# ---------------------------------------------------------------------------


def interior_node_positions(params: SimulationParameters, axis: str = "x") -> NDArray[np.float64]:
    """Physical coordinates of the interior nodes along *axis*.

    Node-centred quantities — ψ, and anything derived from it — live at the
    interior nodes ``i = 1 … N-1``, so entry ``i-1`` of an interior array sits at
    ``x = i·h``.  The domain edges are at ``0`` and ``N·h``.
    """
    spacing = {"x": params.hx, "y": params.hy, "z": params.hz}[axis]
    count = {"x": params.Nx, "y": params.Ny, "z": params.Nz}[axis]
    return np.arange(1, count) * spacing


def plaquette_positions(params: SimulationParameters, axis: str = "x") -> NDArray[np.float64]:
    """Physical coordinates of the plaquettes held in an interior array.

    ``B`` is a plaquette quantity: the plaquette anchored at node ``i`` spans
    ``[i·h, (i+1)·h]`` and is centred at ``(i + ½)h``.  An interior array holds
    the anchors ``i = 1 … N-1``.

    The subtlety is the boundary condition, which pins the *whole ring* of
    plaquettes — including the ghost anchor ``0``, centred at ``½h``, which the
    interior array does not carry.  The pinned-to-pinned span is therefore
    ``½h … (N-½)h``, of width ``(N-1)h``, and entry ``i-1`` of the array sits at
    ``i·h`` from the low end of it.  These are the coordinates returned, so they
    can be handed straight to a continuum solution posed on ``[0, (N-1)h]``.

    Using the plaquette centres measured from ``0`` instead — the obvious
    choice — displaces the profile by one cell and turns a second-order
    comparison into a first-order one.
    """
    spacing = {"x": params.hx, "y": params.hy, "z": params.hz}[axis]
    count = {"x": params.Nx, "y": params.Ny, "z": params.Nz}[axis]
    return np.arange(1, count) * spacing


def london_domain_width(params: SimulationParameters, axis: str = "x") -> float:
    """Width of the pinned-to-pinned plaquette span, ``(N-1)h``."""
    spacing = {"x": params.hx, "y": params.hy, "z": params.hz}[axis]
    count = {"x": params.Nx, "y": params.Ny, "z": params.Nz}[axis]
    return (count - 1) * spacing


# ---------------------------------------------------------------------------
# London limit: |ψ| = 1, ∇²B = B/λ²
# ---------------------------------------------------------------------------


def _cosh_ratio(
    k: float,
    a: NDArray[np.float64] | float,
    b: float,
) -> NDArray[np.float64]:
    """``cosh(k·a) / cosh(k·b)`` for ``|a| <= b``, without overflowing.

    Evaluated directly, both cosh terms overflow once ``k·b`` exceeds about 709,
    which for this series happens around the 400th term and turns the whole sum
    into ``nan``.  Rewriting as

        cosh(ka)/cosh(kb) = (e^{k(a-b)} + e^{-k(a+b)}) / (1 + e^{-2kb})

    leaves every exponent non-positive when ``|a| <= b``, so the expression is
    bounded by construction.
    """
    a = np.asarray(a, dtype=float)
    return (np.exp(k * (a - b)) + np.exp(-k * (a + b))) / (1.0 + np.exp(-2.0 * k * b))


def london_slab_1d(
    x: NDArray[np.float64] | float,
    width: float,
    lam: float,
    b0: float = 1.0,
) -> NDArray[np.float64]:
    """Field in a slab of thickness *width* with ``B = b0`` on both faces.

    ``B(x) = b0 cosh((x - W/2)/λ) / cosh(W/(2λ))``, the solution of
    ``B'' = B/λ²``.  Valid when the sample is effectively infinite in the
    transverse directions; use :func:`london_square_2d` for a finite square,
    where transverse screening makes the decay measurably faster.
    """
    x = np.asarray(x, dtype=float)
    return b0 * np.cosh((x - 0.5 * width) / lam) / np.cosh(0.5 * width / lam)


def london_square_2d(
    x: NDArray[np.float64] | float,
    y: NDArray[np.float64] | float,
    width: float,
    lam: float,
    b0: float = 1.0,
    n_terms: int = 201,
) -> NDArray[np.float64]:
    """Exact solution of ``∇²B = B/λ²`` on ``[0, W]²`` with ``B = b0`` on ∂Ω.

    Superposition of the two one-sided problems — ``b0`` on the pair of x-faces
    and zero on the y-faces, plus the transpose.  Each is a Fourier sine series
    in the transverse coordinate with cosh decay in the longitudinal one:

    .. math::
        B_1(x, y) = \\sum_{n\\ \\mathrm{odd}} \\frac{4 b_0}{n\\pi}
            \\sin\\frac{n\\pi y}{W}\\;
            \\frac{\\cosh k_n (x - W/2)}{\\cosh k_n W/2},
        \\qquad k_n^2 = \\frac{1}{\\lambda^2} + \\left(\\frac{n\\pi}{W}\\right)^2 .

    Parameters
    ----------
    x, y : array_like
        Coordinates in ``[0, W]``, broadcast against each other.
    width : float
        Side of the square.
    lam : float
        Penetration depth.  In solver units this is ``κ``.
    b0 : float, default 1.0
        Field on the boundary.
    n_terms : int, default 201
        Highest term kept.  Only odd terms contribute.

    Notes
    -----
    The series converges to ``b0`` on the open edges but to ``b0/2`` at the
    corners, where the two one-sided problems each contribute a jump; stay a few
    cells away from a corner when comparing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = np.zeros(np.broadcast(x, y).shape, dtype=float)
    half = 0.5 * width
    for n in range(1, n_terms + 1, 2):
        k = np.sqrt(1.0 / lam**2 + (n * np.pi / width) ** 2)
        amplitude = 4.0 * b0 / (n * np.pi)
        total = total + amplitude * (
            np.sin(n * np.pi * y / width) * _cosh_ratio(k, x - half, half)
            + np.sin(n * np.pi * x / width) * _cosh_ratio(k, y - half, half)
        )
    return total


# ---------------------------------------------------------------------------
# Pair-breaking wall: zero field, ψ'' = -ψ + ψ³
# ---------------------------------------------------------------------------


def gl_wall_interface_value(tau: float = INSULATOR_RELAXATION_TIME) -> float:
    """``|ψ|`` at a superconductor/insulator interface, from matching.

    In the superconductor ``ψ'' = -ψ + ψ³`` has the first integral
    ``ψ' = (1 - ψ²)/√2``.  In the insulator the solver's relaxation term gives
    ``ψ'' = ψ/τ``, hence ``ψ = u e^{x/√τ}`` and ``ψ' = u/√τ``.  Requiring ψ and
    ψ' to be continuous leaves ``√τ u² + √2 u - √τ = 0``, whose positive root is
    returned.

    For the solver's ``τ = 0.1`` this is ``u ≈ 0.2134``: the interface is a
    strong but not perfect pair breaker, which is why the wall profile is
    ``tanh`` shifted by a definite amount rather than ``tanh`` through the
    origin.
    """
    root_tau = np.sqrt(tau)
    return float((-SQRT2 + np.sqrt(2.0 + 4.0 * tau)) / (2.0 * root_tau))


def gl_wall_profile(
    x: NDArray[np.float64] | float,
    tau: float = INSULATOR_RELAXATION_TIME,
) -> NDArray[np.float64]:
    """``|ψ|(x)`` across a pair-breaking wall, interface at ``x = 0``.

    Superconductor at ``x > 0``, insulator at ``x < 0``:

    .. math::
        |\\psi|(x) = \\begin{cases}
            \\tanh\\!\\big((x - x_0)/\\sqrt2\\big), & x \\ge 0 \\\\
            u\\, e^{x/\\sqrt\\tau}, & x < 0
        \\end{cases}

    with ``u`` from :func:`gl_wall_interface_value` and
    ``x₀ = -√2 · arctanh(u)``.  There are **no free parameters** — the offset is
    fixed by the matching condition, so a fit is not needed and a disagreement
    is a real disagreement.

    The ``√2`` is the physics being checked: the Ginzburg-Landau healing length
    is ``√2 ξ``, not ``ξ``.

    Notes
    -----
    On the lattice the material coefficient jumps *between* nodes, so the
    effective interface lies at the midpoint of the last insulator node and the
    first superconducting one.  Placing the origin on either node instead
    displaces the profile by half a cell and degrades the comparison from second
    order to first.
    """
    u = gl_wall_interface_value(tau)
    x0 = -SQRT2 * np.arctanh(u)
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0.0,
        np.tanh((x - x0) / SQRT2),
        u * np.exp(np.minimum(x, 0.0) / np.sqrt(tau)),
    )
