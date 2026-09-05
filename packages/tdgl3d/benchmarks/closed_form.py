r"""Closed-form solutions the three codes are measured against.

Two of the benchmarks in this package have exact answers, and neither
has a fitted parameter.

**A thin disk in a perpendicular field.**  With :math:`|\psi| = 1` the
Ginzburg-Landau equations collapse to the thin-film London equation

.. math::  \mathbf{K} = -\frac{1}{\mu_0\Lambda}\,\mathbf{A},
           \qquad \Lambda = \lambda^2/d,

for the sheet current :math:`\mathbf{K}` and the *total* in-plane vector
potential.  Both ends of the :math:`\Lambda/R` axis are solvable:

* :math:`\Lambda \gg R` — the induced part of **A** is negligible, so
  **K** follows the applied vector potential alone.  A disk is the one
  shape where this is exact rather than approximate: the symmetric gauge
  :math:`\mathbf{A} = \tfrac12 B r\,\hat\varphi` already satisfies
  :math:`\nabla\cdot\mathbf{A} = 0` *and* :math:`\mathbf{A}\cdot\hat n = 0`
  on a circular edge, so no gauge correction is needed to keep current
  inside the film.  On a square it is not, which is why the benchmark
  uses a disk.  See :func:`london_disk_sheet_current` and
  :func:`london_disk_moment`.
* :math:`\Lambda \ll R` — complete screening, the classical
  perfectly-diamagnetic thin disk.  See :func:`ideal_disk_sheet_current`
  and :func:`ideal_disk_moment`.

Between them there is no closed form, so that range measures the codes
against *each other* rather than against an exact answer.

**A pair-breaking wall.**  At zero field the gauge field drops out and
:math:`\psi'' = -\psi + \psi^3`, whose first integral
:math:`\psi' = (1 - \psi^2)/\sqrt2` (:func:`gl_wall_first_integral`) is
the offset-free statement that the healing length is
:math:`\sqrt2\,\xi` and not :math:`\xi`.  Checking the first integral
rather than :math:`\tanh((x-x_0)/\sqrt2)` avoids having to locate the
interface, which the two codes place differently.

Units
-----
Every function here is unit-system agnostic: pass ``H_a`` (the applied
field over the vacuum permeability), ``R`` and ``Lambda`` in any
consistent system and the moment comes back in that system.  In the
dimensionless Ginzburg-Landau units used by ``tdgl3d``, :math:`\mu_0`
is :math:`1/\kappa^2`, so ``H_a`` is :math:`\kappa^2 B_z` and
:math:`\Lambda = \kappa^2 / \int|\psi|^2\,\mathrm{d}z`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SQRT2",
    "gl_wall_first_integral",
    "ideal_disk_moment",
    "ideal_disk_sheet_current",
    "ideal_over_london",
    "london_disk_moment",
    "london_disk_sheet_current",
]

SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Thin disk, Λ ≫ R:  the applied vector potential alone
# ---------------------------------------------------------------------------

def london_disk_sheet_current(
    r: NDArray[np.float64] | float,
    H_a: float,
    Lambda: float,
) -> NDArray[np.float64]:
    r"""Azimuthal sheet current of a weakly screening disk.

    ``K_φ(r) = -μ₀ H_a r / (2Λ)``, from :math:`\mathbf{K} = -\mathbf{A}/\mu_0\Lambda`
    with the applied vector potential in the symmetric gauge and the
    induced part dropped.  Exact in the limit ``Λ/R → ∞``; the leading
    correction is ``O(R/Λ)``.

    Written with ``μ₀ H_a`` rather than ``B`` so the same expression
    serves the dimensionless units, where ``μ₀ = 1/κ²``.
    """
    r = np.asarray(r, dtype=float)
    return -0.5 * H_a * r / Lambda


def london_disk_moment(H_a: float, R: float, Lambda: float) -> float:
    r"""Magnetic moment of a weakly screening disk, ``-π H_a R⁴/(8Λ)``.

    From :math:`m = \tfrac12\int (\mathbf{r}\times\mathbf{K})_z\,dA` with
    :func:`london_disk_sheet_current`.  The ``μ₀`` in **K** cancels
    against the one in the definition of the moment, so the answer is in
    current × area for any consistent unit system.
    """
    return -np.pi * H_a * R**4 / (8.0 * Lambda)


# ---------------------------------------------------------------------------
# Thin disk, Λ ≪ R:  complete screening
# ---------------------------------------------------------------------------

def ideal_disk_sheet_current(
    r: NDArray[np.float64] | float,
    H_a: float,
    R: float,
) -> NDArray[np.float64]:
    r"""Azimuthal sheet current of a perfectly screening disk.

    ``K_φ(r) = -(4/π) H_a r / √(R² - r²)``, the ``Λ → 0`` Meissner state.
    It diverges at the rim, so compare it on ``r/R ≲ 0.9`` and use
    :func:`ideal_disk_moment` — an integral, and finite — for the
    headline number.
    """
    r = np.asarray(r, dtype=float)
    return -(4.0 / np.pi) * H_a * r / np.sqrt(np.maximum(R * R - r * r, 0.0))


def ideal_disk_moment(H_a: float, R: float) -> float:
    """Magnetic moment of a perfectly screening disk, ``-(8/3) H_a R³``.

    The integral of :func:`ideal_disk_sheet_current`, and the standard
    result for a perfectly diamagnetic thin disk.
    """
    return -(8.0 / 3.0) * H_a * R**3


def ideal_over_london(lambda_over_r: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Where the two asymptotes cross, as ``m_ideal / m_London = (64/3π)(Λ/R)``.

    Normalising every code's moment by its own ``m_London`` collapses all
    of them onto one dimensionless curve of ``μ`` against ``Λ/R``, which
    must approach ``1`` at large ``Λ/R`` and this straight line at small
    ``Λ/R``.  No unit conversion between the codes is involved.
    """
    return (64.0 / (3.0 * np.pi)) * np.asarray(lambda_over_r, dtype=float)


# ---------------------------------------------------------------------------
# Pair-breaking wall
# ---------------------------------------------------------------------------

def gl_wall_first_integral(psi: NDArray[np.float64] | float) -> NDArray[np.float64]:
    r"""``ψ' = (1 - ψ²)/√2`` — the first integral of ``ψ'' = -ψ + ψ³``.

    The boundary condition that fixes the constant of integration is
    ``ψ → 1``, ``ψ' → 0`` in the bulk, so this holds pointwise in the
    superconductor with no interface position, matching constant or fit
    anywhere in it.  A code that healed over ``ξ`` instead of ``√2 ξ``
    would come out high by ``√2``.
    """
    psi = np.asarray(psi, dtype=float)
    return (1.0 - psi**2) / SQRT2
