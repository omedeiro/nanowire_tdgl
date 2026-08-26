"""Tabulate the solver's error against every result known independently of it.

``docs/physics_test_report.md`` lists every structured check the physics suites
record, grouped by the suite that produced it.  This script answers a different
question: **for each result that is known without running the solver, how far off
is the solver, and how much of the error budget does that use?**

Which check measures the deviation from which known result is curation, not
something derivable from a log file, so it is written down here as
:data:`ANALYTIC`, :data:`ORDERS`, :data:`EXACT`, :data:`BOUNDS` and
:data:`MODEL`.  Everything
numeric — measured value, requirement, pass/fail — is read from
``logs/test_*.json``, which the ``phys_log`` fixture writes.  A reference whose
test or check label has moved is rendered as ``not in logs`` and reported on
stderr with a non-zero exit, so the table cannot quietly go stale.

The sections are separated because the size of an error means something
different in each:

``analytic``
    A continuum solution of the Ginzburg-Landau equations.  The deviation is
    discretisation error and must fall with ``h`` or ``dt``; the tolerance is a
    bound stated up front.
``orders``
    The rate at which that deviation falls.  A wrong operator with the right
    magnitude still gets the order wrong, so these are the stronger rows.
``exact``
    An identity the *discrete* equations satisfy exactly — a lattice identity, a
    symmetry, or a second code path solving the same discrete problem.  The
    deviation is round-off; a residual that scales with ``h`` instead means two
    stencils are paired wrongly and is never a tolerance to widen.
``bounds``
    Real physics with no closed form to subtract — an inequality, a sign, a
    direction.  No error, but a violation is still a wrong solver.
``model``
    The closed-form reference checked against itself, so a solver that agrees
    with a wrong model cannot look verified.

Usage::

    python3 -m pytest packages/tdgl3d/tests/test_verification_*.py \\
        packages/tdgl3d/tests/test_physics_validation.py -q
    python3 docs/generate_error_table.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Reference:
    """One known result, and the check that measures the solver's error against it."""

    result: str
    """The known result, as a formula in solver units."""

    source: str
    """Where the result comes from — the limit, the identity or the derivation."""

    test: str
    """``test_name`` as recorded in the log, including any parametrisation."""

    check: str
    """``label`` of the check inside that test."""

    condition: str = ""
    """The resolution, grid or field the row was measured at."""


# ---------------------------------------------------------------------------
# Continuum solutions: the deviation is discretisation error
# ---------------------------------------------------------------------------

ANALYTIC: list[Reference] = [
    Reference(
        "λ = κ",
        "London limit of ∇²B = B/λ²: B ~ e^{−x/κ}",
        "test_london_penetration_depth_equals_kappa[kappa=1.5]",
        "λ from the screening profile",
        "κ = 1.5, h = 0.5 ξ",
    ),
    Reference(
        "λ = κ",
        "London limit of ∇²B = B/λ²: B ~ e^{−x/κ}",
        "test_london_penetration_depth_equals_kappa[kappa=3.0]",
        "λ from the screening profile",
        "κ = 3.0, h = 0.5 ξ",
    ),
    Reference(
        "λ = κ",
        "the screening length is the physics, not the stencil width",
        "test_penetration_depth_converges_with_grid_refinement",
        "|λ − κ| at h = 0.5",
        "κ = 2.0, 16 ξ square",
    ),
    Reference(
        "B(x, y) = london_square_2d",
        "exact Fourier solution of ∇²B = B/λ² with B = B₀ on ∂Ω",
        "test_bfield_matches_the_exact_london_solution",
        "rms |solver − model| / B₀ at h = 1 ξ",
        "16 ξ square, κ = 2, B = 0.02",
    ),
    Reference(
        "B(x, y) = london_square_2d",
        "exact Fourier solution of ∇²B = B/λ² with B = B₀ on ∂Ω",
        "test_bfield_matches_the_exact_london_solution",
        "rms |solver − model| / B₀ at h = 0.25 ξ",
        "16 ξ square, κ = 2, B = 0.02",
    ),
    Reference(
        "B(x, y) = london_square_2d",
        "worst point rather than the rms, at the finest grid",
        "test_bfield_matches_the_exact_london_solution",
        "max |solver − model| / B₀ at h = 0.25 ξ",
        "16 ξ square, κ = 2, B = 0.02",
    ),
    Reference(
        "E₀ = B, hence H_c2 = 1",
        "lowest Landau level of −(∇ − iA)² in a uniform field",
        "test_lowest_landau_level_of_covariant_laplacian[Bz=0.1]",
        "lowest eigenvalue E₀",
        "24 ξ square, h = 0.5 ξ",
    ),
    Reference(
        "E₀ = B, hence H_c2 = 1",
        "lowest Landau level of −(∇ − iA)² in a uniform field",
        "test_lowest_landau_level_of_covariant_laplacian[Bz=0.2]",
        "lowest eigenvalue E₀",
        "24 ξ square, h = 0.5 ξ",
    ),
    Reference(
        "|ψ| = tanh((x − x₀)/√2)",
        "ψ'' = −ψ + ψ³ at a pair-breaking wall, x₀ fixed by matching",
        "test_order_parameter_matches_the_exact_wall_solution",
        "rms error at h = 1 ξ",
        "24 ξ strip, B = 0",
    ),
    Reference(
        "|ψ| = tanh((x − x₀)/√2)",
        "ψ'' = −ψ + ψ³ at a pair-breaking wall, x₀ fixed by matching",
        "test_order_parameter_matches_the_exact_wall_solution",
        "rms error at h = 0.25 ξ",
        "24 ξ strip, B = 0",
    ),
    Reference(
        "ψ′ = (1 − ψ²)/√2",
        "first integral of ψ'' = −ψ + ψ³ — no interface position, no fit",
        "test_order_parameter_matches_the_exact_wall_solution",
        "rms |ψ′ − (1 − ψ²)/√2| at h = 1 ξ",
        "24 ξ strip, B = 0",
    ),
    Reference(
        "ψ′ = (1 − ψ²)/√2",
        "first integral of ψ'' = −ψ + ψ³ — no interface position, no fit",
        "test_order_parameter_matches_the_exact_wall_solution",
        "rms |ψ′ − (1 − ψ²)/√2| at h = 0.25 ξ",
        "24 ξ strip, B = 0",
    ),
    Reference(
        "u = 0.213422 at the interface",
        "positive root of √τ u² + √2 u − √τ = 0 at the solver's τ = 0.1",
        "test_order_parameter_matches_the_exact_wall_solution",
        "|ψ| at the interface from matching",
        "matching condition, not a run",
    ),
    Reference(
        "|ψ| = 1 in the ground state",
        "the minimum of −|ψ|² + ½|ψ|⁴ at B = 0",
        "test_zero_field_ground_state_is_the_uniform_condensate",
        "min |ψ|",
        "10×8, h = 0.5 ξ, from 0.3 noise",
    ),
    Reference(
        "τ_insulator = 0.1",
        "the relaxation time the insulator term is written with",
        "test_insulator_order_parameter_decays_with_the_stated_time_constant",
        "fitted τ",
        "S/I/S stack, 2-cell oxide",
    ),
    Reference(
        "∇²ψ = −|k|²ψ for ψ = e^{ik·x}",
        "manufactured solution: the operator's own truncation error",
        "test_covariant_laplacian_is_second_order_accurate",
        "error at h = 0.1",
        "k = (0.7, 0.4)",
    ),
    Reference(
        "|ψ|² = 1 after a run at 0.9 dt_CFL",
        "dt < h²/(4κ²(d − 1)), the limit set by the stiff κ²∇×∇× term",
        "test_forward_euler_is_stable_below_the_cfl_limit[3d]",
        "max|ψ|² at dt = 0.9 dt_CFL",
        "6×6×6, h = 0.5 ξ, κ = 2",
    ),
]


# ---------------------------------------------------------------------------
# How fast the deviation falls
# ---------------------------------------------------------------------------

ORDERS: list[Reference] = [
    Reference(
        "the covariant Laplacian is O(h²)",
        "against ∇²e^{ik·x} = −|k|²e^{ik·x}",
        "test_covariant_laplacian_is_second_order_accurate",
        "observed order of accuracy",
        "h = 0.4, 0.2, 0.1 ξ",
    ),
    Reference(
        "forward Euler is O(dt)",
        "Richardson differencing of the solver's own trajectory",
        "test_forward_euler_is_first_order_in_dt",
        "observed order in dt",
        "dt = dt_CFL/2 … /32",
    ),
    Reference(
        "B → London solution at O(h²)",
        "the curl-curl operator's discretisation error, edges excluded",
        "test_bfield_matches_the_exact_london_solution",
        "observed order in h (bulk, >1 ξ from an edge)",
        "h = 1, 0.5, 0.25 ξ",
    ),
    Reference(
        "B → London solution over the whole profile",
        "mixes the second-order bulk with the series' Gibbs floor",
        "test_bfield_matches_the_exact_london_solution",
        "observed order in h (whole profile)",
        "h = 1, 0.5, 0.25 ξ",
    ),
    Reference(
        "|ψ| → tanh profile",
        "bounded below 2 by the coefficient jump between nodes",
        "test_order_parameter_matches_the_exact_wall_solution",
        "observed order in h",
        "h = 1, 0.5, 0.25 ξ",
    ),
    Reference(
        "ψ′ → (1 − ψ²)/√2",
        "the √2 healing length, with no position or fit in the comparison",
        "test_order_parameter_matches_the_exact_wall_solution",
        "observed order in h (first integral)",
        "h = 1, 0.5, 0.25 ξ",
    ),
]


# ---------------------------------------------------------------------------
# Discrete identities: the deviation is round-off
# ---------------------------------------------------------------------------

EXACT: list[Reference] = [
    Reference(
        "∇·B = 0",
        "forward divergence of the forward plaquette curl",
        "test_divergence_of_discrete_curl_is_exactly_zero[6x7x8]",
        "max|∇·B| / max|B|",
        "6×7×8",
    ),
    Reference(
        "∇·(∇×∇×A) = 0",
        "the link equation cannot source charge",
        "test_curl_curl_operator_is_divergence_free[7x6x6]",
        "max|∇·(∇×∇×A)| / scale",
        "7×6×6",
    ),
    Reference(
        "F = −½ per unit volume",
        "−|ψ|² + ½|ψ|⁴ evaluated at |ψ| = 1",
        "test_uniform_state_is_an_exact_fixed_point",
        "condensation energy per unit volume",
        "7×6×5 uniform state",
    ),
    Reference(
        "the uniform state is a fixed point",
        "the Meissner ground state may not drift",
        "test_uniform_state_is_an_exact_fixed_point",
        "max|dX/dt|",
        "7×6×5",
    ),
    Reference(
        "fluxoid number ∈ ℤ",
        "Σ wrap(Δθ − φ) + Σφ = 2πn around any closed loop",
        "test_plaquette_vorticity_is_an_exact_integer",
        "max |vorticity − nearest integer|",
        "20×20, κ = 2, B = 0.5",
    ),
    Reference(
        "lattice Stokes",
        "loop fluxoid = Σ of the plaquettes it encloses",
        "test_fluxoid_equals_enclosed_vorticity_for_any_contour",
        "max |fluxoid − enclosed vorticity|",
        "nested square contours",
    ),
    Reference(
        "lattice Stokes on a non-convex loop",
        "the fluxoid belongs to the region, not to the path",
        "test_fluxoid_equals_enclosed_vorticity_for_any_contour",
        "|staircase fluxoid − enclosed vorticity|",
        "L-shaped contour",
    ),
    Reference(
        "fluxoid ∈ ℤ around a hole",
        "the ring is multiply connected, so flux enters in whole quanta",
        "test_flux_enters_in_whole_quanta",
        "max |fluxoid − nearest integer|",
        "S/I/S ring above threshold",
    ),
    Reference(
        "F(G·X) = G·F(X)",
        "local U(1) covariance of the right-hand side",
        "test_rhs_is_gauge_covariant[6x7x5]",
        "max|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)|",
        "6×7×5",
    ),
    Reference(
        "J_s is gauge invariant",
        "an observable may not move under ψ → ψe^{iχ}, φ → φ + Δχ",
        "test_observables_are_gauge_invariant[6x7x5]",
        "max ΔJ_s",
        "6×7×5",
    ),
    Reference(
        "the vortex count is gauge invariant",
        "a winding number cannot depend on the gauge",
        "test_vortex_count_is_gauge_invariant",
        "max Δ(plaquette vorticity)",
        "relaxed state carrying 8 vortices",
    ),
    Reference(
        "F is a Lyapunov functional",
        "TDGL is the gradient flow of gl_free_energy",
        "test_free_energy_decreases_monotonically_at_zero_field",
        "steps on which F increased",
        "12×11, h = 0.5 ξ, B = 0",
    ),
    Reference(
        "∇·J_s = 0 in steady state",
        "∂(∇·A)/∂t = ∇·J_s, so a stationary gauge field forces it",
        "test_supercurrent_is_divergence_free_in_steady_state",
        "max|∇·J_s| · h / max|J_s|",
        "14×14, κ = 2, B = 0.1",
    ),
    Reference(
        "J_n = 0 on a superconductor/vacuum face",
        "carried by the ψ boundary condition, not by the zeroed links",
        "test_normal_supercurrent_vanishes_on_external_boundaries",
        "max|J_n| on x_lo face",
        "8×7×6",
    ),
    Reference(
        "J_z = 0 on the stack's z-faces",
        "the same condition on a heterostructure",
        "test_trilayer_external_z_boundary_jn",
        "max |J_z| on the top face",
        "S/I/S stack, B = 0.1",
    ),
    Reference(
        "B = B_applied on every boundary plaquette",
        "a Dirichlet condition, applied exactly once per plaquette",
        "test_applied_flux_on_boundary_plaquettes[9x7]",
        "max deviation of boundary ring from B_applied",
        "9×7, split hi/hi corner",
    ),
    Reference(
        "B = B_applied on a pinned plaquette",
        "where the solver is exact, checked with no model in the way",
        "test_bfield_matches_the_exact_london_solution",
        "|B(pinned plaquette) − B_applied| / B₀",
        "worst of h = 1, 0.5, 0.25 ξ",
    ),
    Reference(
        "B → −B leaves |ψ| unchanged",
        "invariance of the GL equations under B → −B with ψ → ψ*",
        "test_field_reversal_flips_b_and_preserves_psi",
        "max|Bz(+B) + Bz(−B)|",
        "9×7, B = 0.4",
    ),
    Reference(
        "C4 symmetry of a square device",
        "a square sample in a uniform Bz is invariant under 90°",
        "test_c4_symmetry_of_a_square_device",
        "max|Bz − R₉₀Bz|",
        "10×10, h = 0.5 ξ",
    ),
    Reference(
        "mirror symmetry on Nx ≠ Ny",
        "a reflection symmetry a transposed index would break",
        "test_mirror_symmetry_of_a_rectangular_device",
        "max|Bz(x) − Bz(−x)|",
        "12×8, h = 0.5 ξ",
    ),
    Reference(
        "the relaxed S/I/S ring is symmetric",
        "a symmetric device relaxed from a noiseless state",
        "test_the_relaxed_ring_is_symmetric",
        "max |ψ| asymmetry under x → −x",
        "S/I/S ring, B = 0.1",
    ),
    Reference(
        "the two B evaluators agree",
        "one curl stencil, one answer, on any grid shape",
        "test_bfield_evaluators_agree[5x7x6]",
        "max|eval_bfield(all interior) − reference|",
        "5×7×6",
    ),
    Reference(
        "the 3-D path reduces to the 2-D one",
        "a z-invariant problem is the same discrete problem in both",
        "test_three_dimensional_solver_reproduces_the_two_dimensional_london_solution",
        "max |B_3D − B_2D| / B₀",
        "16 ξ square, Nz = 4",
    ),
    Reference(
        "trapezoidal ≡ Euler as dt → 0",
        "two integrators of one right-hand side",
        "test_trapezoidal_agrees_with_euler_in_the_small_dt_limit",
        "max|X_trapezoidal − X_euler| / |X|",
        "5×4, B = 0.25",
    ),
    Reference(
        "L_covariant(A = 0) = L_standard",
        "the Peierls factors are the only difference from a 5-point Laplacian",
        "test_covariant_laplacian_reduces_to_the_standard_laplacian",
        "max|L_covariant(A=0) − L_standard|",
        "9×7, hx ≠ hy",
    ),
    Reference(
        "LPHI_x diagonal = −2κ²(1/h_y² + 1/h_z²)",
        "a spatially varying κ enters the operator node by node",
        "test_trilayer_kappa_discontinuity",
        "LPHI_x diagonal in the superconductor",
        "S/I/S, κ = 2",
    ),
    Reference(
        "the stack is symmetric about its mid-plane",
        "equal superconducting layers, both interfaces owned by the oxide",
        "test_stack_is_mirror_symmetric_about_its_midplane[(4, 2)]",
        "κ asymmetry under z → Nz − z",
        "4/2/4 cells",
    ),
    Reference(
        "a hole at [3, 7] is centred on the film",
        "closed-region ray casting rather than the half-open tiling rule",
        "test_centred_hole_is_centred[(10.0, 4.0, 1.0)]",
        "hole centre x",
        "10 ξ film, 4 ξ hole",
    ),
]


# ---------------------------------------------------------------------------
# The reference models, checked against themselves
# ---------------------------------------------------------------------------

MODEL: list[Reference] = [
    Reference(
        "the series solves ∇²B = B/λ²",
        "substituted back into the equation it claims to solve",
        "test_london_series_satisfies_its_own_equation",
        "max |∇²B − B/λ²| at the finer grid",
        "400² check grid",
    ),
    Reference(
        "that residual is the check's own stencil",
        "halving the check grid must quarter an O(h²) residual",
        "test_london_series_satisfies_its_own_equation",
        "residual ratio on halving the check grid",
        "200² vs 400²",
    ),
    Reference(
        "Gibbs floor of the truncated series",
        "the accuracy floor every comparison against it inherits",
        "test_london_series_satisfies_its_own_equation",
        "edge ringing at the default n_terms = 2001",
        "≥ 2 ξ from a corner",
    ),
    Reference(
        "slab = wide limit of the square",
        "london_slab_1d is the transverse-infinite limit of the series",
        "test_london_slab_is_the_wide_limit_of_the_square",
        "max |square − slab| at W = 32 λ",
        "W = 64 ξ, λ = 2 ξ",
    ),
    Reference(
        "transverse screening is real at W = 4 λ",
        "a limit already reached at small width would not be a limit",
        "test_london_slab_is_the_wide_limit_of_the_square",
        "max |square − slab| at W = 4 λ",
        "W = 8 ξ, λ = 2 ξ",
    ),
]


# ---------------------------------------------------------------------------
# Known bounds and directions: real physics, but no closed form to subtract
# ---------------------------------------------------------------------------

BOUNDS: list[Reference] = [
    Reference(
        "no vortices below H_c1",
        "H_c1 ≈ 0.15 at κ = 2; flux is expelled entirely below it",
        "test_no_vortices_in_the_meissner_state",
        "vortex count",
        "16×16, κ = 2, B = 0.03",
    ),
    Reference(
        "every vortex has winding sign(B)",
        "mixed ±1 windings in a uniform field are unphysical",
        "test_vortex_winding_sign_follows_the_applied_field[Bz=-0.5]",
        "common winding",
        "20×20, κ = 2, B = −0.5",
    ),
    Reference(
        "vortices are singly quantised at this field",
        "multiply quantised cores are unstable well below H_c2",
        "test_vortex_winding_sign_follows_the_applied_field[Bz=0.5]",
        "max |winding|",
        "20×20, κ = 2, B = 0.5",
    ),
    Reference(
        "count ≤ B·A/Φ₀",
        "screening keeps the interior field below the applied one",
        "test_vortex_count_increases_with_the_applied_field",
        "count / (B·A/Φ₀) at Bz = 0.5",
        "16×16, κ = 2",
    ),
    Reference(
        "the interior stays screened in the mixed state",
        "vortices admit flux but do not abolish the Meissner currents",
        "test_vortex_count_increases_with_the_applied_field",
        "mean interior Bz / applied at Bz = 0.5",
        "16×16, κ = 2",
    ),
    Reference(
        "B_exp · A_hole ~ Φ₀",
        "the ring gives way when the hole has gathered about one quantum",
        "test_expulsion_threshold_is_bracketed",
        "B_exp · A_hole / Φ₀",
        "S/I/S ring, 4 ξ hole, h = 1 ξ",
    ),
    Reference(
        "a larger hole expels less",
        "more flux gathered per unit field, so the threshold falls",
        "test_a_larger_hole_expels_less",
        "B_exp(6 ξ hole) / B_exp(4 ξ hole)",
        "S/I/S ring, 4 ξ and 6 ξ holes",
    ),
]


@dataclass(frozen=True)
class Section:
    """One table in the report."""

    title: str
    note: str
    value_header: str
    references: list[Reference]

    raw_value: bool = False
    """Tabulate the measured number itself rather than its deviation.

    An observed order of accuracy *is* the quantity of interest; reporting how
    far it sits from the expected order would hide it.
    """


SECTIONS: list[Section] = [
    Section(
        "Closed-form solutions — the error is discretisation error",
        "Each row compares the solver with a continuum result that holds "
        "independently of it, so the deviation is the solver's own "
        "discretisation error and must fall when the grid or the step is "
        "refined. The next section is the statement that it does.",
        "Error",
        ANALYTIC,
    ),
    Section(
        "Observed orders of accuracy",
        "A wrong operator with the right magnitude still gets the order wrong, "
        "so these rows are stronger than any single comparison above. The "
        "column is the observed order itself, and higher is better; where the "
        "requirement is a floor rather than a target there is no budget to "
        "report.",
        "Observed",
        ORDERS,
        raw_value=True,
    ),
    Section(
        "Discrete identities — the error is round-off",
        "These hold for the *discrete* equations, so the deviation is round-off "
        "and nothing else. A residual that scales with `h` instead means two "
        "stencils have been paired wrongly; it is never a tolerance to widen.",
        "Error",
        EXACT,
    ),
    Section(
        "Known bounds and directions — no closed form to subtract",
        "Real physics with no exact solution to compare against: an inequality, "
        "a sign, or a direction the answer has to move in. They cannot give an "
        "error, but a solver that violates one is wrong regardless of how well "
        "it does on the rows above.",
        "Value",
        BOUNDS,
        raw_value=True,
    ),
    Section(
        "The reference models, checked against themselves",
        "A solver that agrees with a wrong reference looks verified. These rows "
        "check the closed-form models before the solver is compared with them, "
        "and record the floor that comparison inherits.",
        "Value",
        MODEL,
    ),
]


# ---------------------------------------------------------------------------
# Known results with no row yet
# ---------------------------------------------------------------------------

UNCOVERED_HEADER = (
    "| Known result | In solver units | Why there is no row yet |\n"
    "|---|---|---|"
)

UNCOVERED: list[tuple[str, str, str]] = [
    (
        "Lower critical field H_c1",
        "(ln κ + ½)/(2κ²) = 0.149 at κ = 2",
        "only bracketed qualitatively — no vortices at B = 0.03, several by "
        "B = 0.35. A number needs a sample many λ wide and a seeded "
        "perturbation, because the Bean-Livingston barrier delays entry well "
        "above H_c1 in a mesoscopic device.",
    ),
    (
        "Thermodynamic critical field H_c",
        "1/(√2 κ) = 0.354 at κ = 2, from κ²H_c² = ½",
        "follows from the same functional the suite already evaluates: the "
        "condensation density −½ is checked and the magnetic density is κ²B², "
        "but the field at which they balance is never measured.",
    ),
    (
        "Surface nucleation field H_c3",
        "1.695 H_c2",
        "the solver uses the gauge-covariant no-current wall condition that "
        "produces surface superconductivity, and the H_c2 row already "
        "diagonalises the same operator — the surface branch is simply not "
        "asked for.",
    ),
    (
        "Depairing current J_d",
        "2/(3√3) = 0.3849 at q = 1/√3, from f² = 1 − q²",
        "there is no way to impose a transport current or a fixed phase "
        "gradient: `SimulationParameters` carries only the (unimplemented) "
        "periodic flags, so the current-carrying uniform branch cannot be set "
        "up.",
    ),
    (
        "Isolated vortex field profile",
        "B(r) ∝ K₀(r/κ) for r ≫ ξ, carrying Φ₀ = 2π",
        "vorticity is checked as an integer winding number, but the magnetic "
        "flux of a single vortex and its radial profile are not. Needs a "
        "sample several λ across so the core is not interacting with the "
        "edges.",
    ),
    (
        "Abrikosov lattice",
        "β_A = 1.1596 for the triangular lattice",
        "the suite counts vortices and checks their sign, never the geometry "
        "they settle into or the energy ratio that selects it.",
    ),
    (
        "Little-Parks periodicity",
        "the ring's ground state is periodic in Φ₀",
        "the expulsion suite sees the fluxoid step through integers at one "
        "field, but never that the branch structure repeats with period Φ₀.",
    ),
    (
        "london_slab_1d against the solver",
        "B = B₀ cosh((x − W/2)/λ) / cosh(W/2λ)",
        "the slab solution is only ever compared with the 2-D series. Running "
        "the solver on a slab geometry would test the same operator through a "
        "different boundary configuration.",
    ),
]


def _load_logs(log_dir: Path) -> dict[str, dict[str, Any]]:
    """Index every logged test by name."""
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(log_dir.glob("test_*.json")):
        with open(path) as handle:
            entry = json.load(handle)
        results[entry["test_name"]] = entry
    return results


def _escape(text: str) -> str:
    """Escape the pipes in ``|ψ|`` and friends so the table survives."""
    return str(text).replace("|", "\\|")


def _fmt(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _measurement(check: dict[str, Any]) -> tuple[float, str, Optional[float]]:
    """``(value, requirement, budget_used)`` for one check.

    ``check_close`` names an expected value, so the value tabulated is the
    deviation from it.  ``check_below`` and ``check_above`` are already written
    as a residual or a floor, so the measured number is tabulated as it stands.
    A floor is not an error budget, so it reports no fraction used.
    """
    measured = float(check["measured"])
    expected = check["expected"]
    tolerance = check["tolerance"]

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        error = abs(measured - float(expected))
        limit = float(tolerance)
        requirement = f"= {_fmt(float(expected))} ± {_fmt(limit)}"
        return error, requirement, (error / limit if limit > 0 else None)
    if isinstance(expected, str) and expected.startswith(">="):
        return measured, f"≥ {_fmt(float(tolerance))}", None
    if isinstance(expected, str) and expected.startswith("<="):
        limit = float(tolerance)
        return measured, f"≤ {_fmt(limit)}", (measured / limit if limit > 0 else None)
    return measured, str(expected), None


def _render(
    section: Section, logs: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str], list[tuple[float, str]]]:
    """Markdown lines for *section*, the unresolved references, and the margins.

    The margins are ``(budget used, row description)`` for every row that has a
    budget, so the summary at the top of the report can name the rows with the
    least room left without a second pass over the logs.
    """
    lines = [
        f"## {section.title}",
        "",
        section.note,
        "",
        f"| Known result | Where it comes from | Measured at | {section.value_header} "
        "| Required | Budget used | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    missing: list[str] = []
    margins: list[tuple[float, str]] = []

    for ref in section.references:
        entry = logs.get(ref.test)
        check = None
        if entry is not None:
            check = next((c for c in entry["checks"] if c["label"] == ref.check), None)
        head = (
            f"| {_escape(ref.result)} | {_escape(ref.source)} | {_escape(ref.condition)} "
        )
        if check is None:
            missing.append(f"{ref.test} · {ref.check}")
            lines.append(head + "| not in logs | — | — | — |")
            continue

        value, requirement, used = _measurement(check)
        if section.raw_value:
            value = float(check["measured"])
        status = "PASS" if check["passed"] else "**FAIL**"
        lines.append(
            head + f"| {_fmt(value)} | {_escape(requirement)} | {_fmt(used)} | {status} |"
        )
        if used is not None and math.isfinite(used):
            where = f" ({ref.condition})" if ref.condition else ""
            margins.append((used, f"{ref.result}{where} — {ref.check}"))

    lines.append("")
    return lines, missing, margins


def generate_table(log_dir: Path) -> tuple[str, list[str]]:
    """Render the markdown table, and report references that no log could satisfy."""
    logs = _load_logs(log_dir)
    if not logs:
        return "# Error Against Known Solutions\n\nNo test log files found.\n", []

    newest = max(entry.get("timestamp", "") for entry in logs.values())
    lines = [
        "# Error Against Known Solutions",
        "",
        f"**Run timestamp:** {newest}",
        "",
        "Generated by `docs/generate_error_table.py` from `logs/test_*.json`; the",
        "mapping from a known result to the check that measures it is curated in",
        "that script. `docs/physics_test_report.md` lists every check the suites",
        "record — this table lists only the ones anchored to something the solver",
        "cannot influence, which is the subset that can say the solver is *right*",
        "rather than merely self-consistent.",
        "",
        "**Error** is the deviation from the known result: `|measured − known|`",
        "where the check names a value, and the residual itself where the check",
        "bounds one. The sections whose fourth column reads *Observed* or *Value*",
        "tabulate the measured number instead, because there the number is the",
        "point — an order of accuracy or a ratio, not a discrepancy.",
        "",
        "**Budget used** is the error over the tolerance the test allowed. It is",
        "the review column: a value near 1 marks a check with no room left, and a",
        "value of 1e-6 marks a tolerance that is not bounding anything. A",
        "requirement written `≥`, or an interval, is not an error budget, so those",
        "rows report no fraction.",
        "",
    ]

    all_missing: list[str] = []
    all_margins: list[tuple[float, str]] = []
    body: list[str] = []
    for section in SECTIONS:
        rendered, missing, margins = _render(section, logs)
        body.extend(rendered)
        all_missing.extend(missing)
        all_margins.extend(margins)

    tightest = sorted(all_margins, reverse=True)[:5]
    if tightest:
        lines.append("## Least room left")
        lines.append("")
        lines.append(
            "The rows using the most of their error budget. These are where the "
            "suite would notice a regression first — and where a change that "
            "moves the physics slightly will fail before anything else does."
        )
        lines.append("")
        for used, description in tightest:
            lines.append(f"- **{used:.0%}** — {_escape(description)}")
        lines.append("")

    lines.extend(body)

    lines.extend(
        [
            "## Known results with no row yet",
            "",
            "Curated, not measured: Ginzburg-Landau results that are known in closed",
            "form and that this solver could be held to, but currently is not. Listed",
            "so the coverage of the table above is legible — a short table of known",
            "solutions is only reassuring if what it leaves out is written down.",
            "",
            UNCOVERED_HEADER,
        ]
    )
    for result, value, reason in UNCOVERED:
        lines.append(f"| {_escape(result)} | {_escape(value)} | {_escape(reason)} |")
    lines.append("")

    return "\n".join(lines), all_missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabulate the solver's error against known solutions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs"),
        help="Directory containing test_*.json log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "physics_error_table.md",
        help="Path to write the markdown table.",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: log directory not found at {args.input}")
        print(
            "Run the physics tests first, from the repository root:\n"
            "    python3 -m pytest packages/tdgl3d/tests/test_verification_*.py "
            "packages/tdgl3d/tests/test_physics_validation.py -q"
        )
        return

    table, missing = generate_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Table written to {args.output}")

    if missing:
        print(
            f"\n{len(missing)} reference(s) could not be resolved. Either the logs "
            "come from a partial run, or a test or check label has moved and the "
            "reference list in this script needs updating:",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
