# Physics conventions and exact discrete identities

Reference for anyone touching the solver core. Every statement here is asserted
by a test in `packages/tdgl3d/tests/test_verification_*.py`; the test named after
each section is the one that would fail if the convention is broken.

## Units

Lengths in ξ, time in ξ²/D, fields in Φ₀/(2πξ²). In this normalisation three
numbers anchor everything:

| Quantity | Value | Verified by |
|----------|-------|-------------|
| Flux quantum Φ₀ | 2π | `test_plaquette_vorticity_is_an_exact_integer` |
| London penetration depth λ | κ (in ξ) | `test_london_penetration_depth_equals_kappa` |
| Upper critical field H_c2 | 1 | `test_lowest_landau_level_of_covariant_laplacian` |

H_c2 = 1 follows from the linearised GL equation `∂ψ/∂t = [1 − (∇ − iA)²]ψ`
together with the lowest Landau level `E₀ = B`: the uniform state is unstable to
superconductivity exactly while `B < 1`. **An applied field above 1 leaves a
normal metal, not a vortex lattice** — a common way to produce a plot with no
physics in it.

H_c1 is much lower (≈ 0.15 for κ = 2), but in a mesoscopic sample the
Bean-Livingston surface barrier delays vortex entry well above it, so the
equilibrium count `B·A/Φ₀` is an upper bound, not a prediction.

## Gauge convention

The covariant derivative is `D = ∇ − iA`. On the lattice the link variable
`φ_μ[m] = ∫ A_μ dl` over the link from node `m` to `m + e_μ`, and

* the Peierls factor multiplying `ψ_{m+e}` in the covariant Laplacian is
  `e^{−iφ_μ[m]}`, and the one multiplying `ψ_{m−e}` is `e^{+iφ_μ[m−e]}`;
* the link supercurrent is `J_μ = Im(ψ*_m e^{−iφ_μ[m]} ψ_{m+e}) / h_μ`;
* the field equation is `∂A/∂t = J_s − κ²∇×∇×A`;
* the gauge transformation is `ψ → ψ e^{iχ}`, `φ_μ → φ_μ + (χ_{m+e} − χ_m)`.

These four are one convention, not four choices. Flipping the Peierls sign in
the Laplacian alone still screens correctly (the term linear in `A` keeps its
sign) and still nucleates vortices, so it survives a Meissner test and a vortex
test — but it makes `∇θ` and `A` enter the current with opposite signs, and the
symptom is a **mix of +1 and −1 windings in a uniform applied field**.
`test_rhs_is_gauge_covariant` catches it directly, as an O(1) violation.

## Exact discrete identities

These hold to round-off, not to discretisation order. If one of them shows an
O(h) residual, the stencils have been paired wrongly — do not widen the
tolerance.

| Identity | Requires | Verified by |
|----------|----------|-------------|
| `∇·B = 0` | *forward* divergence of the *forward* plaquette curl | `test_divergence_of_discrete_curl_is_exactly_zero` |
| `∇·(∇×∇×A) = 0` | backward divergence of the link equation | `test_curl_curl_operator_is_divergence_free` |
| fluxoid number ∈ ℤ | `Σ wrap(Δθ − φ) + Σφ = 2πn` around any closed loop | `test_plaquette_vorticity_is_an_exact_integer` |
| lattice Stokes | loop fluxoid = enclosed plaquette vorticity | `test_fluxoid_equals_enclosed_vorticity_for_any_contour` |
| Lyapunov decay | `gl_free_energy` matches the RHS term by term | `test_free_energy_decreases_monotonically_at_zero_field` |

The free energy in `tdgl3d.physics.free_energy` is the exact lattice counterpart
of the functional the solver descends; at zero applied field it must decrease on
*every* step. A tolerance derived from the observed drift makes that check
unfalsifiable, which is how a sign error in the RHS can sit undetected behind a
passing test.

## Index ordering

The two numberings run in **opposite** directions, and on a cubic grid the
difference is invisible:

| Array | Layout | Strides |
|-------|--------|---------|
| Full grid (`dim_x`) | i-fastest: `i + (Nx+1)j + (Nx+1)(Ny+1)k` | `(1, Nx+1, (Nx+1)(Ny+1))` |
| Interior (`n_interior`) | i-slowest, C order over `(Nx-1, Ny-1, Nz-1)` | `((Ny-1)·nz, nz, 1)` with `nz = max(Nz-1, 1)` |

Consequences worth remembering:

* `interior_to_full` is **not sorted** — `np.searchsorted` on it returns the
  wrong node. Build a dict, or use the strides.
* `reshape(Nx-1, Ny-1, max(Nz-1,1))` on an interior array is correct and indexes
  `[i, j, k]`.
* Applying full-grid strides to an interior array transposes x and z.

`test_interior_numbering_matches_documented_strides` and
`test_bfield_evaluators_agree` run on deliberately non-cubic grids for this
reason.

## Boundary conditions

* The normal link variables on the outer faces are zeroed — a gauge choice, and
  the reason local gauge transformations in the tests are supported strictly
  inside the boundary.
* `J_n = 0` on every external face comes from the ψ boundary condition, not from
  the zeroed links: on the high faces the normal link is a live degree of
  freedom and the condition is carried by the ghost value
  `ψ_{N} = e^{iφ}ψ_{N-1}`. Checking that the links are zero therefore verifies
  much less than checking the current
  (`test_normal_supercurrent_vanishes_on_external_boundaries`).
* Each boundary plaquette is given the applied flux by offsetting the one ghost
  link that closes it. Where two *hi* faces meet, the plaquette is closed by two
  such links and the offset is split between them; a full offset on both put
  `2 B_applied` on that plaquette and left an unbalanced curl-curl force that
  made its links drift without bound. `test_applied_flux_on_boundary_plaquettes`
  measures the whole boundary ring.
* The ghost-ring corner plaquette at `(0, 0)` carries zero flux and cannot be
  given any: the gauge choice above forces `A_x = 0` along `x = 0` *and*
  `A_y = 0` along `y = 0`, which together pin `B` there. That plaquette never
  enters the dynamics (no interior stencil references it) and is outside the
  reported interior field, so it is a diagnostic artifact rather than a physics
  error.

## Symmetry: nodes versus plaquettes

`ψ` lives on the interior **nodes** `1 … Nx-1`, a set that the reflection
`i → Nx − i` maps onto itself, so the full array can be compared with its
reverse. `B` lives on **plaquettes** anchored at those nodes but spanning
`1 … Nx`; the mirror image of the anchor-1 plaquette is the ghost anchor-0
plaquette, which the interior array does not carry. Drop the last anchor before
reflecting, or a perfectly symmetric solution reads as ~1e-3 asymmetric.

## Heterostructures

Two settings decide whether an S/I/S stack behaves like one, and neither is
guarded by the type system:

* **A non-superconducting layer needs κ > 0.** In a layer with `ψ = 0` the
  supercurrent term of `∂A/∂t = J_s − κ²∇×∇×A` vanishes, so the layer relaxes
  towards the magnetostatic solution `∇×∇×A = 0` at a rate set by κ² — any
  positive κ gives the same steady state. `κ = 0` removes the only remaining
  term: the gauge field is frozen at its initial value and the layer transmits
  nothing. `Layer(kappa=0.0, is_superconductor=False)` is therefore a modelling
  choice with consequences, not a neutral way of saying "not a superconductor"
  (`test_insulator_kappa_controls_field_transmission`).
* **Superconducting layers need to be thicker than the proximity length.** The
  oxide suppresses ψ over roughly a coherence length on each side of the
  interface. A 1 ξ layer is suppressed all the way through — |ψ| ≈ 1e-4 — and
  every phase-derived quantity measured on it is noise, while still producing
  plausible-looking fluxoid staircases and field scans. 4 ξ layers recover
  |ψ| ≈ 0.99 in their middle. `test_the_ring_is_superconducting` is the cheap
  guard; run it before trusting anything downstream.

## Geometry: two conventions that quietly break symmetry

Both of these produced a device that looked symmetric in every summary number
while being asymmetric at the 1e-3 level in the fields.

* **Ray casting is half-open.** `point_in_polygon` counts a point on the
  low-x/low-y edge as outside and one on the high-x/high-y edge as inside — a
  consistent tiling rule, but not a mirror-symmetric one. Holes are normally
  specified at round coordinates, so their edges land exactly on grid nodes and
  the carved region comes out shifted by half a cell: a hole given as `[3, 7]`
  removed nodes 4…7, centred at 5.5 in a film centred at 5.
  `identify_hole_nodes` now takes the **closed** region by default
  (`edge_tolerance`), so `[3, 7]` removes nodes 3…7 (`test_centred_hole_is_centred`).
* **Layer thicknesses are in cells; materials live on nodes.** The two
  interfaces of an S/I/S stack are shared between layers. Assigning each node to
  the cell range `[k_start, k_end)` containing it gives the lower interface to
  the oxide and the upper one to the top layer, so the top layer ends up with
  one more superconducting node than the bottom. Both interfaces belong to the
  oxide instead; equal superconducting thicknesses then give equal node counts
  and the stack is exactly symmetric about its mid-plane
  (`test_stack_is_mirror_symmetric_about_its_midplane`). The oxide occupies
  `insulator.thickness_z + 1` nodes as a result.

With both fixed, a relaxed S/I/S ring started from a noiseless state is
symmetric under x → −x, y → −y, z → −z and a 90° rotation to ~1e-16
(`test_the_relaxed_ring_is_symmetric`).

Note that a noiseless symmetric device relaxes to an **exact** fixed point
(residual ~1e-14). That is the right state for a symmetry check, but it means a
metastable branch can only be broken by round-off, which delays flux entry by an
amount set by floating-point precision rather than by the energy barrier. Seed a
perturbation when measuring an entry threshold, and check the answer against a
much smaller one.

## SI units

`tdgl3d.GLUnits` converts between SI and solver units. It needs ξ **at the
temperature of interest**, not ξ₀: Ginzburg-Landau is a near-T_c theory and
ξ(T) = ξ₀/√(1 − T/T_c) diverges there, while κ = λ/ξ does not. The same physical
geometry is a different simulation at different temperatures — a 4 µm film is
40 ξ across for Nb at T/T_c ≈ 0.86 (ξ = 100 nm) and 400 ξ across well below T_c,
which is the difference between a minute and an intractable run.

## Analytical benchmarks

Two limits of the coupled equations have closed-form solutions, and each isolates
one of them. `tdgl3d.physics.analytic` carries both, along with the coordinate
helpers the comparisons need.

* **London** — `|ψ| = 1` (weak field), so the ψ-equation drops out and the gauge
  field obeys `∇²B = B/λ²`. On a square with the field pinned on the boundary
  `london_square_2d` is the exact Fourier solution. Two practical notes: the
  series converges to `B₀` on the open edges but `B₀/2` at the corners, so stay
  a few cells clear of a corner; and the naive `cosh(ka)/cosh(kb)` overflows to
  `nan` past about the 400th term, which is why `_cosh_ratio` exists.
* **Pair-breaking wall** — zero field, so the gauge field drops out and
  `ψ'' = -ψ + ψ³` gives `tanh((x - x₀)/√2)`. The offset is **not fitted**:
  matching ψ and ψ' to the insulator's `ψ = u e^{x/√τ}` gives
  `√τ u² + √2 u - √τ = 0`, hence `u = 0.213422` and `x₀ = -0.306536 ξ` at the
  solver's `τ = 0.1` (`INSULATOR_RELAXATION_TIME`). The `√2` is the physics: the
  Ginzburg-Landau healing length is `√2 ξ`, not `ξ`.

**Both comparisons depend on where the discrete quantities sit, to half a cell.**
Getting it wrong degrades the observed order from 2 to 1 while leaving the
profile looking correct, which is a slow thing to notice:

* `B` is plaquette-centred, and the boundary condition pins the whole ring of
  plaquettes — *including the ghost anchor 0* that the interior array does not
  carry. The pinned-to-pinned span is `(N-1)h` wide and interior entry `i-1`
  sits at `i·h` from its low end, which is what `plaquette_positions` returns.
  Using the plaquette centres measured from `0` displaces the profile by a full
  cell.
* The material coefficient jumps *between* nodes, so the effective interface is
  the midpoint of the last insulator node and the first superconducting one.
  Anchoring on either node costs a factor of `h`.

**A closed-form reference has its own error, and it can be the larger one.** The
truncated series' Gibbs ringing does not shrink with `h`, so once it is used as
the reference for a convergence study it becomes the accuracy floor. At the
original 201-term default it capped the observed order of the max error at 1.14;
at 2001 it is 1.73. Before reading a convergence order as a statement about the
solver, check that the model is the more accurate of the two.

Where the boundary condition is Dirichlet the solver is *exact*, not
approximate, and should be checked that way — against the applied field itself
rather than against the series. The pinned boundary plaquettes agree to 6e-16.

Measured against these, the solver reaches rms `4.08e-03 · B₀` (London) and
`4.97e-02` in `|ψ|` (wall) at `h = 1 ξ`, converging at order 1.82 and 1.69
respectively. Neither reaches a clean 2, for reasons that are understood rather
than tolerated: the London residual mixes a second-order bulk with the series'
floor, and the wall's coefficient jump is a first-order feature locally, which
bounds an rms over a window holding `O(1)` such points at `h^1.5`. Because the
wall comparison needs an interface position at all, it is backed by the
offset-free form of the same physics — the first integral `ψ' = (1 - ψ²)/√2`
checked pointwise, which involves no position, matching constant or fit and
holds to rms `8.03e-04` at `h = 0.25 ξ`.

**A z-invariant problem is the cheapest 3-D check there is.** With no
z-dependence in the geometry or the boundary condition, the 3-D discrete
equations reduce term-for-term to the 2-D ones — `∂²/∂z²` annihilates a
z-invariant field and `φ_z` is driven only by `J_{s,z} = 0` — so the two paths
must reach the *same fixed point*, to solver precision rather than to
discretisation error. Measured: 2.2e-16 spread across z-slices, 1.7e-10 against
the 2-D run. This is what catches an index-ordering bug in the 3-D path, which
is otherwise expensive to see.

See `test_verification_analytic.py` and gallery §14.

## Numerical limits

* Forward Euler is stable for `dt < h²/(4κ²(d-1))` — the limit is set by the
  stiff `κ²∇×∇×` term, not by the ψ equation, and it **depends on dimension**.
  The familiar `h²/(4κ²)` is the 2D case; in 3D each link variable picks up a
  second transverse Laplacian direction, the spectral radius of the curl-curl
  block doubles (61.6 → 123.1 on an 8³ grid at κ=2, h=0.5) and the limit halves.
  Measured limits: 1.06x `h²/(4κ²)` in 2D, 0.72x in 3D — a 3D run at "0.9 CFL"
  using the 2D formula diverges. `physics_helpers.cfl_limit` carries the factor;
  `test_forward_euler_is_stable_below_the_cfl_limit` covers both dimensions.
* The covariant Laplacian is second-order accurate in `h`; forward Euler is
  first-order in `dt`. Both orders are measured, not assumed.
* The trapezoidal integrator's Jacobian-vector products are finite differences,
  so its Newton-GCR tolerances cannot usefully be pushed below the differencing
  noise floor (~1e-5 works; 1e-8 fails to converge or takes minutes).
* The trapezoidal integrator does not pay for itself at these parameters. Its
  Krylov iteration count grows about as fast as the step size it buys, so cost
  per unit simulated time bottoms out around 2.8x forward Euler's — measured
  over `dt` from 0.02 to 0.8. Every figure in `docs/figures/` uses Euler.
* **A diagonal (Jacobi) preconditioner does not fix that.** Measured 0.92x to
  1.10x against no preconditioner across κ = 2, 5, 10, h = 1 and 0.5, and step
  sizes from 2x to 32x the explicit limit. The operator is nearly
  constant-coefficient: within each block the diagonal is the same number at
  every node, so scaling by it is close to a uniform rescale and changes no
  eigenvalue ratio. The conditioning is the Laplacian's spread across spatial
  frequencies, which only a frequency-aware method touches — multigrid, or a
  sine-transform solve, which the structured Cartesian grid would suit.
  `tgcr_matrix_free_trap(..., scaling=...)` is the seam one plugs into.
