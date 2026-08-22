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

## Numerical limits

* Forward Euler is stable for `dt < h²/(4κ²)` — the limit is set by the stiff
  `κ²∇×∇×` term, not by the ψ equation. `physics_helpers.cfl_limit` computes it.
* The covariant Laplacian is second-order accurate in `h`; forward Euler is
  first-order in `dt`. Both orders are measured, not assumed.
* The trapezoidal integrator's Jacobian-vector products are finite differences,
  so its Newton-GCR tolerances cannot usefully be pushed below the differencing
  noise floor (~1e-5 works; 1e-8 fails to converge or takes minutes).
