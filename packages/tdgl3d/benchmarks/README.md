# Cross-tool benchmarks

Three codes, the same problems, and closed-form answers wherever one
exists. The point is to say two different things with numbers:

* how far each code is from an **exact solution**, and
* how far each code is from **the other codes**, including in the
  regimes where no exact solution exists.

The second is not implied by the first. Two codes can each sit within a
percent of a closed form in the limit where it applies and disagree by
much more in the crossover between limits, which is where the interesting
physics is and where nothing can be checked analytically.

## The three codes are not the same model

| | `tdgl3d` | pyTDGL | SuperScreen |
|---|---|---|---|
| Equations | 3-D TDGL, ψ and **A** | 2-D TDGL, ψ and sheet **A** | thin-film London only |
| Order parameter | yes | yes | **no** |
| Geometry | Cartesian grid, real thickness | triangular mesh, zero thickness | triangular mesh, zero thickness |
| Screening | 3-D Maxwell in a box of vacuum | Biot-Savart from the sheet current | Biot-Savart from the sheet current |
| Pearl length Λ | an outcome of κ, thickness and ψ | an input, λ²/d | an input |
| Surface against vacuum | pair-breaking | free (∇ψ·n̂ = 0) | n/a |

Those differences decide which pairs can be compared on which problem,
and they are the reason the benchmarks come in two kinds rather than one.

## Benchmarks

### `pearl_disk.py` — a thin disk in a perpendicular field

Magnetostatics, which all three codes do. The reported quantity is the
dimensionless `μ = m / m_London`, the magnetic moment over its
weak-screening closed form, computed in each code's own units so nothing
has to be converted between them. Plotted against `Λ/R`, all three
should fall on one curve with an exact answer at each end:

* `Λ/R → ∞`: `μ → 1` — the London limit, exact for a *disk* because the
  symmetric gauge already satisfies both `∇·A = 0` and `A·n̂ = 0` on a
  circular edge.
* `Λ/R → 0`: `μ → (64/3π)(Λ/R)` — the perfectly diamagnetic thin disk,
  `m = -(8/3) H_a R³`.

Between them there is no closed form, and the codes are compared with
each other instead.

### `gl_wall.py` — a pair-breaking wall

The order-parameter equation on its own, which SuperScreen does not
have. At zero field Ginzburg-Landau reduces to `ψ'' = -ψ + ψ³`, whose
first integral `ψ' = (1 - ψ²)/√2` holds pointwise with no interface
position, matching constant or fit in it. That last property is what
makes it usable here: the two codes create the wall differently and put
the interface in different places, so a `tanh((x - x₀)/√2)` comparison
would be comparing offsets.

## Running

The two reference codes are optional dependencies:

```bash
pip install tdgl superscreen        # pyTDGL and SuperScreen
cd packages/tdgl3d
python3 -m benchmarks.run superscreen   # ~1 minute
python3 -m benchmarks.run pytdgl        # ~20 minutes
python3 -m benchmarks.run tdgl3d        # ~25 minutes
python3 -m benchmarks.run tdgl3d-convergence   # ~45 minutes
python3 -m benchmarks.run wall          # ~5 minutes
python3 -m benchmarks.run report        # writes REPORT.md
python3 ../../docs/figures/cross_tool_benchmark.py
```

Each subcommand writes its own key into `results.json`, so a sweep that
fails half way through does not lose the half that finished.

## Traps worth knowing about

* **SuperScreen's mesh buffer.** `Device.make_mesh` pads the film polygon
  by 5% of its bounding box by default and solves on the padded polygon.
  For a disk that is a 10% larger radius and a moment 1.5% high at every
  Λ — a systematic that does not shrink under mesh refinement, so it
  cannot be mistaken for discretisation error. The runner passes
  `buffer=0.0`.
* **`μ = 1` is an asymptote, not a value.** Screening pulls `μ` below 1
  by about `0.15 R/Λ`, so `|μ - 1|` at finite Λ is mostly physics.
  `report.weak_limit` fits that term out and quotes the intercept, which
  the closed form does fix at 1.
* **Resolving Λ.** In the complete-screening limit the sheet current
  varies over Λ, so a mesh coarser than Λ is measuring the mesh. The
  small-`Λ/R` end of the sweep is mesh-limited for both thin-film codes
  and is reported as such.
* **tdgl3d's vacuum is pair-breaking**, so a film a few ξ thick has
  `|ψ| < 1` everywhere and its sheet superfluid density is not the
  geometric thickness. The runner measures `∫|ψ|²dz` and reports Λ from
  that rather than from `κ²/d` — for the sweep's film the two differ by
  46%. That interface is also not grid-converged at `h = 1`: the
  insulator relaxes ψ over `√0.1 ≈ 0.32 ξ`, so how hard the vacuum
  pair-breaks the film still depends on the spacing.
* **`h = 1` hides unit bugs.** Two of the errors this benchmark found —
  the missing `1/h` in the supercurrent, and a κ² cached across a change
  of κ — are exactly the identity on a unit cubic grid, which is what
  every test, example and figure in the repository uses. The
  grid-refinement probe is there for that reason and not only for
  accuracy.
