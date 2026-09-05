# Physics Verification Report

**Run timestamp:** 2026-08-29T17:59:25.538942
**Tests:** 87/87 passed, 0 failed
**Checks:** 294/294 passed (0 failed)

Each check records the measured value, the value physics requires and the tolerance allowed, so every line below is falsifiable. Tolerances near machine precision mark exact discrete identities; the wider ones are discretisation error bounds stated up front rather than fitted to the measurement.

## Checks

### Gauge invariance

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_global_phase_rotation_is_exact_symmetry** | | | | |
| max\|dψ/dt rotation error\| | 7.022e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max\|dφ/dt change\| | 4.441e-16 | <= 1e-12 | 1.000e-12 | PASS |
| **test_observables_are_gauge_invariant[10x10x1]** | | | | |
| max Δ\|ψ\| | 2.220e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max ΔB | 2.776e-16 | <= 1e-11 | 1.000e-11 | PASS |
| max ΔJ_s | 2.220e-16 | <= 1.04e-11 | 1.041e-11 | PASS |
| Δ free energy | 0 | <= 5.47e-08 | 5.466e-08 | PASS |
| **test_observables_are_gauge_invariant[6x7x5]** | | | | |
| max Δ\|ψ\| | 2.220e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max ΔB | 2.220e-16 | <= 1e-11 | 1.000e-11 | PASS |
| max ΔJ_s | 1.665e-16 | <= 1.55e-11 | 1.553e-11 | PASS |
| Δ free energy | 2.842e-14 | <= 1.9e-07 | 1.904e-07 | PASS |
| **test_observables_are_gauge_invariant[9x7x1]** | | | | |
| max Δ\|ψ\| | 3.331e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max ΔB | 3.331e-16 | <= 1e-11 | 1.000e-11 | PASS |
| max ΔJ_s | 2.220e-16 | <= 1e-11 | 1.000e-11 | PASS |
| Δ free energy | 0 | <= 5.61e-08 | 5.608e-08 | PASS |
| **test_rhs_covariant_with_material_map** | | | | |
| max covariance violation | 3.997e-15 | <= 1.84e-10 | 1.843e-10 | PASS |
| **test_rhs_is_gauge_covariant[10x10x1]** | | | | |
| max\|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)\| | 9.155e-16 | <= 8.15e-11 | 8.146e-11 | PASS |
| max\|dφ/dt(GX) − dφ/dt(X)\| | 2.276e-15 | <= 8.15e-11 | 8.146e-11 | PASS |
| **test_rhs_is_gauge_covariant[6x7x5]** | | | | |
| max\|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)\| | 1.335e-15 | <= 1.21e-10 | 1.211e-10 | PASS |
| max\|dφ/dt(GX) − dφ/dt(X)\| | 5.773e-15 | <= 1.21e-10 | 1.211e-10 | PASS |
| **test_rhs_is_gauge_covariant[9x7x1]** | | | | |
| max\|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)\| | 9.930e-16 | <= 1.02e-10 | 1.016e-10 | PASS |
| max\|dφ/dt(GX) − dφ/dt(X)\| | 3.553e-15 | <= 1.02e-10 | 1.016e-10 | PASS |
| **test_vortex_count_is_gauge_invariant** | | | | |
| vortices present (test would be vacuous otherwise) | 8 | >= 1 | 1 | PASS |
| vortex count after gauge change | 8 | 8 | 0 | PASS |
| max Δ(plaquette vorticity) | 2.297e-16 | <= 1e-09 | 1.000e-09 | PASS |
| max \|winding change\| | 1.110e-16 | <= 1e-09 | 1.000e-09 | PASS |

### Conservation laws and identities

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_curl_curl_operator_is_divergence_free[7x6x6]** | | | | |
| max\|∇·(∇×∇×A)\| / scale | 3.243e-16 | <= 1e-13 | 1.000e-13 | PASS |
| **test_curl_curl_operator_is_divergence_free[9x8x1]** | | | | |
| max\|∇·(∇×∇×A)\| / scale | 2.918e-16 | <= 1e-13 | 1.000e-13 | PASS |
| **test_divergence_of_discrete_curl_is_exactly_zero[6x7x8]** | | | | |
| bulk nodes tested | 120 | >= 1 | 1 | PASS |
| max\|∇·B\| / max\|B\| | 3.204e-16 | <= 1e-13 | 1.000e-13 | PASS |
| **test_divergence_of_discrete_curl_is_exactly_zero[8x7x1]** | | | | |
| bulk nodes tested | 30 | >= 1 | 1 | PASS |
| max\|∇·B\| / max\|B\| | 0 | <= 1e-13 | 1.000e-13 | PASS |
| **test_forward_euler_is_stable_below_the_cfl_limit[2d]** | | | | |
| max\|ψ\|² at dt = 0.9 dt_CFL | 1 | 1 | 0.05 | PASS |
| min\|ψ\|² at dt = 0.9 dt_CFL | 1 | 1 | 0.05 | PASS |
| run at dt = 4 dt_CFL loses the superconducting state | 1 | >= 1 | 1 | PASS |
| **test_forward_euler_is_stable_below_the_cfl_limit[3d]** | | | | |
| max\|ψ\|² at dt = 0.9 dt_CFL | 1.00005 | 1 | 0.05 | PASS |
| min\|ψ\|² at dt = 0.9 dt_CFL | 1.00002 | 1 | 0.05 | PASS |
| run at dt = 4 dt_CFL loses the superconducting state | 1 | >= 1 | 1 | PASS |
| **test_free_energy_decreases_monotonically_at_zero_field** | | | | |
| energy released (test would be vacuous otherwise) | 44.1063 | >= 1 | 1 | PASS |
| steps on which F increased | 0 | <= 0 | 0 | PASS |
| worst single-step ΔF / energy released | -9.570e-10 | <= 1e-09 | 1.000e-09 | PASS |
| **test_free_energy_decreases_while_relaxing_in_a_field** | | | | |
| energy released | 53.0219 | >= 0.5 | 0.5 | PASS |
| worst single-step ΔF / energy released | -1.320e-08 | <= 1e-06 | 1.000e-06 | PASS |
| **test_normal_supercurrent_vanishes_on_external_boundaries** | | | | |
| bulk current scale (non-trivial state) | 0.39737 | >= 0.0001 | 1.000e-04 | PASS |
| max\|J_n\| on x_lo face | 4.229e-19 | <= 1e-12 | 1.000e-12 | PASS |
| max\|J_n\| on x_hi face | 8.535e-19 | <= 1e-12 | 1.000e-12 | PASS |
| max\|J_n\| on y_lo face | 4.442e-19 | <= 1e-12 | 1.000e-12 | PASS |
| max\|J_n\| on y_hi face | 7.671e-19 | <= 1e-12 | 1.000e-12 | PASS |
| max\|J_n\| on z_lo face | 4.120e-19 | <= 1e-12 | 1.000e-12 | PASS |
| max\|J_n\| on z_hi face | 5.866e-19 | <= 1e-12 | 1.000e-12 | PASS |
| **test_supercurrent_is_divergence_free_in_steady_state** | | | | |
| state drift between saved steps | 8.183e-12 | <= 1e-06 | 1.000e-06 | PASS |
| max\|∇·J_s\| · h / max\|J_s\| | 5.588e-14 | <= 1e-06 | 1.000e-06 | PASS |
| **test_uniform_state_is_an_exact_fixed_point** | | | | |
| max\|dX/dt\| | 0 | <= 1e-13 | 1.000e-13 | PASS |
| kinetic + magnetic energy | 0 | <= 1e-13 | 1.000e-13 | PASS |
| condensation energy per unit volume | -0.5 | -0.5 | 1.000e-12 | PASS |

### Symmetry and boundary conditions

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_applied_field_vectors_are_uniform_on_each_face** | | | | |
| max deviation of Bz on x_lo | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of Bz on x_hi | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of Bz on y_lo | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of Bz on y_hi | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of Bx on y_lo | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of Bx on z_hi | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of By on x_hi | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max deviation of By on z_lo | 0 | <= 1e-14 | 1.000e-14 | PASS |
| **test_applied_flux_on_boundary_plaquettes[8x8]** | | | | |
| flux on the hi/hi corner plaquette | 0.12 | 0.12 | 1.200e-11 | PASS |
| max deviation of boundary ring from B_applied | 2.776e-17 | <= 1.2e-11 | 1.201e-11 | PASS |
| state drift once relaxed | 8.954e-11 | <= 1e-08 | 1.000e-08 | PASS |
| screened interior field / applied | 0.941456 | <= 0.99 | 0.99 | PASS |
| **test_applied_flux_on_boundary_plaquettes[9x7]** | | | | |
| flux on the hi/hi corner plaquette | 0.12 | 0.12 | 1.200e-11 | PASS |
| max deviation of boundary ring from B_applied | 2.776e-17 | <= 1.2e-11 | 1.201e-11 | PASS |
| state drift once relaxed | 8.286e-11 | <= 1e-08 | 1.000e-08 | PASS |
| screened interior field / applied | 0.942774 | <= 0.99 | 0.99 | PASS |
| **test_bfield_evaluators_agree[5x7x6]** | | | | |
| max\|eval_bfield(subset) − reference\| | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max\|eval_bfield(all interior) − reference\| | 0 | <= 1e-14 | 1.000e-14 | PASS |
| len(bfield_interior) | 60 | 60 | 0 | PASS |
| **test_bfield_evaluators_agree[6x6x1]** | | | | |
| max\|eval_bfield(subset) − reference\| | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max\|eval_bfield(all interior) − reference\| | 0 | <= 1e-14 | 1.000e-14 | PASS |
| len(bfield_interior) | 16 | 16 | 0 | PASS |
| **test_bfield_evaluators_agree[9x5x1]** | | | | |
| max\|eval_bfield(subset) − reference\| | 0 | <= 1e-14 | 1.000e-14 | PASS |
| max\|eval_bfield(all interior) − reference\| | 0 | <= 1e-14 | 1.000e-14 | PASS |
| len(bfield_interior) | 21 | 21 | 0 | PASS |
| **test_c4_symmetry_of_a_square_device** | | | | |
| Bz contrast (screening present, so the test is non-trivial) | 0.0322938 | >= 0.001 | 0.001 | PASS |
| max\|ψ\| − R₉₀\|ψ\|\| | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max\|Bz − R₉₀Bz\| | 5.551e-17 | <= 1e-12 | 1.000e-12 | PASS |
| **test_field_reversal_flips_b_and_preserves_psi** | | | | |
| B scale (non-trivial state) | 0.334784 | >= 0.001 | 0.001 | PASS |
| max\|Bz(+B) + Bz(−B)\| | 0 | <= 1e-12 | 1.000e-12 | PASS |
| max\| \|ψ(+B)\| − \|ψ(−B)\| \| | 0 | <= 1e-12 | 1.000e-12 | PASS |
| **test_indices_are_within_bounds_on_ragged_grids** | | | | |
| index arrays checked | 67 | >= 40 | 40 | PASS |
| worst overshoot past the last valid index | -1 | <= -1 | -1 | PASS |
| **test_interior_numbering_matches_documented_strides[5x7x6]** | | | | |
| mismatched entries of interior_to_full | 0 | <= 0 | 0 | PASS |
| reshape stride mismatch | 0 | <= 0 | 0 | PASS |
| **test_interior_numbering_matches_documented_strides[6x6x1]** | | | | |
| mismatched entries of interior_to_full | 0 | <= 0 | 0 | PASS |
| reshape stride mismatch | 0 | <= 0 | 0 | PASS |
| **test_interior_numbering_matches_documented_strides[9x5x1]** | | | | |
| mismatched entries of interior_to_full | 0 | <= 0 | 0 | PASS |
| reshape stride mismatch | 0 | <= 0 | 0 | PASS |
| **test_mirror_symmetry_of_a_rectangular_device** | | | | |
| Bz contrast | 0.0272986 | >= 0.001 | 0.001 | PASS |
| max\|ψ(x) − ψ(−x)\| | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max\|ψ(y) − ψ(−y)\| | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max\|Bz(x) − Bz(−x)\| | 5.551e-17 | <= 1e-12 | 1.000e-12 | PASS |
| max\|Bz(y) − Bz(−y)\| | 8.327e-17 | <= 1e-12 | 1.000e-12 | PASS |
| **test_solution_reshape_helpers_are_consistent** | | | | |
| shape[0] | 10 | 10 | 0 | PASS |
| shape[1] | 5 | 5 | 0 | PASS |
| max\|reshape − stride-indexed\| | 0 | <= 1e-15 | 1.000e-15 | PASS |

### Analytic limits

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_bfield_matches_the_exact_london_solution** | | | | |
| min \|ψ\| at the coarsest h (London limit is applicable) | 0.999627 | >= 0.99 | 0.99 | PASS |
| field at the centre / B₀ | 0.0702701 | <= 0.2 | 0.2 | PASS |
| \|B(pinned plaquette) − B_applied\| / B₀ | 5.551e-16 | <= 1e-12 | 1.000e-12 | PASS |
| rms \|solver − model\| / B₀ at h = 1 ξ | 0.00408256 | <= 0.005 | 0.005 | PASS |
| rms \|solver − model\| / B₀ at h = 0.25 ξ | 3.290e-04 | <= 0.0005 | 5.000e-04 | PASS |
| max \|solver − model\| / B₀ at h = 0.25 ξ | 4.463e-04 | <= 0.001 | 0.001 | PASS |
| observed order in h (whole profile) | 1.81668 | >= 1.7 | 1.7 | PASS |
| observed order in h (bulk, >1 ξ from an edge) | 1.84113 | >= 1.75 | 1.75 | PASS |
| **test_covariant_laplacian_is_second_order_accurate** | | | | |
| observed order of accuracy | 1.99835 | 2 | 0.15 | PASS |
| error at h = 0.1 | 2.214e-04 | <= 0.001 | 0.001 | PASS |
| **test_covariant_laplacian_reduces_to_the_standard_laplacian** | | | | |
| max\|L_covariant(A=0) − L_standard\| | 0 | <= 1e-13 | 1.000e-13 | PASS |
| **test_forward_euler_is_first_order_in_dt** | | | | |
| observed order in dt | 1.00371 | 1 | 0.1 | PASS |
| Richardson error at the smallest dt | 1.854e-05 | <= 0.001 | 0.001 | PASS |
| **test_insulator_order_parameter_decays_with_the_stated_time_constant** | | | | |
| points used in the fit | 30 | >= 5 | 5 | PASS |
| fitted τ | 0.0929158 | 0.1 | 0.02 | PASS |
| residual \|ψ\| in the insulator | 0.0532326 | <= 0.15 | 0.15 | PASS |
| **test_london_penetration_depth_equals_kappa[kappa=1.5]** | | | | |
| state drift (equilibrium reached) | 3.696e-08 | <= 1e-06 | 1.000e-06 | PASS |
| min \|ψ\| (still Meissner, no vortices) | 0.999794 | >= 0.9 | 0.9 | PASS |
| λ from the screening profile | 1.61931 | 1.5 | 0.15 | PASS |
| **test_london_penetration_depth_equals_kappa[kappa=3.0]** | | | | |
| state drift (equilibrium reached) | 1.009e-07 | <= 1e-06 | 1.000e-06 | PASS |
| min \|ψ\| (still Meissner, no vortices) | 0.998855 | >= 0.9 | 0.9 | PASS |
| λ from the screening profile | 3.13849 | 3 | 0.3 | PASS |
| **test_london_series_satisfies_its_own_equation** | | | | |
| max \|∇²B − B/λ²\| at the finer grid | 2.400e-05 | <= 0.0001 | 1.000e-04 | PASS |
| residual ratio on halving the check grid | 4.00022 | 4 | 1 | PASS |
| max \|B − B₀\| on an edge, 1 ξ from a corner | 0.00150745 | <= 0.002 | 0.002 | PASS |
| finite values at n_terms = 201 | 1 | 1 | 0 | PASS |
| finite values at n_terms = 2001 | 1 | 1 | 0 | PASS |
| finite values at n_terms = 8001 | 1 | 1 | 0 | PASS |
| edge ringing at the default n_terms = 2001 | 5.883e-04 | <= 0.002 | 0.002 | PASS |
| ringing ratio 2001/201 | 0.0734807 | <= 0.2 | 0.2 | PASS |
| ringing ratio 8001/2001 | 0.249962 | <= 0.5 | 0.5 | PASS |
| **test_london_slab_is_the_wide_limit_of_the_square** | | | | |
| max \|square − slab\| at W = 4 λ | 0.1223 | >= 0.05 | 0.05 | PASS |
| max \|square − slab\| at W = 32 λ | 2.249e-07 | <= 1e-06 | 1.000e-06 | PASS |
| deviation ratio W = 8 λ / 4 λ | 0.239485 | <= 0.3 | 0.3 | PASS |
| deviation ratio W = 16 λ / 8 λ | 0.0222226 | <= 0.3 | 0.3 | PASS |
| deviation ratio W = 32 λ / 16 λ | 3.455e-04 | <= 0.3 | 0.3 | PASS |
| edge deviation floor at W = 32 λ | 3.180e-04 | <= 0.001 | 0.001 | PASS |
| **test_lowest_landau_level_of_covariant_laplacian[Bz=0.1]** | | | | |
| max\|∇×A − B\| for the Landau-gauge links | 5.274e-16 | <= 1e-13 | 1.000e-13 | PASS |
| non-Hermiticity of −(∇ − iA)² | 0 | <= 1e-13 | 1.000e-13 | PASS |
| lowest eigenvalue E₀ | 0.100006 | 0.1 | 0.003 | PASS |
| **test_lowest_landau_level_of_covariant_laplacian[Bz=0.2]** | | | | |
| max\|∇×A − B\| for the Landau-gauge links | 1.055e-15 | <= 1e-13 | 1.000e-13 | PASS |
| non-Hermiticity of −(∇ − iA)² | 0 | <= 1e-13 | 1.000e-13 | PASS |
| lowest eigenvalue E₀ | 0.198753 | 0.2 | 0.006 | PASS |
| **test_order_parameter_matches_the_exact_wall_solution** | | | | |
| \|ψ\| at the interface from matching | 0.213422 | 0.2134 | 1.000e-04 | PASS |
| rms error at h = 1 ξ | 0.0496938 | <= 0.08 | 0.08 | PASS |
| rms error at h = 0.25 ξ | 0.00476747 | <= 0.01 | 0.01 | PASS |
| max error at h = 0.25 ξ | 0.0162006 | <= 0.03 | 0.03 | PASS |
| observed order in h | 1.69088 | >= 1.5 | 1.5 | PASS |
| rms \|ψ′ − (1 − ψ²)/√2\| at h = 1 ξ | 0.00934297 | <= 0.02 | 0.02 | PASS |
| rms \|ψ′ − (1 − ψ²)/√2\| at h = 0.25 ξ | 8.034e-04 | <= 0.002 | 0.002 | PASS |
| observed order in h (first integral) | 1.76982 | >= 1.5 | 1.5 | PASS |
| **test_penetration_depth_converges_with_grid_refinement** | | | | |
| \|λ − κ\| at h = 1.0 | 0.348405 | <= 0.7 | 0.7 | PASS |
| \|λ − κ\| at h = 0.5 | 0.105629 | <= 0.2 | 0.2 | PASS |
| error ratio fine/coarse | 0.30318 | <= 0.75 | 0.75 | PASS |
| **test_three_dimensional_solver_reproduces_the_two_dimensional_london_solution** | | | | |
| min \|ψ\| in the 3-D run | 0.999627 | >= 0.99 | 0.99 | PASS |
| spread of Bz across z-slices / B₀ | 1.110e-16 | <= 1e-10 | 1.000e-10 | PASS |
| max \|B_3D − B_2D\| / B₀ | 1.706e-10 | <= 1e-08 | 1.000e-08 | PASS |
| max \|B_3D − model\| / B₀ | 0.00490852 | <= 0.01 | 0.01 | PASS |
| **test_trapezoidal_agrees_with_euler_in_the_small_dt_limit** | | | | |
| max\|X_trapezoidal − X_euler\| / \|X\| | 1.563e-05 | <= 0.001 | 0.001 | PASS |
| **test_zero_field_ground_state_is_the_uniform_condensate** | | | | |
| min \|ψ\| | 1 | 1 | 1.000e-04 | PASS |
| max \|ψ\| | 1 | 1 | 1.000e-04 | PASS |
| max \|B\| in the relaxed state | 5.551e-17 | <= 1e-06 | 1.000e-06 | PASS |
| max \|dX/dt\| at the fixed point | 4.461e-15 | <= 0.0001 | 1.000e-04 | PASS |

### Vortices and flux quantisation

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_fluxoid_equals_enclosed_vorticity_for_any_contour** | | | | |
| max \|fluxoid − nearest integer\| | 1.776e-15 | <= 1e-09 | 1.000e-09 | PASS |
| max \|fluxoid − enclosed vorticity\| | 1.776e-15 | <= 1e-09 | 1.000e-09 | PASS |
| \|staircase fluxoid − enclosed vorticity\| | 2.665e-15 | <= 1e-09 | 1.000e-09 | PASS |
| **test_no_vortices_in_the_meissner_state** | | | | |
| vortex count | 0 | 0 | 0 | PASS |
| max \|vorticity\| anywhere | 1.657e-18 | <= 1e-09 | 1.000e-09 | PASS |
| min \|ψ\| | 0.999159 | >= 0.95 | 0.95 | PASS |
| **test_plaquette_vorticity_is_an_exact_integer** | | | | |
| plaquettes carrying vorticity | 8 | >= 1 | 1 | PASS |
| max \|vorticity − nearest integer\| | 2.474e-16 | <= 1e-10 | 1.000e-10 | PASS |
| **test_vortex_count_increases_with_the_applied_field** | | | | |
| largest decrease in count along the sweep | -4 | <= 0 | 0 | PASS |
| increase from the lowest to the highest field | 12 | >= 1 | 1 | PASS |
| count / (B·A/Φ₀) at Bz = 0.35 | 0 | <= 1 | 1 | PASS |
| count / (B·A/Φ₀) at Bz = 0.5 | 0.19635 | <= 1 | 1 | PASS |
| count / (B·A/Φ₀) at Bz = 0.7 | 0.420749 | <= 1 | 1 | PASS |
| mean interior Bz / applied at Bz = 0.35 | 0.42922 | <= 1 | 1 | PASS |
| mean interior Bz / applied at Bz = 0.5 | 0.681309 | <= 1 | 1 | PASS |
| mean interior Bz / applied at Bz = 0.7 | 0.853044 | <= 1 | 1 | PASS |
| **test_vortex_winding_sign_follows_the_applied_field[Bz=-0.5]** | | | | |
| vortices detected | 8 | >= 1 | 1 | PASS |
| distinct winding values | 1 | 1 | 0 | PASS |
| common winding | -1 | -1 | 0 | PASS |
| max \|winding\| | 1 | 1 | 0 | PASS |
| **test_vortex_winding_sign_follows_the_applied_field[Bz=0.5]** | | | | |
| vortices detected | 8 | >= 1 | 1 | PASS |
| distinct winding values | 1 | 1 | 0 | PASS |
| common winding | 1 | 1 | 0 | PASS |
| max \|winding\| | 1 | 1 | 0 | PASS |
| **test_vortices_grow_from_zero_and_saturate** | | | | |
| vortex count at t = 0 | 0 | 0 | 0 | PASS |
| final vortex count | 8 | >= 1 | 1 | PASS |
| time of first vortex entry | 10 | <= 30 | 30 | PASS |
| std/mean of the count over the final quarter | 0 | <= 0.25 | 0.25 | PASS |

### Heterostructures

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_centred_hole_is_centred[(10.0, 4.0, 0.5)]** | | | | |
| nodes carved out | 81 | >= 1 | 1 | PASS |
| hole centre x | 5 | 5 | 1.000e-12 | PASS |
| hole centre y | 5 | 5 | 1.000e-12 | PASS |
| carved width | 4 | 4 | 1.000e-12 | PASS |
| material map asymmetry under x → −x | 0 | <= 0 | 0 | PASS |
| material map asymmetry under y → −y | 0 | <= 0 | 0 | PASS |
| material map asymmetry under z → −z | 0 | <= 0 | 0 | PASS |
| **test_centred_hole_is_centred[(10.0, 4.0, 1.0)]** | | | | |
| nodes carved out | 25 | >= 1 | 1 | PASS |
| hole centre x | 5 | 5 | 1.000e-12 | PASS |
| hole centre y | 5 | 5 | 1.000e-12 | PASS |
| carved width | 4 | 4 | 1.000e-12 | PASS |
| material map asymmetry under x → −x | 0 | <= 0 | 0 | PASS |
| material map asymmetry under y → −y | 0 | <= 0 | 0 | PASS |
| material map asymmetry under z → −z | 0 | <= 0 | 0 | PASS |
| **test_centred_hole_is_centred[(12.0, 5.0, 0.5)]** | | | | |
| nodes carved out | 121 | >= 1 | 1 | PASS |
| hole centre x | 6 | 6 | 1.000e-12 | PASS |
| hole centre y | 6 | 6 | 1.000e-12 | PASS |
| carved width | 5 | 5 | 1.000e-12 | PASS |
| material map asymmetry under x → −x | 0 | <= 0 | 0 | PASS |
| material map asymmetry under y → −y | 0 | <= 0 | 0 | PASS |
| material map asymmetry under z → −z | 0 | <= 0 | 0 | PASS |
| **test_centred_hole_is_centred[(12.0, 6.0, 1.0)]** | | | | |
| nodes carved out | 49 | >= 1 | 1 | PASS |
| hole centre x | 6 | 6 | 1.000e-12 | PASS |
| hole centre y | 6 | 6 | 1.000e-12 | PASS |
| carved width | 6 | 6 | 1.000e-12 | PASS |
| material map asymmetry under x → −x | 0 | <= 0 | 0 | PASS |
| material map asymmetry under y → −y | 0 | <= 0 | 0 | PASS |
| material map asymmetry under z → −z | 0 | <= 0 | 0 | PASS |
| **test_declared_oxide_kappa_does_not_change_the_field** | | | | |
| max \|Bz(κ_ox = 0) − Bz(κ_ox = κ_SC)\| / applied | 0 | <= 1e-12 | 1.000e-12 | PASS |
| **test_insulator_kappa_is_not_the_maxwell_coefficient[kappa=0.0]** | | | | |
| Bz in the insulator / applied | 0.950271 | >= 0.5 | 0.5 | PASS |
| **test_insulator_kappa_is_not_the_maxwell_coefficient[kappa=2.0]** | | | | |
| Bz in the insulator / applied | 0.950271 | >= 0.5 | 0.5 | PASS |
| **test_insulator_mask_suppresses_the_order_parameter** | | | | |
| insulator nodes present | 60 | >= 1 | 1 | PASS |
| mean \|ψ\| in the insulator | 0.0243144 | <= 0.15 | 0.15 | PASS |
| max \|ψ\| in the superconductor | 0.965494 | >= 0.95 | 0.95 | PASS |
| mean \|ψ\| in the superconductor | 0.843018 | >= 0.75 | 0.75 | PASS |
| **test_magnetic_kappa_override_is_plaquette_centred** | | | | |
| max \|M - Mᵀ\| / \|M\| | 0 | <= 1e-12 | 1.000e-12 | PASS |
| largest eigenvalue of the symmetric part | 1.690e-13 | <= 2.56e-07 | 2.560e-07 | PASS |
| **test_stack_is_mirror_symmetric_about_its_midplane[(3, 1)]** | | | | |
| sc_mask asymmetry under z → Nz − z | 0 | <= 0 | 0 | PASS |
| κ asymmetry under z → Nz − z | 0 | <= 1e-15 | 1.000e-15 | PASS |
| superconducting nodes below vs above the mid-plane | 3 | 3 | 0 | PASS |
| **test_stack_is_mirror_symmetric_about_its_midplane[(4, 2)]** | | | | |
| sc_mask asymmetry under z → Nz − z | 0 | <= 0 | 0 | PASS |
| κ asymmetry under z → Nz − z | 0 | <= 1e-15 | 1.000e-15 | PASS |
| superconducting nodes below vs above the mid-plane | 4 | 4 | 0 | PASS |
| **test_stack_is_mirror_symmetric_about_its_midplane[(5, 2)]** | | | | |
| sc_mask asymmetry under z → Nz − z | 0 | <= 0 | 0 | PASS |
| κ asymmetry under z → Nz − z | 0 | <= 1e-15 | 1.000e-15 | PASS |
| superconducting nodes below vs above the mid-plane | 5 | 5 | 0 | PASS |
| **test_trilayer_external_z_boundary_jn** | | | | |
| max \|J_z\| on the bottom face | 1.095e-21 | <= 1e-12 | 1.000e-12 | PASS |
| max \|J_z\| on the top face | 2.121e-21 | <= 1e-12 | 1.000e-12 | PASS |
| **test_trilayer_kappa_discontinuity** | | | | |
| LPHI_x diagonal in the superconductor | -16 | -16 | 1.000e-12 | PASS |
| LPHI_x diagonal in the insulator | -16 | -16 | 1.000e-12 | PASS |
| **test_trilayer_superconducting_layers_screen** | | | | |
| Bz in the bottom Nb layer / applied | 0.852883 | <= 0.98 | 0.98 | PASS |
| Bz in the top Nb layer / applied | 0.861574 | <= 0.98 | 0.98 | PASS |
| Bz in the bottom Nb layer / applied | 0.852883 | >= 0 | 0 | PASS |
| top/bottom screening asymmetry | 1.01019 | 1 | 0.15 | PASS |

### Flux expulsion by a ring

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_a_larger_hole_expels_less** | | | | |
| both scans bracket a threshold | 1 | >= 1 | 1 | PASS |
| B_exp(6 ξ hole) / B_exp(4 ξ hole) | 0.685185 | <= 0.9 | 0.9 | PASS |
| B_exp · A_hole / Φ₀ for the 4 ξ hole | 0.687549 | [0.2, 3] | [0.2, 3] | PASS |
| B_exp · A_hole / Φ₀ for the 6 ξ hole | 1.05997 | [0.2, 3] | [0.2, 3] | PASS |
| **test_expulsion_threshold_is_bracketed** | | | | |
| scan brackets the threshold | 1 | >= 1 | 1 | PASS |
| expulsion field B_exp | 0.27 | [0.05, 0.9] | [0.05, 0.9] | PASS |
| B_exp · A_hole / Φ₀ | 0.687549 | [0.2, 3] | [0.2, 3] | PASS |
| fields that admitted flux | 3 | >= 3 | 3 | PASS |
| largest increase in entry time with field | -1.99688 | <= 0 | 0 | PASS |
| **test_flux_enters_in_whole_quanta** | | | | |
| fluxoid at t = 0 | 0 | 0 | 1.000e-09 | PASS |
| final \|fluxoid\| | 4 | >= 1 | 1 | PASS |
| max \|fluxoid − nearest integer\| | 8.882e-16 | <= 1e-09 | 1.000e-09 | PASS |
| largest decrease along the history | 1.332e-15 | <= 1e-09 | 1.000e-09 | PASS |
| **test_fluxoid_does_not_depend_on_the_contour** | | | | |
| \|fluxoid\| (non-trivial, so the check has content) | 4 | >= 0.5 | 0.5 | PASS |
| spread across contour margins | 2.220e-15 | <= 1e-09 | 1.000e-09 | PASS |
| \|fluxoid − nearest integer\| | 8.882e-16 | <= 1e-09 | 1.000e-09 | PASS |
| **test_ring_expels_flux_below_threshold** | | | | |
| max \|fluxoid\| over the whole run | 1.877e-17 | <= 1e-09 | 1.000e-09 | PASS |
| max \|dX/dt\| at the end | 6.090e-14 | <= 0.001 | 0.001 | PASS |
| **test_the_relaxed_ring_is_symmetric** | | | | |
| \|ψ\| scale (non-trivial) | 0.928977 | >= 0.5 | 0.5 | PASS |
| Bz scale (non-trivial) | 0.0482529 | >= 0.001 | 0.001 | PASS |
| max \|ψ\| asymmetry under x → −x | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max \|ψ\| asymmetry under y → −y | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max \|ψ\| asymmetry under z → −z | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max \|ψ\| asymmetry under 90° rotation | 1.110e-16 | <= 1e-12 | 1.000e-12 | PASS |
| max Bz asymmetry under x → −x | 2.776e-17 | <= 4.83e-14 | 4.825e-14 | PASS |
| max Bz asymmetry under y → −y | 2.776e-17 | <= 4.83e-14 | 4.825e-14 | PASS |
| max Bz asymmetry under z → −z | 2.776e-17 | <= 4.83e-14 | 4.825e-14 | PASS |
| max Bz asymmetry under 90° rotation | 2.082e-17 | <= 4.83e-14 | 4.825e-14 | PASS |
| **test_the_ring_is_superconducting** | | | | |
| max \|ψ\| in the superconducting layers | 0.928977 | >= 0.9 | 0.9 | PASS |
| mean \|ψ\| in the superconducting layers | 0.660557 | >= 0.5 | 0.5 | PASS |
| max \|ψ\| in the oxide and the hole | 0.103677 | <= 0.25 | 0.25 | PASS |

### Other

| Check | Measured | Expected | Tolerance | Status |
|-------|----------|----------|-----------|--------|
| **test_empty_box_reproduces_the_applied_field[Bx]** | | | | |
| least-squares residual of the steady state | 2.845e-14 | <= 1e-09 | 1.000e-09 | PASS |
| max \|Bx - applied\| / applied | 1.058e-14 | <= 1e-12 | 1.000e-12 | PASS |
| **test_empty_box_reproduces_the_applied_field[By]** | | | | |
| least-squares residual of the steady state | 2.116e-14 | <= 1e-09 | 1.000e-09 | PASS |
| max \|By - applied\| / applied | 1.093e-14 | <= 1e-12 | 1.000e-12 | PASS |
| **test_empty_box_reproduces_the_applied_field[Bz]** | | | | |
| least-squares residual of the steady state | 2.776e-14 | <= 1e-09 | 1.000e-09 | PASS |
| max \|Bz - applied\| / applied | 1.232e-14 | <= 1e-12 | 1.000e-12 | PASS |
| **test_far_field_converges** | | | | |
| far-field error at 4 cells / error at 2 cells | 0.450993 | <= 1 | 1 | PASS |
| far-field error at 8 cells / error at 4 cells | 0.250791 | <= 1 | 1 | PASS |
| **test_flux_crowds_into_the_vacuum_beside_the_film** | | | | |
| max \|dX/dt\| at the final state | 9.408e-09 | <= 0.001 | 0.001 | PASS |
| peak Bz in the vacuum beside the film / applied | 1.01696 | >= 1 | 1.005 | PASS |
| Bz at the film centre / applied | 0.806862 | <= 0.9 | 0.9 | PASS |
| **test_kappa_contrast_without_current_changes_nothing** | | | | |
| max \|Bz - applied\| / applied, declared κ = 0.0 | 9.714e-15 | <= 1e-12 | 1.000e-12 | PASS |
| max \|Bz - applied\| / applied, declared κ = 1.0 | 9.714e-15 | <= 1e-12 | 1.000e-12 | PASS |
| max \|Bz - applied\| / applied, declared κ = 4.0 | 9.714e-15 | <= 1e-12 | 1.000e-12 | PASS |
| **test_lateral_vacuum_unpins_the_film_edge** | | | | |
| max Bz over metal nodes / applied, no lateral margin | 1 | 1 | 1.000e-09 | PASS |
| max Bz over metal nodes / applied, with a lateral margin | 0.987366 | <= 0.999 | 0.999 | PASS |
| **test_padded_stack_is_mirror_symmetric** | | | | |
| max \|Bz(z) - Bz(-z)\| / applied | 2.220e-16 | <= 1e-12 | 1.000e-12 | PASS |

## Test details

### test_a_larger_hole_expels_less

_the expulsion field is set by the hole area at fixed arm width_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** holes=[4, 6], arm=3, kappa=2, t_hold=30
- **PASS** both scans bracket a threshold: measured 1, expected >= 1 — must be at least 1
- **PASS** B_exp(6 ξ hole) / B_exp(4 ξ hole): measured 0.685185, expected <= 0.9 — a larger hole gathers more flux per unit field, so it gives way sooner
- **PASS** B_exp · A_hole / Φ₀ for the 4 ξ hole: measured 0.687549, expected [0.2, 3] — the threshold sits within a factor of a few of one flux quantum
- **PASS** B_exp · A_hole / Φ₀ for the 6 ξ hole: measured 1.05997, expected [0.2, 3] — the threshold sits within a factor of a few of one flux quantum
- **Diagnostics:**
  - `B_exp_hole4`: 0.27
  - `B_exp_hole6`: 0.185
  - `summary_hole4`: B_exp = 0.2700 ± 0.0500 (hold time 30)
  - `summary_hole6`: B_exp = 0.1850 ± 0.0350 (hold time 30)

### test_applied_field_vectors_are_uniform_on_each_face

_a uniform applied field must be uniform on every face it is imposed on_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=7, Ny=6, Nz=5, Bx=0.3, By=-0.2, Bz=1
- **PASS** max deviation of Bz on x_lo: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of Bz on x_hi: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of Bz on y_lo: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of Bz on y_hi: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of Bx on y_lo: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of Bx on z_hi: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of By on x_hi: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max deviation of By on z_lo: measured 0, expected <= 1e-14 — must not exceed 1e-14

### test_applied_flux_on_boundary_plaquettes[8x8]

_the boundary condition must impose B_applied on each boundary plaquette exactly once_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=8, h=0.5, Bz=0.12
- **PASS** flux on the hi/hi corner plaquette: measured 0.12, expected 0.12 — must be B_applied, not 2 B_applied
- **PASS** max deviation of boundary ring from B_applied: measured 2.776e-17, expected <= 1.2e-11 — must not exceed 1.2e-11
- **PASS** state drift once relaxed: measured 8.954e-11, expected <= 1e-08 — an over-counted corner drives an unbounded drift of the corner links
- **PASS** screened interior field / applied: measured 0.941456, expected <= 0.99 — the interior must be screened below the applied field
- **Diagnostics:**
  - `corner_hi_hi`: 0.12
  - `boundary_ring_min`: 0.12
  - `boundary_ring_max`: 0.12
  - `interior_min`: 0.0981056

### test_applied_flux_on_boundary_plaquettes[9x7]

_the boundary condition must impose B_applied on each boundary plaquette exactly once_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=7, h=0.5, Bz=0.12
- **PASS** flux on the hi/hi corner plaquette: measured 0.12, expected 0.12 — must be B_applied, not 2 B_applied
- **PASS** max deviation of boundary ring from B_applied: measured 2.776e-17, expected <= 1.2e-11 — must not exceed 1.2e-11
- **PASS** state drift once relaxed: measured 8.286e-11, expected <= 1e-08 — an over-counted corner drives an unbounded drift of the corner links
- **PASS** screened interior field / applied: measured 0.942774, expected <= 0.99 — the interior must be screened below the applied field
- **Diagnostics:**
  - `corner_hi_hi`: 0.12
  - `boundary_ring_min`: 0.12
  - `boundary_ring_max`: 0.12
  - `interior_min`: 0.0986514

### test_bfield_evaluators_agree[5x7x6]

_one curl stencil, one answer, on any grid shape_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=5, Ny=7, Nz=6
- **PASS** max|eval_bfield(subset) − reference|: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max|eval_bfield(all interior) − reference|: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** len(bfield_interior): measured 60, expected 60 — |measured - expected| <= 0
- **Diagnostics:**
  - `b_scale`: 1.48934

### test_bfield_evaluators_agree[6x6x1]

_one curl stencil, one answer, on any grid shape_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=6, Nz=1
- **PASS** max|eval_bfield(subset) − reference|: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max|eval_bfield(all interior) − reference|: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** len(bfield_interior): measured 16, expected 16 — |measured - expected| <= 0
- **Diagnostics:**
  - `b_scale`: 1.10429

### test_bfield_evaluators_agree[9x5x1]

_one curl stencil, one answer, on any grid shape_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=5, Nz=1
- **PASS** max|eval_bfield(subset) − reference|: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** max|eval_bfield(all interior) − reference|: measured 0, expected <= 1e-14 — must not exceed 1e-14
- **PASS** len(bfield_interior): measured 21, expected 21 — |measured - expected| <= 0
- **Diagnostics:**
  - `b_scale`: 1.3361

### test_bfield_matches_the_exact_london_solution

_the screened field profile matches the closed-form London solution_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** length=16, kappa=2, Bz=0.02, h_values=[1, 0.5, 0.25]
- **PASS** min |ψ| at the coarsest h (London limit is applicable): measured 0.999627, expected >= 0.99 — the model assumes |ψ| = 1; the check is void if ψ is suppressed
- **PASS** field at the centre / B₀: measured 0.0702701, expected <= 0.2 — the sample really is screening, so the comparison has content
- **PASS** |B(pinned plaquette) − B_applied| / B₀: measured 5.551e-16, expected <= 1e-12 — a Dirichlet condition: the solver is exact here, model aside
- **PASS** rms |solver − model| / B₀ at h = 1 ξ: measured 0.00408256, expected <= 0.005 — must not exceed 0.005
- **PASS** rms |solver − model| / B₀ at h = 0.25 ξ: measured 3.290e-04, expected <= 0.0005 — must not exceed 0.0005
- **PASS** max |solver − model| / B₀ at h = 0.25 ξ: measured 4.463e-04, expected <= 0.001 — must not exceed 0.001
- **PASS** observed order in h (whole profile): measured 1.81668, expected >= 1.7 — mixes the second-order bulk with the series' Gibbs floor
- **PASS** observed order in h (bulk, >1 ξ from an edge): measured 1.84113, expected >= 1.75 — the curl-curl operator's own discretisation error
- **Diagnostics:**
  - `rms_error_over_B0`: 1.0=0.00408256, 0.5=0.00105244, 0.25=3.290e-04
  - `rms_bulk_error_over_B0`: 1.0=0.00429348, 0.5=0.00109372, 0.25=3.345e-04
  - `max_error_over_B0`: 1.0=0.00490852, 0.5=0.00132821, 0.25=4.463e-04
  - `profile_over_B0`: 1.0=[0.61943, 0.39059, 0.253843, 0.172957, 0.126159, 0.10068, 0.0894936, 0.0894936, … (15 values)], 0.5=[0.783351, 0.615293, 0.48506, 0.384263, 0.306361, 0.246264, 0.200011, 0.164535, … (31 values)], 0.25=[0.884435, 0.782637, 0.692983, 0.614041, 0.544549, 0.483392, 0.429587, 0.382264, … (63 values)]

### test_c4_symmetry_of_a_square_device

_a square sample in a uniform out-of-plane field is invariant under 90° rotation_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=10, Ny=10, h=0.5, Bz=0.15
- **PASS** Bz contrast (screening present, so the test is non-trivial): measured 0.0322938, expected >= 0.001 — must be at least 0.001
- **PASS** max|ψ| − R₉₀|ψ||: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|Bz − R₉₀Bz|: measured 5.551e-17, expected <= 1e-12 — must not exceed 1e-12
- **Diagnostics:**
  - `psi_contrast`: 0.00476886
  - `bz_contrast`: 0.0322938

### test_centred_hole_is_centred[(10.0, 4.0, 0.5)]

_the carved geometry must inherit the symmetry of the polygon it was given_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** length=10, hole=4, h=0.5
- **PASS** nodes carved out: measured 81, expected >= 1 — must be at least 1
- **PASS** hole centre x: measured 5, expected 5 — |measured - expected| <= 1e-12
- **PASS** hole centre y: measured 5, expected 5 — |measured - expected| <= 1e-12
- **PASS** carved width: measured 4, expected 4 — |measured - expected| <= 1e-12
- **PASS** material map asymmetry under x → −x: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under y → −y: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under z → −z: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `carved_nodes`: 81
  - `hole_centre`: [5, 5]
  - `hole_width`: 4

### test_centred_hole_is_centred[(10.0, 4.0, 1.0)]

_the carved geometry must inherit the symmetry of the polygon it was given_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** length=10, hole=4, h=1
- **PASS** nodes carved out: measured 25, expected >= 1 — must be at least 1
- **PASS** hole centre x: measured 5, expected 5 — |measured - expected| <= 1e-12
- **PASS** hole centre y: measured 5, expected 5 — |measured - expected| <= 1e-12
- **PASS** carved width: measured 4, expected 4 — |measured - expected| <= 1e-12
- **PASS** material map asymmetry under x → −x: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under y → −y: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under z → −z: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `carved_nodes`: 25
  - `hole_centre`: [5, 5]
  - `hole_width`: 4

### test_centred_hole_is_centred[(12.0, 5.0, 0.5)]

_the carved geometry must inherit the symmetry of the polygon it was given_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** length=12, hole=5, h=0.5
- **PASS** nodes carved out: measured 121, expected >= 1 — must be at least 1
- **PASS** hole centre x: measured 6, expected 6 — |measured - expected| <= 1e-12
- **PASS** hole centre y: measured 6, expected 6 — |measured - expected| <= 1e-12
- **PASS** carved width: measured 5, expected 5 — |measured - expected| <= 1e-12
- **PASS** material map asymmetry under x → −x: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under y → −y: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under z → −z: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `carved_nodes`: 121
  - `hole_centre`: [6, 6]
  - `hole_width`: 5

### test_centred_hole_is_centred[(12.0, 6.0, 1.0)]

_the carved geometry must inherit the symmetry of the polygon it was given_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** length=12, hole=6, h=1
- **PASS** nodes carved out: measured 49, expected >= 1 — must be at least 1
- **PASS** hole centre x: measured 6, expected 6 — |measured - expected| <= 1e-12
- **PASS** hole centre y: measured 6, expected 6 — |measured - expected| <= 1e-12
- **PASS** carved width: measured 6, expected 6 — |measured - expected| <= 1e-12
- **PASS** material map asymmetry under x → −x: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under y → −y: measured 0, expected <= 0 — must not exceed 0
- **PASS** material map asymmetry under z → −z: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `carved_nodes`: 49
  - `hole_centre`: [6, 6]
  - `hole_width`: 6

### test_covariant_laplacian_is_second_order_accurate

_the discrete Laplacian must converge at second order_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** k=[0.7, 0.4], h_values=[0.4, 0.2, 0.1]
- **PASS** observed order of accuracy: measured 1.99835, expected 2 — |measured - expected| <= 0.15
- **PASS** error at h = 0.1: measured 2.214e-04, expected <= 0.001 — must not exceed 0.001
- **Diagnostics:**
  - `errors`: 0.1=2.214e-04, 0.2=8.851e-04, 0.4=0.00353402

### test_covariant_laplacian_reduces_to_the_standard_laplacian

_the Peierls factors must be the only difference from the plain Laplacian_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=7, hx=0.4, hy=0.6
- **PASS** max|L_covariant(A=0) − L_standard|: measured 0, expected <= 1e-13 — must not exceed 1e-13
- **Diagnostics:**
  - `operator_scale`: 18.0556

### test_curl_curl_operator_is_divergence_free[7x6x6]

_the discrete curl-curl operator must annihilate gradients exactly_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=7, Ny=6, Nz=6
- **PASS** max|∇·(∇×∇×A)| / scale: measured 3.243e-16, expected <= 1e-13 — must not exceed 1e-13
- **Diagnostics:**
  - `operator_scale`: 10.9543

### test_curl_curl_operator_is_divergence_free[9x8x1]

_the discrete curl-curl operator must annihilate gradients exactly_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=8, Nz=1
- **PASS** max|∇·(∇×∇×A)| / scale: measured 2.918e-16, expected <= 1e-13 — must not exceed 1e-13
- **Diagnostics:**
  - `operator_scale`: 6.08859

### test_declared_oxide_kappa_does_not_change_the_field

_the declared oxide κ is inert; only Layer.magnetic_kappa can change the field_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** kappa_sc=2, Bz=0.1
- **PASS** max |Bz(κ_ox = 0) − Bz(κ_ox = κ_SC)| / applied: measured 0, expected <= 1e-12 — both resolve to the same vacuum Maxwell coefficient
- **Diagnostics:**
  - `profile_kappa_zero`: [0.920444, 0.931527, 0.948655, 0.963995, 0.972603, 0.975334, 0.972603, 0.963995, … (11 values)]
  - `profile_kappa_matched`: [0.920444, 0.931527, 0.948655, 0.963995, 0.972603, 0.975334, 0.972603, 0.963995, … (11 values)]

### test_divergence_of_discrete_curl_is_exactly_zero[6x7x8]

_∇·B must vanish identically, not merely to discretisation order_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=7, Nz=8
- **PASS** bulk nodes tested: measured 120, expected >= 1 — must be at least 1
- **PASS** max|∇·B| / max|B|: measured 3.204e-16, expected <= 1e-13 — forward-difference divergence of the forward plaquette curl
- **Diagnostics:**
  - `n_bulk_nodes`: 120
  - `B_scale`: 2.07885

### test_divergence_of_discrete_curl_is_exactly_zero[8x7x1]

_∇·B must vanish identically, not merely to discretisation order_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=7, Nz=1
- **PASS** bulk nodes tested: measured 30, expected >= 1 — must be at least 1
- **PASS** max|∇·B| / max|B|: measured 0, expected <= 1e-13 — forward-difference divergence of the forward plaquette curl
- **Diagnostics:**
  - `n_bulk_nodes`: 30
  - `B_scale`: 1.6196

### test_empty_box_reproduces_the_applied_field[Bx]

_the applied-field boundary condition is exact in vacuum_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=8, Nz=8, applied=0.02, component=Bx
- **PASS** least-squares residual of the steady state: measured 2.845e-14, expected <= 1e-09 — a non-zero residual would mean no steady state exists
- **PASS** max |Bx - applied| / applied: measured 1.058e-14, expected <= 1e-12 — vacuum carries the applied field unchanged
- **Diagnostics:**
  - `steady_state_residual`: 2.845e-14

### test_empty_box_reproduces_the_applied_field[By]

_the applied-field boundary condition is exact in vacuum_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=8, Nz=8, applied=0.02, component=By
- **PASS** least-squares residual of the steady state: measured 2.116e-14, expected <= 1e-09 — a non-zero residual would mean no steady state exists
- **PASS** max |By - applied| / applied: measured 1.093e-14, expected <= 1e-12 — vacuum carries the applied field unchanged
- **Diagnostics:**
  - `steady_state_residual`: 2.116e-14

### test_empty_box_reproduces_the_applied_field[Bz]

_the applied-field boundary condition is exact in vacuum_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=8, Nz=8, applied=0.02, component=Bz
- **PASS** least-squares residual of the steady state: measured 2.776e-14, expected <= 1e-09 — a non-zero residual would mean no steady state exists
- **PASS** max |Bz - applied| / applied: measured 1.232e-14, expected <= 1e-12 — vacuum carries the applied field unchanged
- **Diagnostics:**
  - `steady_state_residual`: 2.776e-14

### test_expulsion_threshold_is_bracketed

_the ring expels flux up to a definite field and admits quanta above it_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** hole=4, arm=3, kappa=2, h=1, t_hold=30
- **PASS** scan brackets the threshold: measured 1, expected >= 1 — must be at least 1
- **PASS** expulsion field B_exp: measured 0.27, expected [0.05, 0.9] — above the lowest field scanned and below H_c2 = 1
- **PASS** B_exp · A_hole / Φ₀: measured 0.687549, expected [0.2, 3] — the threshold is set by the fluxoid scale, not by the grid
- **PASS** fields that admitted flux: measured 3, expected >= 3 — must be at least 3
- **PASS** largest increase in entry time with field: measured -1.99688, expected <= 0 — entry times [13.978124999999885, 5.990625000000023, 3.99375000000001] must fall as the field rises
- **Diagnostics:**
  - `fields`: [0.05, 0.15, 0.22, 0.32, 0.45, 0.6]
  - `final_fluxoid`: [1.104e-18, 1.325e-17, -3.534e-17, 4, 4, 4]
  - `entry_times`: [—, —, —, 13.9781, 5.99063, 3.99375]
  - `summary`: B_exp = 0.2700 ± 0.0500 (hold time 30)

### test_far_field_converges

_the far-field boundary condition converges as the vacuum grows_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** paddings=[2, 4, 8], kappa=2, applied_Bz=0.02, Nx=16, lateral_margin=3
- **PASS** far-field error at 4 cells / error at 2 cells: measured 0.450993, expected <= 1 — doubling the padding must not make the far field worse
- **PASS** far-field error at 8 cells / error at 4 cells: measured 0.250791, expected <= 1 — must not exceed 1
- **Diagnostics:**
  - `far_field_error_by_padding`: 2=0.199485, 4=0.0899665, 8=0.0225628
  - `max_abs_dXdt`: 2=5.655e-09, 4=8.246e-09, 8=1.239e-08

### test_field_reversal_flips_b_and_preserves_psi

_the GL equations are invariant under B → −B combined with ψ → ψ*_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=7, Bz=0.4, n_steps=60
- **PASS** B scale (non-trivial state): measured 0.334784, expected >= 0.001 — must be at least 0.001
- **PASS** max|Bz(+B) + Bz(−B)|: measured 0, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max| |ψ(+B)| − |ψ(−B)| |: measured 0, expected <= 1e-12 — must not exceed 1e-12
- **Diagnostics:**
  - `B_scale`: 0.334784

### test_flux_crowds_into_the_vacuum_beside_the_film

_expelled flux crowds into the vacuum beside the film_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=16, Nz=18, kappa=2, applied_Bz=0.02, lateral_margin=3, vacuum_cells=5
- **PASS** max |dX/dt| at the final state: measured 9.408e-09, expected <= 0.001 — the line below is read at a steady state
- **PASS** peak Bz in the vacuum beside the film / applied: measured 1.01696, expected >= 1 — flux pushed out of the film has to go somewhere
- **PASS** Bz at the film centre / applied: measured 0.806862, expected <= 0.9 — and it came from here
- **Diagnostics:**
  - `bz_line_over_applied`: [1.00668, 1.01696, 0.987366, 0.924368, 0.865433, 0.825874, 0.806862, 0.806862, … (15 values)]
  - `max_abs_dXdt`: 9.408e-09

### test_flux_enters_in_whole_quanta

_fluxoid quantisation holds instant by instant, including mid-entry_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** hole=4, Bz=0.6, t_hold=30
- **PASS** fluxoid at t = 0: measured 0, expected 0 — |measured - expected| <= 1e-09
- **PASS** final |fluxoid|: measured 4, expected >= 1 — must be at least 1
- **PASS** max |fluxoid − nearest integer|: measured 8.882e-16, expected <= 1e-09 — must not exceed 1e-09
- **PASS** largest decrease along the history: measured 1.332e-15, expected <= 1e-09 — flux accumulates; it does not leak back out at fixed field
- **Diagnostics:**
  - `fluxoid_history`: [0, -7.068e-17, 4, 4, 4, 4, 4, 4, … (17 values)]
  - `entry_time`: 3.99375

### test_fluxoid_does_not_depend_on_the_contour

_the fluxoid counts what the contour encloses, not how it is drawn_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** hole=4, Bz=0.45, margins=[1, 1.5, 2]
- **PASS** |fluxoid| (non-trivial, so the check has content): measured 4, expected >= 0.5 — must be at least 0.5
- **PASS** spread across contour margins: measured 2.220e-15, expected <= 1e-09 — must not exceed 1e-09
- **PASS** |fluxoid − nearest integer|: measured 8.882e-16, expected <= 1e-09 — must not exceed 1e-09
- **Diagnostics:**
  - `fluxoid_by_margin`: 1.0=4, 1.5=4, 2.0=4

### test_fluxoid_equals_enclosed_vorticity_for_any_contour

_the fluxoid is a topological invariant of the region, not of the path_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=20, kappa=2, Bz=0.5
- **PASS** max |fluxoid − nearest integer|: measured 1.776e-15, expected <= 1e-09 — must not exceed 1e-09
- **PASS** max |fluxoid − enclosed vorticity|: measured 1.776e-15, expected <= 1e-09 — must not exceed 1e-09
- **PASS** |staircase fluxoid − enclosed vorticity|: measured 2.665e-15, expected <= 1e-09 — must not exceed 1e-09
- **Diagnostics:**
  - `contours`: square_pad2=fluxoid=8, enclosed_vorticity=8, square_pad4=fluxoid=8, enclosed_vorticity=8, square_pad6=fluxoid=-3.534e-17, enclosed_vorticity=0
  - `staircase_fluxoid`: 6
  - `staircase_enclosed`: 6

### test_forward_euler_is_first_order_in_dt

_explicit Euler must show first-order global convergence_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=7, t_stop=0.5, dt_max=0.0078125
- **PASS** observed order in dt: measured 1.00371, expected 1 — |measured - expected| <= 0.1
- **PASS** Richardson error at the smallest dt: measured 1.854e-05, expected <= 0.001 — must not exceed 0.001
- **Diagnostics:**
  - `richardson_errors`: 9.766e-04=1.854e-05, 1.953e-03=3.713e-05, 3.906e-03=7.441e-05, 7.812e-03=1.495e-04

### test_forward_euler_is_stable_below_the_cfl_limit[2d]

_the explicit step size limit is set by the κ²∇×∇× term and depends on dimension_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=6, Nz=1, h=0.5, kappa=2, cfl_limit=0.015625
- **PASS** max|ψ|² at dt = 0.9 dt_CFL: measured 1, expected 1 — relaxes to the uniform state without amplification
- **PASS** min|ψ|² at dt = 0.9 dt_CFL: measured 1, expected 1 — |measured - expected| <= 0.05
- **PASS** run at dt = 4 dt_CFL loses the superconducting state: measured 1, expected >= 1 — must be at least 1
- **Diagnostics:**
  - `cfl_limit`: 0.015625
  - `max_psi2_stable`: 1
  - `min_psi2_stable`: 1
  - `unstable_run_diverged`: True

### test_forward_euler_is_stable_below_the_cfl_limit[3d]

_the explicit step size limit is set by the κ²∇×∇× term and depends on dimension_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=6, Nz=6, h=0.5, kappa=2, cfl_limit=0.0078125
- **PASS** max|ψ|² at dt = 0.9 dt_CFL: measured 1.00005, expected 1 — relaxes to the uniform state without amplification
- **PASS** min|ψ|² at dt = 0.9 dt_CFL: measured 1.00002, expected 1 — |measured - expected| <= 0.05
- **PASS** run at dt = 4 dt_CFL loses the superconducting state: measured 1, expected >= 1 — must be at least 1
- **Diagnostics:**
  - `cfl_limit`: 0.0078125
  - `max_psi2_stable`: 1.00005
  - `min_psi2_stable`: 1.00002
  - `unstable_run_diverged`: True

### test_free_energy_decreases_monotonically_at_zero_field

_TDGL is a gradient flow of the GL free energy, so F(t) is non-increasing_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=12, Ny=11, h=0.5, kappa=2, dt=0.007812, n_steps=400
- **PASS** energy released (test would be vacuous otherwise): measured 44.1063, expected >= 1 — must be at least 1
- **PASS** steps on which F increased: measured 0, expected <= 0 — out of 400 steps
- **PASS** worst single-step ΔF / energy released: measured -9.570e-10, expected <= 1e-09 — must not exceed 1e-09
- **Diagnostics:**
  - `F_initial`: 30.3563
  - `F_final`: -13.75
  - `energy_released`: 44.1063
  - `n_steps_increasing`: 0

### test_free_energy_decreases_while_relaxing_in_a_field

_relaxation at fixed applied field lowers the free energy_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=12, h=0.5, kappa=2, Bz=0.15
- **PASS** energy released: measured 53.0219, expected >= 0.5 — must be at least 0.5
- **PASS** worst single-step ΔF / energy released: measured -1.320e-08, expected <= 1e-06 — must not exceed 1e-06
- **Diagnostics:**
  - `F_initial`: 40.1134
  - `F_final`: -12.9085
  - `energy_released`: 53.0219

### test_global_phase_rotation_is_exact_symmetry

_global U(1) symmetry holds even with an applied field on the boundary_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, alpha=0.7, Bz=0.4
- **PASS** max|dψ/dt rotation error|: measured 7.022e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|dφ/dt change|: measured 4.441e-16, expected <= 1e-12 — must not exceed 1e-12

### test_indices_are_within_bounds_on_ragged_grids

_index arrays must stay in range for every grid aspect ratio_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** grids=4x9x3, 9x4x3, 3x4x9, 5x5x2
- **PASS** index arrays checked: measured 67, expected >= 40 — must be at least 40
- **PASS** worst overshoot past the last valid index: measured -1, expected <= -1 — must not exceed -1
- **Diagnostics:**
  - `index_arrays_checked`: 67

### test_insulator_kappa_is_not_the_maxwell_coefficient[kappa=0.0]

_a declared oxide κ, zero included, does not change what the oxide transmits_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nz=12, kappa_sc=2, kappa_insulator_declared=0, Bz=0.1
- **PASS** Bz in the insulator / applied: measured 0.950271, expected >= 0.5 — ψ = 0 means no screening current, so the oxide lets the field through
- **Diagnostics:**
  - `bz_profile_over_applied`: [0.894392, 0.906366, 0.925004, 0.942021, 0.951946, 0.95517, 0.951946, 0.942021, … (11 values)]
  - `insulator_mean_over_applied`: 0.950271

### test_insulator_kappa_is_not_the_maxwell_coefficient[kappa=2.0]

_a declared oxide κ, zero included, does not change what the oxide transmits_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nz=12, kappa_sc=2, kappa_insulator_declared=2, Bz=0.1
- **PASS** Bz in the insulator / applied: measured 0.950271, expected >= 0.5 — ψ = 0 means no screening current, so the oxide lets the field through
- **Diagnostics:**
  - `bz_profile_over_applied`: [0.894392, 0.906366, 0.925004, 0.942021, 0.951946, 0.95517, 0.951946, 0.942021, … (11 values)]
  - `insulator_mean_over_applied`: 0.950271

### test_insulator_mask_suppresses_the_order_parameter

_the material mask must separate superconducting from insulating nodes_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=4, Ny=5, Nz=12, sc_thickness=4
- **PASS** insulator nodes present: measured 60, expected >= 1 — must be at least 1
- **PASS** mean |ψ| in the insulator: measured 0.0243144, expected <= 0.15 — residual value is proximity leakage from the adjacent layers
- **PASS** max |ψ| in the superconductor: measured 0.965494, expected >= 0.95 — the middle of a 4-cell layer must recover the bulk condensate
- **PASS** mean |ψ| in the superconductor: measured 0.843018, expected >= 0.75 — must be at least 0.75
- **Diagnostics:**
  - `n_insulator_nodes`: 60
  - `psi_z_profile`: [0.965494, 0.900012, 0.663547, 0.0556877, 0.00470601, 7.843e-04, 0.00470601, 0.0556877, … (11 values)]
  - `sc_mask_z_profile`: [1, 1, 1, 0, 0, 0, 0, 0, … (11 values)]

### test_insulator_order_parameter_decays_with_the_stated_time_constant

_the insulator relaxation term must act on its documented time scale_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nz=6, tau_expected=0.1
- **PASS** points used in the fit: measured 30, expected >= 5 — must be at least 5
- **PASS** fitted τ: measured 0.0929158, expected 0.1 — |measured - expected| <= 0.02
- **PASS** residual |ψ| in the insulator: measured 0.0532326, expected <= 0.15 — proximity leakage from the neighbouring superconductors
- **Diagnostics:**
  - `tau_fit`: 0.0929158
  - `psi_steady_state`: 0.0532326

### test_interior_numbering_matches_documented_strides[5x7x6]

_interior arrays are C-ordered over (Nx-1, Ny-1, Nz-1)_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=5, Ny=7, Nz=6
- **PASS** mismatched entries of interior_to_full: measured 0, expected <= 0 — must not exceed 0
- **PASS** reshape stride mismatch: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `strides`: [30, 5, 1]

### test_interior_numbering_matches_documented_strides[6x6x1]

_interior arrays are C-ordered over (Nx-1, Ny-1, Nz-1)_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=6, Nz=1
- **PASS** mismatched entries of interior_to_full: measured 0, expected <= 0 — must not exceed 0
- **PASS** reshape stride mismatch: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `strides`: [5, 1, 1]

### test_interior_numbering_matches_documented_strides[9x5x1]

_interior arrays are C-ordered over (Nx-1, Ny-1, Nz-1)_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=5, Nz=1
- **PASS** mismatched entries of interior_to_full: measured 0, expected <= 0 — must not exceed 0
- **PASS** reshape stride mismatch: measured 0, expected <= 0 — must not exceed 0
- **Diagnostics:**
  - `strides`: [4, 1, 1]

### test_kappa_contrast_without_current_changes_nothing

_the Maxwell coefficient is a property of the vacuum, not of the material_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=8, Nz=12, kappa_ref=2, applied_Bz=0.02, slab_z_nodes=5-7
- **PASS** max |Bz - applied| / applied, declared κ = 0.0: measured 9.714e-15, expected <= 1e-12 — no current anywhere, so the field must stay uniform
- **PASS** max |Bz - applied| / applied, declared κ = 1.0: measured 9.714e-15, expected <= 1e-12 — no current anywhere, so the field must stay uniform
- **PASS** max |Bz - applied| / applied, declared κ = 4.0: measured 9.714e-15, expected <= 1e-12 — no current anywhere, so the field must stay uniform
- **Diagnostics:**
  - `max_relative_error_by_declared_kappa`: 0.0=9.714e-15, 1.0=9.714e-15, 4.0=9.714e-15

### test_lateral_vacuum_unpins_the_film_edge

_the applied-field condition must land on vacuum, not on the metal_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=16, kappa=2, applied_Bz=0.02, lateral_margin_bare=0, lateral_margin_padded=3
- **PASS** max Bz over metal nodes / applied, no lateral margin: measured 1, expected 1 — prescribed, not solved for — this is the artefact
- **PASS** max Bz over metal nodes / applied, with a lateral margin: measured 0.987366, expected <= 0.999 — every metal node is now screened below the applied field
- **Diagnostics:**
  - `bz_line_no_margin`: [0.742088, 0.581467, 0.477082, 0.40824, 0.363647, 0.336855, 0.324272, 0.324272, … (15 values)]
  - `bz_line_with_margin`: [1.00668, 1.01696, 0.987366, 0.924368, 0.865433, 0.825874, 0.806862, 0.806862, … (15 values)]

### test_london_penetration_depth_equals_kappa[kappa=1.5]

_λ = κ in these units — the field decays as exp(-x/κ) into the bulk_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=24, h=0.5, kappa=1.5, Bz=0.02, L_over_lambda=8
- **PASS** state drift (equilibrium reached): measured 3.696e-08, expected <= 1e-06 — must not exceed 1e-06
- **PASS** min |ψ| (still Meissner, no vortices): measured 0.999794, expected >= 0.9 — must be at least 0.9
- **PASS** λ from the screening profile: measured 1.61931, expected 1.5 — London penetration depth must equal the GL parameter κ
- **Diagnostics:**
  - `lambda_fit`: 1.61931
  - `profile`: [0.0144655, 0.0105224, 0.00771877, 0.00573052, 0.00432546, 0.00333788]
  - `state_drift`: 3.696e-08

### test_london_penetration_depth_equals_kappa[kappa=3.0]

_λ = κ in these units — the field decays as exp(-x/κ) into the bulk_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=48, h=0.5, kappa=3, Bz=0.02, L_over_lambda=8
- **PASS** state drift (equilibrium reached): measured 1.009e-07, expected <= 1e-06 — must not exceed 1e-06
- **PASS** min |ψ| (still Meissner, no vortices): measured 0.998855, expected >= 0.9 — must be at least 0.9
- **PASS** λ from the screening profile: measured 3.13849, expected 3 — London penetration depth must equal the GL parameter κ
- **Diagnostics:**
  - `lambda_fit`: 3.13849
  - `profile`: [0.0169869, 0.014442, 0.0122933, 0.01048, 0.00895069, 0.00766161, 0.0065758, 0.0056619, … (12 values)]
  - `state_drift`: 1.009e-07

### test_london_series_satisfies_its_own_equation

_the analytical model must solve the equation it claims to solve_

- **Status:** PASS
- **Duration:** 0.263s
- **Parameters:** width=16, lambda=2, n_grid=[200, 400]
- **PASS** max |∇²B − B/λ²| at the finer grid: measured 2.400e-05, expected <= 0.0001 — dominated by the five-point stencil used to check it
- **PASS** residual ratio on halving the check grid: measured 4.00022, expected 4 — O(h²) means the residual belongs to the difference stencil
- **PASS** max |B − B₀| on an edge, 1 ξ from a corner: measured 0.00150745, expected <= 0.002 — Gibbs ringing from the truncated square wave; falls as 1/n_terms
- **PASS** finite values at n_terms = 201: measured 1, expected 1 — |measured - expected| <= 0
- **PASS** finite values at n_terms = 2001: measured 1, expected 1 — |measured - expected| <= 0
- **PASS** finite values at n_terms = 8001: measured 1, expected 1 — |measured - expected| <= 0
- **PASS** edge ringing at the default n_terms = 2001: measured 5.883e-04, expected <= 0.002 — this is the floor any comparison against the series inherits
- **PASS** ringing ratio 2001/201: measured 0.0734807, expected <= 0.2 — Gibbs error at fixed distance falls like 1/n_terms
- **PASS** ringing ratio 8001/2001: measured 0.249962, expected <= 0.5 — still falling past the default, so the default is not a plateau
- **Diagnostics:**
  - `max_pde_residual`: 200=9.600e-05, 400=2.400e-05
  - `max_edge_error`: 200=0.00157947, 400=0.00150745
  - `edge_finite_at_201_terms`: True
  - `edge_finite_at_2001_terms`: True
  - `edge_finite_at_8001_terms`: True
  - `edge_ringing`: 201=0.0080059, 2001=5.883e-04, 8001=1.470e-04

### test_london_slab_is_the_wide_limit_of_the_square

_the 1-D slab solution is the wide-square limit of the 2-D series_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** lambda=2, widths_over_lambda=[4, 8, 16, 32]
- **PASS** max |square − slab| at W = 4 λ: measured 0.1223, expected >= 0.05 — transverse screening is a real effect at this aspect ratio
- **PASS** max |square − slab| at W = 32 λ: measured 2.249e-07, expected <= 1e-06 — the two must agree once the square is wide enough
- **PASS** deviation ratio W = 8 λ / 4 λ: measured 0.239485, expected <= 0.3 — monotone approach, not a coincidence at one width
- **PASS** deviation ratio W = 16 λ / 8 λ: measured 0.0222226, expected <= 0.3 — monotone approach, not a coincidence at one width
- **PASS** deviation ratio W = 32 λ / 16 λ: measured 3.455e-04, expected <= 0.3 — monotone approach, not a coincidence at one width
- **PASS** edge deviation floor at W = 32 λ: measured 3.180e-04, expected <= 0.001 — Gibbs ringing, not a disagreement: it does not fall with W
- **Diagnostics:**
  - `core_deviation`: 4.0=0.1223, 8.0=0.029289, 16.0=6.509e-04, 32.0=2.249e-07
  - `full_deviation`: 4.0=0.1223, 8.0=0.029289, 16.0=6.509e-04, 32.0=3.180e-04

### test_lowest_landau_level_of_covariant_laplacian[Bz=0.1]

_the covariant Laplacian's ground-state energy is the lowest Landau level_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** L=24, h=0.5, Bz=0.1
- **PASS** max|∇×A − B| for the Landau-gauge links: measured 5.274e-16, expected <= 1e-13 — the test field must really be uniform before the spectrum means anything
- **PASS** non-Hermiticity of −(∇ − iA)²: measured 0, expected <= 1e-13 — must not exceed 1e-13
- **PASS** lowest eigenvalue E₀: measured 0.100006, expected 0.1 — E₀ = B places H_c2 at B = 1
- **Diagnostics:**
  - `E0`: 0.100006
  - `B_c2_implied`: 0.99994

### test_lowest_landau_level_of_covariant_laplacian[Bz=0.2]

_the covariant Laplacian's ground-state energy is the lowest Landau level_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** L=24, h=0.5, Bz=0.2
- **PASS** max|∇×A − B| for the Landau-gauge links: measured 1.055e-15, expected <= 1e-13 — the test field must really be uniform before the spectrum means anything
- **PASS** non-Hermiticity of −(∇ − iA)²: measured 0, expected <= 1e-13 — must not exceed 1e-13
- **PASS** lowest eigenvalue E₀: measured 0.198753, expected 0.2 — E₀ = B places H_c2 at B = 1
- **Diagnostics:**
  - `E0`: 0.198753
  - `B_c2_implied`: 1.00627

### test_magnetic_kappa_override_is_plaquette_centred

_the curl-curl operator is self-adjoint and dissipative for any ν_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** kappa_ref=2, magnetic_kappa_insulator=8, operator_norm=256
- **PASS** max |M - Mᵀ| / |M|: measured 0, expected <= 1e-12 — ν read per plaquette, so the term is the gradient of Σ ν_p B_p²
- **PASS** largest eigenvalue of the symmetric part: measured 1.690e-13, expected <= 2.56e-07 — the magnetic term may only remove energy, never add it

### test_mirror_symmetry_of_a_rectangular_device

_reflection symmetry on a non-square grid — a transposed index would break it_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=12, Ny=8, h=0.5, Bz=0.15
- **PASS** Bz contrast: measured 0.0272986, expected >= 0.001 — must be at least 0.001
- **PASS** max|ψ(x) − ψ(−x)|: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|ψ(y) − ψ(−y)|: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|Bz(x) − Bz(−x)|: measured 5.551e-17, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|Bz(y) − Bz(−y)|: measured 8.327e-17, expected <= 1e-12 — must not exceed 1e-12
- **Diagnostics:**
  - `psi_contrast`: 0.00447933

### test_no_vortices_in_the_meissner_state

_below H_c1 flux is expelled and the order parameter stays uniform_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=16, kappa=2, Bz=0.03
- **PASS** vortex count: measured 0, expected 0 — |measured - expected| <= 0
- **PASS** max |vorticity| anywhere: measured 1.657e-18, expected <= 1e-09 — must not exceed 1e-09
- **PASS** min |ψ|: measured 0.999159, expected >= 0.95 — no cores means no suppression of the order parameter
- **Diagnostics:**
  - `n_vortices`: 0
  - `psi_min`: 0.999159

### test_normal_supercurrent_vanishes_on_external_boundaries

_no supercurrent may cross the superconductor/vacuum interface_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, Ny=7, Nz=6, Bz=0.4
- **PASS** bulk current scale (non-trivial state): measured 0.39737, expected >= 0.0001 — must be at least 0.0001
- **PASS** max|J_n| on x_lo face: measured 4.229e-19, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|J_n| on x_hi face: measured 8.535e-19, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|J_n| on y_lo face: measured 4.442e-19, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|J_n| on y_hi face: measured 7.671e-19, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|J_n| on z_lo face: measured 4.120e-19, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max|J_n| on z_hi face: measured 5.866e-19, expected <= 1e-12 — must not exceed 1e-12
- **Diagnostics:**
  - `bulk_current_scale`: 0.39737

### test_observables_are_gauge_invariant[10x10x1]

_every measurable quantity must be independent of the gauge_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=10, Ny=10, Nz=1, kappa=2
- **PASS** max Δ|ψ|: measured 2.220e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max ΔB: measured 2.776e-16, expected <= 1e-11 — must not exceed 1e-11
- **PASS** max ΔJ_s: measured 2.220e-16, expected <= 1.04e-11 — must not exceed 1.04e-11
- **PASS** Δ free energy: measured 0, expected <= 5.47e-08 — must not exceed 5.47e-08
- **Diagnostics:**
  - `B_scale`: 0.891155
  - `J_scale`: 1.04104
  - `free_energy`: 54.6602

### test_observables_are_gauge_invariant[6x7x5]

_every measurable quantity must be independent of the gauge_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=7, Nz=5, kappa=2
- **PASS** max Δ|ψ|: measured 2.220e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max ΔB: measured 2.220e-16, expected <= 1e-11 — must not exceed 1e-11
- **PASS** max ΔJ_s: measured 1.665e-16, expected <= 1.55e-11 — must not exceed 1.55e-11
- **PASS** Δ free energy: measured 2.842e-14, expected <= 1.9e-07 — must not exceed 1.9e-07
- **Diagnostics:**
  - `B_scale`: 0.941579
  - `J_scale`: 1.55302
  - `free_energy`: 190.385

### test_observables_are_gauge_invariant[9x7x1]

_every measurable quantity must be independent of the gauge_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=7, Nz=1, kappa=3
- **PASS** max Δ|ψ|: measured 3.331e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max ΔB: measured 3.331e-16, expected <= 1e-11 — must not exceed 1e-11
- **PASS** max ΔJ_s: measured 2.220e-16, expected <= 1e-11 — must not exceed 1e-11
- **PASS** Δ free energy: measured 0, expected <= 5.61e-08 — must not exceed 5.61e-08
- **Diagnostics:**
  - `B_scale`: 0.804717
  - `J_scale`: 0.875324
  - `free_energy`: 56.0777

### test_order_parameter_matches_the_exact_wall_solution

_the order parameter heals over √2 ξ, matching the exact 1-D solution_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** length=24, kappa=2, h_values=[1, 0.5, 0.25], tau=0.1
- **PASS** |ψ| at the interface from matching: measured 0.213422, expected 0.2134 — positive root of √τ u² + √2 u − √τ = 0
- **PASS** rms error at h = 1 ξ: measured 0.0496938, expected <= 0.08 — must not exceed 0.08
- **PASS** rms error at h = 0.25 ξ: measured 0.00476747, expected <= 0.01 — must not exceed 0.01
- **PASS** max error at h = 0.25 ξ: measured 0.0162006, expected <= 0.03 — must not exceed 0.03
- **PASS** observed order in h: measured 1.69088, expected >= 1.5 — the coefficient jump is first order locally, so h^1.5 is the floor
- **PASS** rms |ψ′ − (1 − ψ²)/√2| at h = 1 ξ: measured 0.00934297, expected <= 0.02 — offset-free: no interface position enters this identity
- **PASS** rms |ψ′ − (1 − ψ²)/√2| at h = 0.25 ξ: measured 8.034e-04, expected <= 0.002 — must not exceed 0.002
- **PASS** observed order in h (first integral): measured 1.76982, expected >= 1.5 — the √2 healing length, checked without a fit or a position
- **Diagnostics:**
  - `rms_error`: 1.0=0.0496938, 0.5=0.0172373, 0.25=0.00476747
  - `max_error`: 1.0=0.14913, 0.5=0.056927, 0.25=0.0162006
  - `first_integral_residual`: 1.0=0.00934297, 0.5=0.00329477, 0.25=8.034e-04

### test_padded_stack_is_mirror_symmetric

_a mirror-symmetric stack in a symmetric box gives a symmetric field_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nz=18, vacuum_cells=5, layers=3/2/3, kappa=2, applied_Bz=0.02
- **PASS** max |Bz(z) - Bz(-z)| / applied: measured 2.220e-16, expected <= 1e-12 — exact symmetry, not an approximate one
- **Diagnostics:**
  - `bz_profile_over_applied`: [0.937779, 0.928432, 0.907438, 0.869673, 0.806862, 0.771334, 0.793769, 0.83978, … (17 values)]

### test_penetration_depth_converges_with_grid_refinement

_the measured λ must approach κ as the grid is refined_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** L=16, kappa=2, h_values=[1, 0.5]
- **PASS** |λ − κ| at h = 1.0: measured 0.348405, expected <= 0.7 — must not exceed 0.7
- **PASS** |λ − κ| at h = 0.5: measured 0.105629, expected <= 0.2 — must not exceed 0.2
- **PASS** error ratio fine/coarse: measured 0.30318, expected <= 0.75 — refinement must reduce the discretisation error
- **Diagnostics:**
  - `lambda_h1.0`: 2.3484
  - `lambda_h0.5`: 2.10563

### test_plaquette_vorticity_is_an_exact_integer

_Σ wrap(Δθ − φ) + Φ_plaquette is exactly 2π × integer_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=20, kappa=2, Bz=0.5
- **PASS** plaquettes carrying vorticity: measured 8, expected >= 1 — must be at least 1
- **PASS** max |vorticity − nearest integer|: measured 2.474e-16, expected <= 1e-10 — must not exceed 1e-10
- **Diagnostics:**
  - `n_charged_plaquettes`: 8
  - `vorticity_values`: [0, 1]

### test_rhs_covariant_with_material_map

_a spatially varying κ and an insulator mask must not break gauge covariance_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nz=6, kappa_sc=2
- **PASS** max covariance violation: measured 3.997e-15, expected <= 1.84e-10 — must not exceed 1.84e-10
- **Diagnostics:**
  - `rhs_scale`: 18.4327

### test_rhs_is_gauge_covariant[10x10x1]

_dψ/dt must rotate with the gauge phase and dφ/dt must be invariant_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=10, Ny=10, Nz=1, kappa=2
- **PASS** max|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)|: measured 9.155e-16, expected <= 8.15e-11 — ψ-equation must be covariant under ψ→ψe^{iχ}, φ→φ+Δχ
- **PASS** max|dφ/dt(GX) − dφ/dt(X)|: measured 2.276e-15, expected <= 8.15e-11 — the supercurrent source and curl-curl term must be gauge invariant
- **Diagnostics:**
  - `rhs_scale`: 8.14607
  - `gauge_amplitude`: 0.772115

### test_rhs_is_gauge_covariant[6x7x5]

_dψ/dt must rotate with the gauge phase and dφ/dt must be invariant_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=7, Nz=5, kappa=2
- **PASS** max|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)|: measured 1.335e-15, expected <= 1.21e-10 — ψ-equation must be covariant under ψ→ψe^{iχ}, φ→φ+Δχ
- **PASS** max|dφ/dt(GX) − dφ/dt(X)|: measured 5.773e-15, expected <= 1.21e-10 — the supercurrent source and curl-curl term must be gauge invariant
- **Diagnostics:**
  - `rhs_scale`: 12.1096
  - `gauge_amplitude`: 0.764307

### test_rhs_is_gauge_covariant[9x7x1]

_dψ/dt must rotate with the gauge phase and dφ/dt must be invariant_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=9, Ny=7, Nz=1, kappa=3
- **PASS** max|dψ/dt(Gψ) − e^{iχ} dψ/dt(ψ)|: measured 9.930e-16, expected <= 1.02e-10 — ψ-equation must be covariant under ψ→ψe^{iχ}, φ→φ+Δχ
- **PASS** max|dφ/dt(GX) − dφ/dt(X)|: measured 3.553e-15, expected <= 1.02e-10 — the supercurrent source and curl-curl term must be gauge invariant
- **Diagnostics:**
  - `rhs_scale`: 10.1619
  - `gauge_amplitude`: 0.764307

### test_ring_expels_flux_below_threshold

_below threshold the multiply-connected ring keeps the enclosed fluxoid at zero_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** hole=4, arm=3, kappa=2, Bz=0.05, t_hold=30
- **PASS** max |fluxoid| over the whole run: measured 1.877e-17, expected <= 1e-09 — not one quantum enters at any time
- **PASS** max |dX/dt| at the end: measured 6.090e-14, expected <= 0.001 — the expelled state is a fixed point, not a slow transient
- **Diagnostics:**
  - `fluxoid_history`: [0, 0, -7.731e-18, -2.209e-18, -2.209e-18, -1.877e-17, 8.835e-18, -2.209e-18, … (17 values)]
  - `residual`: 6.090e-14

### test_solution_reshape_helpers_are_consistent

_the 2-D view must be indexed [i, j] with the interior strides_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=11, Ny=6
- **PASS** shape[0]: measured 10, expected 10 — |measured - expected| <= 0
- **PASS** shape[1]: measured 5, expected 5 — |measured - expected| <= 0
- **PASS** max|reshape − stride-indexed|: measured 0, expected <= 1e-15 — must not exceed 1e-15
- **Diagnostics:**
  - `shape`: [10, 5]

### test_stack_is_mirror_symmetric_about_its_midplane[(3, 1)]

_equal superconducting layers must give an exactly symmetric material map_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** sc_cells=3, insulator_cells=1, Nz=7
- **PASS** sc_mask asymmetry under z → Nz − z: measured 0, expected <= 0 — must not exceed 0
- **PASS** κ asymmetry under z → Nz − z: measured 0, expected <= 1e-15 — must not exceed 1e-15
- **PASS** superconducting nodes below vs above the mid-plane: measured 3, expected 3 — |measured - expected| <= 0
- **Diagnostics:**
  - `sc_mask_z_profile`: [1, 1, 1, 0, 0, 1, 1, 1]

### test_stack_is_mirror_symmetric_about_its_midplane[(4, 2)]

_equal superconducting layers must give an exactly symmetric material map_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** sc_cells=4, insulator_cells=2, Nz=10
- **PASS** sc_mask asymmetry under z → Nz − z: measured 0, expected <= 0 — must not exceed 0
- **PASS** κ asymmetry under z → Nz − z: measured 0, expected <= 1e-15 — must not exceed 1e-15
- **PASS** superconducting nodes below vs above the mid-plane: measured 4, expected 4 — |measured - expected| <= 0
- **Diagnostics:**
  - `sc_mask_z_profile`: [1, 1, 1, 1, 0, 0, 0, 1, … (11 values)]

### test_stack_is_mirror_symmetric_about_its_midplane[(5, 2)]

_equal superconducting layers must give an exactly symmetric material map_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** sc_cells=5, insulator_cells=2, Nz=12
- **PASS** sc_mask asymmetry under z → Nz − z: measured 0, expected <= 0 — must not exceed 0
- **PASS** κ asymmetry under z → Nz − z: measured 0, expected <= 1e-15 — must not exceed 1e-15
- **PASS** superconducting nodes below vs above the mid-plane: measured 5, expected 5 — |measured - expected| <= 0
- **Diagnostics:**
  - `sc_mask_z_profile`: [1, 1, 1, 1, 1, 0, 0, 0, … (13 values)]

### test_supercurrent_is_divergence_free_in_steady_state

_∂(∇·A)/∂t = ∇·J_s, so a stationary gauge field forces a solenoidal current_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=14, h=0.5, kappa=2, Bz=0.12
- **PASS** state drift between saved steps: measured 8.183e-12, expected <= 1e-06 — must not exceed 1e-06
- **PASS** max|∇·J_s| · h / max|J_s|: measured 5.588e-14, expected <= 1e-06 — must not exceed 1e-06
- **Diagnostics:**
  - `J_scale`: 0.0795678
  - `state_drift_between_saves`: 8.183e-12

### test_the_relaxed_ring_is_symmetric

_the solution must inherit every symmetry of the device_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** hole=4, arm=3, kappa=2, Bz=0.05
- **PASS** |ψ| scale (non-trivial): measured 0.928977, expected >= 0.5 — must be at least 0.5
- **PASS** Bz scale (non-trivial): measured 0.0482529, expected >= 0.001 — must be at least 0.001
- **PASS** max |ψ| asymmetry under x → −x: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max |ψ| asymmetry under y → −y: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max |ψ| asymmetry under z → −z: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max |ψ| asymmetry under 90° rotation: measured 1.110e-16, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max Bz asymmetry under x → −x: measured 2.776e-17, expected <= 4.83e-14 — must not exceed 4.83e-14
- **PASS** max Bz asymmetry under y → −y: measured 2.776e-17, expected <= 4.83e-14 — must not exceed 4.83e-14
- **PASS** max Bz asymmetry under z → −z: measured 2.776e-17, expected <= 4.83e-14 — must not exceed 4.83e-14
- **PASS** max Bz asymmetry under 90° rotation: measured 2.082e-17, expected <= 4.83e-14 — must not exceed 4.83e-14
- **Diagnostics:**
  - `psi_scale`: 0.928977
  - `Bz_scale`: 0.0482529

### test_the_ring_is_superconducting

_the superconducting layers must be thicker than the proximity length_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** hole=4, arm=3, sc_thickness=4, kappa=2
- **PASS** max |ψ| in the superconducting layers: measured 0.928977, expected >= 0.9 — the middle of a 4 ξ layer must recover the bulk condensate
- **PASS** mean |ψ| in the superconducting layers: measured 0.660557, expected >= 0.5 — must be at least 0.5
- **PASS** max |ψ| in the oxide and the hole: measured 0.103677, expected <= 0.25 — must not exceed 0.25
- **Diagnostics:**
  - `psi_max_in_arms`: 0.928977
  - `psi_mean_in_arms`: 0.660557
  - `psi_max_in_insulator_and_hole`: 0.103677

### test_three_dimensional_solver_reproduces_the_two_dimensional_london_solution

_a z-invariant problem is solved identically by the 2-D and 3-D paths_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** h=1, Bz=0.02, Nz_2d=1, Nz_3d=4, n_layers=3
- **PASS** min |ψ| in the 3-D run: measured 0.999627, expected >= 0.99 — must be at least 0.99
- **PASS** spread of Bz across z-slices / B₀: measured 1.110e-16, expected <= 1e-10 — nothing in the problem depends on z, so nothing in the answer may
- **PASS** max |B_3D − B_2D| / B₀: measured 1.706e-10, expected <= 1e-08 — the same discrete equations, so the same fixed point
- **PASS** max |B_3D − model| / B₀: measured 0.00490852, expected <= 0.01 — and the shared fixed point is the physical one
- **Diagnostics:**
  - `n_z_layers`: 3
  - `layer_profiles_over_B0`: [[0.61943, 0.39059, 0.253843, 0.172957, 0.126159, 0.10068, 0.0894936, 0.0894936, … (15 values)], [0.61943, 0.39059, 0.253843, 0.172957, 0.126159, 0.10068, 0.0894936, 0.0894936, … (15 values)], [0.61943, 0.39059, 0.253843, 0.172957, 0.126159, 0.10068, 0.0894936, 0.0894936, … (15 values)]]

### test_trapezoidal_agrees_with_euler_in_the_small_dt_limit

_two independent integrators of the same right-hand side must agree_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=5, Ny=4, t_stop=0.1, Bz=0.25
- **PASS** max|X_trapezoidal − X_euler| / |X|: measured 1.563e-05, expected <= 0.001 — must not exceed 0.001
- **Diagnostics:**
  - `state_scale`: 1.02792
  - `relative_difference`: 1.563e-05

### test_trilayer_external_z_boundary_jn

_the z-faces of the stack are superconductor/vacuum interfaces_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=5, Ny=4, Nz=6, Bz=0.5
- **PASS** max |J_z| on the bottom face: measured 1.095e-21, expected <= 1e-12 — must not exceed 1e-12
- **PASS** max |J_z| on the top face: measured 2.121e-21, expected <= 1e-12 — must not exceed 1e-12
- **Diagnostics:**
  - `bulk_Jz_scale`: 1.524e-07

### test_trilayer_kappa_discontinuity

_the Maxwell coefficient is the vacuum one in every layer_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=4, Ny=5, Nz=6, kappa=2, kappa_insulator_declared=0
- **PASS** LPHI_x diagonal in the superconductor: measured -16, expected -16 — |measured - expected| <= 1e-12
- **PASS** LPHI_x diagonal in the insulator: measured -16, expected -16 — the field energy does not know it is inside an oxide
- **Diagnostics:**
  - `k_superconductor`: 1
  - `k_insulator`: 2

### test_trilayer_superconducting_layers_screen

_both superconducting layers of a magnetically continuous stack screen_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=16, Nz=12, kappa=2, Bz=0.1
- **PASS** Bz in the bottom Nb layer / applied: measured 0.852883, expected <= 0.98 — must not exceed 0.98
- **PASS** Bz in the top Nb layer / applied: measured 0.861574, expected <= 0.98 — must not exceed 0.98
- **PASS** Bz in the bottom Nb layer / applied: measured 0.852883, expected >= 0 — must be at least 0
- **PASS** top/bottom screening asymmetry: measured 1.01019, expected 1 — the stack is symmetric about its mid-plane
- **Diagnostics:**
  - `bz_profile_over_applied`: [0.838297, 0.850563, 0.869788, 0.887649, 0.898405, 0.901966, 0.898405, 0.887649, … (11 values)]
  - `bottom_over_applied`: 0.852883
  - `top_over_applied`: 0.861574

### test_uniform_state_is_an_exact_fixed_point

_the Meissner ground state must not drift_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=7, Ny=6, Nz=5
- **PASS** max|dX/dt|: measured 0, expected <= 1e-13 — must not exceed 1e-13
- **PASS** kinetic + magnetic energy: measured 0, expected <= 1e-13 — a uniform state carries no gradient or field energy
- **PASS** condensation energy per unit volume: measured -0.5, expected -0.5 — -|ψ|² + ½|ψ|⁴ = -½ at |ψ| = 1
- **Diagnostics:**
  - `free_energy_terms`: condensation=-60, kinetic=0, magnetic=0, total=-60

### test_vortex_count_increases_with_the_applied_field

_the mixed state admits more flux quanta as the applied field rises_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=16, kappa=2, Bz_values=[0.35, 0.5, 0.7]
- **PASS** largest decrease in count along the sweep: measured -4, expected <= 0 — counts [0, 4, 12] at Bz [0.35, 0.5, 0.7] must not decrease
- **PASS** increase from the lowest to the highest field: measured 12, expected >= 1 — must be at least 1
- **PASS** count / (B·A/Φ₀) at Bz = 0.35: measured 0, expected <= 1 — screening keeps the interior field below the applied field
- **PASS** count / (B·A/Φ₀) at Bz = 0.5: measured 0.19635, expected <= 1 — screening keeps the interior field below the applied field
- **PASS** count / (B·A/Φ₀) at Bz = 0.7: measured 0.420749, expected <= 1 — screening keeps the interior field below the applied field
- **PASS** mean interior Bz / applied at Bz = 0.35: measured 0.42922, expected <= 1 — the sample still screens in the mixed state
- **PASS** mean interior Bz / applied at Bz = 0.5: measured 0.681309, expected <= 1 — the sample still screens in the mixed state
- **PASS** mean interior Bz / applied at Bz = 0.7: measured 0.853044, expected <= 1 — the sample still screens in the mixed state
- **Diagnostics:**
  - `vortex_counts`: [0, 4, 12]
  - `applied_flux_quanta`: [14.26, 20.37, 28.52]
  - `interior_field_over_applied`: [0.4292, 0.6813, 0.853]

### test_vortex_count_is_gauge_invariant

_plaquette vorticity is a topological invariant of the gauge-field configuration_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=14, kappa=2, Bz=0.6
- **PASS** vortices present (test would be vacuous otherwise): measured 8, expected >= 1 — must be at least 1
- **PASS** vortex count after gauge change: measured 8, expected 8 — |measured - expected| <= 0
- **PASS** max Δ(plaquette vorticity): measured 2.297e-16, expected <= 1e-09 — must not exceed 1e-09
- **PASS** max |winding change|: measured 1.110e-16, expected <= 1e-09 — must not exceed 1e-09
- **Diagnostics:**
  - `n_vortices`: 8
  - `windings`: [1, 1, 1, 1, 1, 1, 1, 1]

### test_vortex_winding_sign_follows_the_applied_field[Bz=-0.5]

_vortices in a uniform field are all of the same chirality as the field_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=20, kappa=2, Bz=-0.5
- **PASS** vortices detected: measured 8, expected >= 1 — must be at least 1
- **PASS** distinct winding values: measured 1, expected 1 — found [-1]
- **PASS** common winding: measured -1, expected -1 — winding sign must match the sign of the applied field
- **PASS** max |winding|: measured 1, expected 1 — singly quantised vortices at this field
- **Diagnostics:**
  - `n_vortices`: 8
  - `winding_values`: [-1]

### test_vortex_winding_sign_follows_the_applied_field[Bz=0.5]

_vortices in a uniform field are all of the same chirality as the field_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=20, kappa=2, Bz=0.5
- **PASS** vortices detected: measured 8, expected >= 1 — must be at least 1
- **PASS** distinct winding values: measured 1, expected 1 — found [1]
- **PASS** common winding: measured 1, expected 1 — winding sign must match the sign of the applied field
- **PASS** max |winding|: measured 1, expected 1 — singly quantised vortices at this field
- **Diagnostics:**
  - `n_vortices`: 8
  - `winding_values`: [1]

### test_vortices_grow_from_zero_and_saturate

_vortices must nucleate from the uniform state and reach a steady number_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=20, kappa=2, Bz=0.5, t_stop=60
- **PASS** vortex count at t = 0: measured 0, expected 0 — |measured - expected| <= 0
- **PASS** final vortex count: measured 8, expected >= 1 — must be at least 1
- **PASS** time of first vortex entry: measured 10, expected <= 30 — must not exceed 30
- **PASS** std/mean of the count over the final quarter: measured 0, expected <= 0.25 — the vortex population must settle rather than keep growing
- **Diagnostics:**
  - `times`: [0, 5, 10, 15, 20, 25, 30, 35, … (13 values)]
  - `vortex_counts`: [0, 0, 8, 8, 8, 8, 8, 8, … (13 values)]
  - `t_first_vortex`: 10

### test_zero_field_ground_state_is_the_uniform_condensate

_|ψ| = 1 minimises −|ψ|² + ½|ψ|⁴; the ground state must reach it_

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=10, Ny=8, h=0.5, kappa=2
- **PASS** min |ψ|: measured 1, expected 1 — |measured - expected| <= 0.0001
- **PASS** max |ψ|: measured 1, expected 1 — |measured - expected| <= 0.0001
- **PASS** max |B| in the relaxed state: measured 5.551e-17, expected <= 1e-06 — must not exceed 1e-06
- **PASS** max |dX/dt| at the fixed point: measured 4.461e-15, expected <= 0.0001 — must not exceed 0.0001
- **Diagnostics:**
  - `psi_min`: 1
  - `psi_max`: 1
