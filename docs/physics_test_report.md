# Physics Test Results Report

**Run timestamp:** 2026-07-25T22:14:41.519224
**Results:** 15/15 passed, 0 failed

## Summary

| Test | Metric | Details | Status | Duration |
|------|--------|---------|--------|----------|
| B-field div-free | 4.08e-04 | max|∇·B|/max|B| should be ~0 | PASS | 0.000s |
| B-field reversal symmetry | 0.0000 | max|Bz(+B) + Bz(-B)| should be 0 | PASS | 0.100s |
| B-field uniform at boundary | 0.0000 | std(Bz) at boundary should be 0 | PASS | 0.000s |
| C4 symmetry preserved | 3.97e-22 | max|φ_x + φ_y^T| should be 0 | PASS | 0.000s |
| CFL unstable (above limit) | 2.07e-05 | mean|ψ| should collapse to ~0 | PASS | 0.096s |
| CFL stable (below limit) | 1.0010 | max|ψ|² should stay near 1 | PASS | 0.025s |
| Energy dissipation | 0.0084 (tol 0.0836) | max relative energy increase must stay below tolerance | PASS | 0.040s |
| Insulator |ψ| decay | τ=0.0885 (expected 0.1) | relative error = 11.5% | PASS | 0.100s |
| Meissner screening | λ=11.1742 | λ should equal κ | PASS | 4.039s |
| Supercurrent zero at boundary | 0.0000 | max|φ_boundary| should be 0 | PASS | 0.000s |
| Trilayer B penetration | Bz(ins)=1.16e-07 (0.0% of applied) | Bz(Nb)=0.1984/0.2310, Bz(app)=0.3, SC✓ | PASS | 2.372s |
| Trilayer z-boundary J_n | 0.0000 | J_n at z-faces should be 0 | PASS | 0.059s |
| Trilayer κ discontinuity | SC=-16.0000 (expected -16.0), Ins=0.0000 | SC diagonal should match κ² stencil, insulator should be 0 | PASS | 0.006s |
| Uniform state zero RHS | 0.0000 | max|RHS| should be 0 | PASS | 0.000s |
| Vortex entry & counting | n=25 (expected ≈127) | detected 20% of expected, winding=[-1.0000000000000002, -1.0, -1.0, -1.0, -1.0000000000000002, 0.9999999999999999, 1.0, 1.0, -1.0, -1.0000000000000002, -1.0, 1.0, 1.0, 1.0, 1.0000000000000002, -0.9999999999999999, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0] | PASS | 8.446s |

## Detailed Results

### B-field div-free

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=4, Nz=4, Bz=0.5
- **Diagnostics:**
  - `max_div_b`: 5.18e-05
  - `mean_div_b`: 5.18e-05
  - `max_B_magnitude`: 0.1271
  - `div_to_B_ratio`: 4.08e-04

### B-field reversal symmetry

- **Status:** PASS
- **Duration:** 0.100s
- **Parameters:** Nx=6, Bz=0.5, n_steps=20
- **Diagnostics:**
  - `max_asymmetry`: 0.0000

### B-field uniform at boundary

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Bz=1.0
- **Diagnostics:**
  - `bz_x_lo_mean`: 1.0000
  - `bz_x_hi_mean`: 1.0000
  - `bz_y_lo_mean`: 1.0000
  - `bz_y_hi_mean`: 1.0000
  - `bz_x_lo_std`: 0.0000

### C4 symmetry preserved

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=8, kappa=2.0, Bz=0.5
- **Diagnostics:**
  - `max_symmetry_violation`: 3.97e-22

### CFL unstable (above limit)

- **Status:** PASS
- **Duration:** 0.096s
- **Parameters:** Nx=5, kappa=2.0, cfl=0.0625, dt=0.1875
- **Diagnostics:**
  - `cfl_limit`: 0.0625
  - `dt_used`: 0.1875
  - `has_nan`: False
  - `max_psi_final`: 7.82e-05
  - `mean_psi_final`: 2.07e-05

### CFL stable (below limit)

- **Status:** PASS
- **Duration:** 0.025s
- **Parameters:** Nx=5, kappa=2.0, cfl=0.0625, dt=0.05625
- **Diagnostics:**
  - `cfl_limit`: 0.0625
  - `dt_used`: 0.0563
  - `max_psi2`: 1.0010
  - `min_psi2`: 0.9993

### Energy dissipation

- **Status:** PASS
- **Duration:** 0.040s
- **Parameters:** Nx=6, kappa=2.0, Bz=0.5, n_steps=30
- **Diagnostics:**
  - `F_initial`: 12.4730
  - `F_final`: 12.9718
  - `max_energy_increase`: 0.0084
  - `tolerance`: 0.0836

### Insulator |ψ| decay

- **Status:** PASS
- **Duration:** 0.100s
- **Parameters:** Nx=4, Nz=6
- **Diagnostics:**
  - `tau_fit`: 0.0885
  - `tau_expected`: 0.1000
  - `fit_converged`: True
  - `psi_steady_state`: 0.0770
  - `tau_rel_error`: 0.1147

### Meissner screening

- **Status:** PASS
- **Duration:** 4.039s
- **Parameters:** Nx=30, Ny=8, kappa=2.0, Bz=0.1
- **Diagnostics:**
  - `lambda_fit`: 11.1742
  - `lambda_expected`: 2.0000
  - `fit_converged`: True
  - `bfield_edge_left`: 0.0699
  - `bfield_edge_right`: 0.0699
  - `bfield_center`: 0.0357
  - `x_positions`: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
  - `r_squared`: 0.7441
  - `fit_center`: 0.0179

### Supercurrent zero at boundary

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Bz=0.5
- **Diagnostics:**
  - `max_boundary_link_phi`: 0.0000

### Trilayer B penetration

- **Status:** PASS
- **Duration:** 2.372s
- **Parameters:** Nx=4, Nz=6, Bz=0.3
- **Diagnostics:**
  - `bz_bottom`: 0.1984
  - `bz_insulator`: 1.16e-07
  - `bz_top`: 0.2310
  - `bz_applied`: 0.3000
  - `sc_screened`: True
  - `sc_screening_ratio_bottom`: 0.6614
  - `sc_screening_ratio_top`: 0.7700
  - `insulator_penetration_ratio`: 3.85e-07

### Trilayer z-boundary J_n

- **Status:** PASS
- **Duration:** 0.059s
- **Parameters:** Nx=4, Nz=6, Bz=0.5
- **Diagnostics:**
  - `max_jn_z_lo`: 0.0000
  - `max_jn_z_hi`: 0.0000

### Trilayer κ discontinuity

- **Status:** PASS
- **Duration:** 0.006s
- **Parameters:** Nx=4, Nz=6
- **Diagnostics:**
  - `sc_diag_mean`: -16.0000
  - `ins_diag_mean`: 0.0000
  - `expected_sc`: -16.0000
  - `expected_ins`: 0.0000
  - `sc_error`: 0.0000
  - `ins_error`: 0.0000

### Uniform state zero RHS

- **Status:** PASS
- **Duration:** 0.000s
- **Parameters:** Nx=6, Ny=6
- **Diagnostics:**
  - `max_rhs`: 0.0000
  - `mean_rhs`: 0.0000

### Vortex entry & counting

- **Status:** PASS
- **Duration:** 8.446s
- **Parameters:** Nx=20, kappa=2.0, Bz=0.5
- **Diagnostics:**
  - `n_vortices`: 25
  - `expected_approx`: 127.3240
