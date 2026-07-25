"""Quick symmetry check on the final solution."""
import numpy as np
import tdgl3d
from tdgl3d.core.parameters import SimulationParameters

Nx = Ny = 20; Nz = 4
params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
field = tdgl3d.AppliedField(Bz=1.0, t_on_fraction=1.0, ramp=True, ramp_fraction=0.5)
device = tdgl3d.Device(params, applied_field=field)

rng = np.random.default_rng(42)
x0 = tdgl3d.StateVector.uniform_superconducting(params).data.copy()
n = params.n_interior
nx_int, ny_int, nz_int = Nx-1, Ny-1, max(Nz-1,1)
amp = 0.50 + 0.50*rng.random((nx_int, ny_int, nz_int))
phase = 2*np.pi*rng.random((nx_int, ny_int, nz_int))
pert = amp * np.exp(1j*phase)
pert_sym = np.zeros_like(pert)
for r in range(4):
    pert_sym += np.rot90(pert, k=r, axes=(0,1))
pert_sym /= 4.0
x0[:n] *= pert_sym.ravel()

sol = tdgl3d.solve(device, t_start=0, t_stop=100, dt=0.01,
    method="euler", x0=x0, save_every=500, progress=True)

psi = sol.psi(step=-1).reshape(nx_int, ny_int, nz_int)
psi2 = np.abs(psi)**2
mid_z = nz_int // 2
s = psi2[:, :, mid_z]
print(f"\nSlice shape: {s.shape}")
print(f"|s - rot90(s)| max: {np.max(np.abs(s - np.rot90(s))):.2e}")
print(f"|s - s.T| max: {np.max(np.abs(s - s.T)):.2e}")
print(f"mean: {np.mean(s):.4f}, min: {np.min(s):.4f}, max: {np.max(s):.4f}")
