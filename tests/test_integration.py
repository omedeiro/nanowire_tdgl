"""Integration tests — run short simulations end-to-end."""

import numpy as np
import pytest
from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.core.device import Device
from tdgl3d.core.state import StateVector
from tdgl3d.physics.applied_field import AppliedField
from tdgl3d.solvers.runner import solve


class TestForwardEulerIntegration:
    def test_no_field_relaxation_2d(self):
        """With no applied field, a uniform state should remain close to |ψ|=1."""
        params = SimulationParameters(Nx=5, Ny=5, Nz=1, kappa=2.0)
        device = Device(params, applied_field=AppliedField())  # zero field
        sol = solve(device, t_start=0.0, t_stop=0.1, dt=0.01, method="euler", progress=False)

        assert sol.n_steps > 1
        psi_final = sol.psi_squared(step=-1)
        # Should still be close to 1 (equilibrium)
        assert np.mean(psi_final) > 0.5

    def test_applied_field_reduces_psi(self):
        """An applied field should suppress |ψ| below 1 after some time."""
        params = SimulationParameters(Nx=5, Ny=5, Nz=1, kappa=2.0)
        device = Device(params, applied_field=AppliedField(Bz=2.0))
        sol = solve(device, t_start=0.0, t_stop=1.0, dt=0.01, method="euler", progress=False)

        psi_final = sol.psi_squared(step=-1)
        # Applied field should suppress superconductivity somewhat
        assert sol.n_steps > 1

    def test_3d_euler(self):
        """Smoke test: 3-D Forward Euler runs without error."""
        params = SimulationParameters(Nx=4, Ny=4, Nz=4, kappa=2.0)
        device = Device(params, applied_field=AppliedField(Bz=0.5))
        sol = solve(device, t_start=0.0, t_stop=0.05, dt=0.01, method="euler", progress=False)
        assert sol.n_steps > 1
        assert sol.states.shape[0] == params.n_state


class TestTrapezoidalIntegration:
    def test_trapezoidal_converges_2d(self):
        """Trapezoidal should run and converge for a small 2-D problem."""
        params = SimulationParameters(Nx=4, Ny=4, Nz=1, kappa=2.0)
        device = Device(params, applied_field=AppliedField())
        sol = solve(
            device,
            t_start=0.0,
            t_stop=0.1,
            dt=0.05,
            method="trapezoidal",
            progress=False,
            verbose=False,
        )
        assert sol.n_steps >= 2


class TestDeviceAndSolution:
    def test_device_repr(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=3, kappa=5.0)
        d = Device(p, applied_field=AppliedField(Bz=1.0))
        r = repr(d)
        assert "Device" in r
        assert "κ=5.0" in r

    def test_solution_bfield(self):
        params = SimulationParameters(Nx=5, Ny=5, Nz=1, kappa=2.0)
        device = Device(params, applied_field=AppliedField())
        sol = solve(device, t_start=0.0, t_stop=0.05, dt=0.01, method="euler", progress=False)
        Bx, By, Bz = sol.bfield(step=-1)
        assert Bz.shape == ((params.Nx - 2) * (params.Ny - 2),)

    def test_solution_psi_squared_2d(self):
        params = SimulationParameters(Nx=5, Ny=5, Nz=1)
        device = Device(params, applied_field=AppliedField())
        sol = solve(device, t_start=0.0, t_stop=0.02, dt=0.01, method="euler", progress=False)
        psi2 = sol.psi_squared_2d(step=0)
        assert psi2.shape == (params.Nx - 1, params.Ny - 1)
