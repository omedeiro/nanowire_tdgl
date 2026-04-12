"""Tests for SimulationParameters."""

import pytest
from tdgl3d.core.parameters import SimulationParameters


class TestSimulationParameters:
    def test_defaults(self):
        p = SimulationParameters()
        assert p.Nx == 10
        assert p.Ny == 10
        assert p.Nz == 1
        assert p.kappa == 5.0

    def test_is_3d(self):
        p2d = SimulationParameters(Nz=1)
        assert not p2d.is_3d
        p3d = SimulationParameters(Nz=4)
        assert p3d.is_3d

    def test_n_interior_2d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        assert p.n_interior == (5 - 1) * (5 - 1) * 1  # 16

    def test_n_interior_3d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=5)
        assert p.n_interior == (5 - 1) * (5 - 1) * (5 - 1)  # 64

    def test_n_state_2d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        # 2D: psi + phi_x + phi_y = 3 * n_interior
        assert p.n_state == 3 * p.n_interior

    def test_n_state_3d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=4)
        # 3D: psi + phi_x + phi_y + phi_z = 4 * n_interior
        assert p.n_state == 4 * p.n_interior

    def test_dim_x_2d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        assert p.dim_x == (5 + 1) * (5 + 1)

    def test_dim_x_3d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=5)
        assert p.dim_x == (5 + 1) * (5 + 1) * (5 + 1)

    def test_strides(self):
        p = SimulationParameters(Nx=6, Ny=8, Nz=4)
        assert p.mj == 7
        assert p.mk == 7 * 9

    def test_validation(self):
        with pytest.raises(ValueError):
            SimulationParameters(Nx=1, Ny=3)
        with pytest.raises(ValueError):
            SimulationParameters(Nx=3, Ny=1)
        with pytest.raises(ValueError):
            SimulationParameters(Nx=3, Ny=3, Nz=0)

    def test_copy(self):
        p = SimulationParameters(Nx=7, kappa=3.0)
        p2 = p.copy()
        p2.kappa = 99.0
        assert p.kappa == 3.0
