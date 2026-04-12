"""Tests for StateVector."""

import numpy as np
import pytest
from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.core.state import StateVector


class TestStateVector:
    def test_uniform_superconducting_2d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        sv = StateVector.uniform_superconducting(p)
        assert sv.data.shape == (p.n_state,)
        np.testing.assert_allclose(np.abs(sv.psi), 1.0)
        np.testing.assert_allclose(sv.phi_x, 0.0)
        np.testing.assert_allclose(sv.phi_y, 0.0)

    def test_uniform_superconducting_3d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=4)
        sv = StateVector.uniform_superconducting(p)
        assert sv.data.shape == (p.n_state,)
        np.testing.assert_allclose(np.abs(sv.psi), 1.0)
        np.testing.assert_allclose(sv.phi_z, 0.0)

    def test_phi_z_raises_for_2d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        sv = StateVector.uniform_superconducting(p)
        with pytest.raises(AttributeError):
            _ = sv.phi_z

    def test_from_components_roundtrip(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=3)
        n = p.n_interior
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        phi_x = rng.standard_normal(n) + 0j
        phi_y = rng.standard_normal(n) + 0j
        phi_z = rng.standard_normal(n) + 0j

        sv = StateVector.from_components(psi, phi_x, phi_y, phi_z, p)
        np.testing.assert_allclose(sv.psi, psi)
        np.testing.assert_allclose(sv.phi_x, phi_x)
        np.testing.assert_allclose(sv.phi_y, phi_y)
        np.testing.assert_allclose(sv.phi_z, phi_z)

    def test_setter(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        sv = StateVector.uniform_superconducting(p)
        new_psi = np.zeros(p.n_interior, dtype=np.complex128)
        sv.psi = new_psi
        np.testing.assert_allclose(sv.psi, 0.0)

    def test_copy_independent(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        sv = StateVector.uniform_superconducting(p)
        sv2 = sv.copy()
        sv2.psi[:] = 0
        np.testing.assert_allclose(np.abs(sv.psi), 1.0)

    def test_wrong_size_raises(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        with pytest.raises(ValueError):
            StateVector(np.zeros(10, dtype=np.complex128), p)
