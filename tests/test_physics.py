"""Tests for the RHS evaluation (eval_f) and physics modules."""

import numpy as np
import pytest
from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.core.state import StateVector
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.physics.rhs import BoundaryVectors, eval_f
from tdgl3d.physics.bfield import eval_bfield
from tdgl3d.physics.applied_field import AppliedField, build_boundary_field_vectors


def _zero_bv(params, idx):
    """Create zero boundary vectors."""
    N = params.dim_x
    return BoundaryVectors(np.zeros(N), np.zeros(N), np.zeros(N))


class TestEvalF:
    def test_output_shape_2d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        u = _zero_bv(p, idx)
        F = eval_f(sv.data, p, idx, u)
        assert F.shape == (p.n_state,)

    def test_output_shape_3d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=4)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        u = _zero_bv(p, idx)
        F = eval_f(sv.data, p, idx, u)
        assert F.shape == (p.n_state,)

    def test_steady_state_uniform_periodic(self):
        """A uniform |ψ|=1 state with zero gauge, zero field, and fully periodic BCs
        should be at exact equilibrium: dX/dt = 0.

        With periodic BCs the Laplacian of a constant is exactly zero, and
        the nonlinear forcing (1-|ψ|²)ψ = 0 when |ψ|=1.
        """
        p = SimulationParameters(
            Nx=6, Ny=6, Nz=1,
            periodic_x=True, periodic_y=True, periodic_z=True,
        )
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        u = _zero_bv(p, idx)
        F = eval_f(sv.data, p, idx, u)
        # With periodic BCs everything should cancel perfectly
        np.testing.assert_allclose(F, 0.0, atol=1e-12)

    def test_eval_f_is_complex(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        u = _zero_bv(p, idx)
        F = eval_f(sv.data, p, idx, u)
        assert F.dtype == np.complex128

    def test_c4_symmetry_with_applied_bz(self):
        """On a square grid with applied Bz, the initial RHS must respect
        C4 rotation symmetry: dφ_x(i,j)/dt = −dφ_y(j,i)/dt."""
        p = SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, kappa=2.0)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        Bx_v, By_v, Bz_v = build_boundary_field_vectors(0, 0, 0.5, p, idx)
        u = BoundaryVectors(Bx_v, By_v, Bz_v)
        F = eval_f(sv.data, p, idx, u)
        n = p.n_interior
        phix = F[n:2 * n].reshape(p.Nx - 1, p.Ny - 1)
        phiy = F[2 * n:3 * n].reshape(p.Nx - 1, p.Ny - 1)
        np.testing.assert_allclose(phix, -phiy.T, atol=1e-12)


class TestBfield:
    def test_bfield_shape_2d(self):
        p = SimulationParameters(Nx=6, Ny=6, Nz=1)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        Bx, By, Bz = eval_bfield(sv.data, p, idx)
        n2 = (p.Nx - 2) * (p.Ny - 2)
        assert Bx.shape == (n2,)
        assert By.shape == (n2,)
        assert Bz.shape == (n2,)

    def test_bfield_shape_3d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=5)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        Bx, By, Bz = eval_bfield(sv.data, p, idx)
        n2 = (p.Nx - 2) * (p.Ny - 2) * (p.Nz - 2)
        assert len(Bx) == n2

    def test_bfield_zero_for_zero_gauge(self):
        """B = curl(A) = 0 when all link variables are zero."""
        p = SimulationParameters(Nx=6, Ny=6, Nz=1)
        idx = construct_indices(p)
        sv = StateVector.uniform_superconducting(p)
        Bx, By, Bz = eval_bfield(sv.data, p, idx)
        np.testing.assert_allclose(Bz, 0.0, atol=1e-14)


class TestAppliedField:
    def test_evaluate_on_off(self):
        af = AppliedField(Bz=1.0, t_on_fraction=0.5)
        bx, by, bz = af.evaluate(0.3, 1.0)
        assert bz == 1.0
        bx, by, bz = af.evaluate(0.6, 1.0)
        assert bz == 0.0

    def test_custom_field_func(self):
        af = AppliedField(field_func=lambda t, ts: (0.0, 0.0, t / ts))
        bx, by, bz = af.evaluate(0.5, 1.0)
        assert bz == pytest.approx(0.5)

    def test_build_boundary_vectors(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        idx = construct_indices(p)
        Bx, By, Bz = build_boundary_field_vectors(0.0, 0.0, 1.0, p, idx)
        # Bz should be nonzero at x and y boundary faces
        assert np.any(Bz != 0)
        # Bx and By should be zero
        np.testing.assert_allclose(Bx, 0.0)
        np.testing.assert_allclose(By, 0.0)
