"""Tests for sparse operator construction."""

import numpy as np
import pytest
import scipy.sparse as sp
from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.operators.sparse_operators import (
    construct_FPHI_x,
    construct_FPHI_y,
    construct_FPHI_z,
    construct_FPSI,
    construct_LPHI_x,
    construct_LPHI_y,
    construct_LPHI_z,
    construct_LPSI_x,
    construct_LPSI_y,
    construct_LPSI_z,
)


@pytest.fixture
def setup_2d():
    p = SimulationParameters(Nx=6, Ny=6, Nz=1, kappa=2.0)
    idx = construct_indices(p)
    n_full = p.dim_x
    y = np.zeros(n_full, dtype=np.complex128)
    x = np.ones(n_full, dtype=np.complex128)
    return p, idx, x, y


@pytest.fixture
def setup_3d():
    p = SimulationParameters(Nx=5, Ny=5, Nz=5, kappa=3.0)
    idx = construct_indices(p)
    n_full = p.dim_x
    y = np.zeros(n_full, dtype=np.complex128)
    x = np.ones(n_full, dtype=np.complex128)
    return p, idx, x, y


class TestLPSI:
    def test_shape_2d(self, setup_2d):
        p, idx, x, y = setup_2d
        L = construct_LPSI_x(y, p, idx)
        assert L.shape == (p.dim_x, p.dim_x)

    def test_shape_3d(self, setup_3d):
        p, idx, x, y = setup_3d
        L = construct_LPSI_x(y, p, idx)
        assert L.shape == (p.dim_x, p.dim_x)

    def test_zero_gauge_is_real_laplacian_2d(self, setup_2d):
        """With zero link variables, LPSI_x should be the standard finite-difference Laplacian."""
        p, idx, x, y = setup_2d
        L = construct_LPSI_x(y, p, idx)
        # Check interior rows: -2 on diagonal, +1 on neighbours
        m = idx.interior_to_full
        L_interior = L[m, :]
        diag = np.array(L_interior[:, m].todense())
        np.testing.assert_allclose(np.diag(diag), -2.0)

    def test_lpsi_y_shape(self, setup_2d):
        p, idx, x, y = setup_2d
        L = construct_LPSI_y(y, p, idx)
        assert L.shape == (p.dim_x, p.dim_x)

    def test_lpsi_z_shape(self, setup_3d):
        p, idx, x, y = setup_3d
        L = construct_LPSI_z(y, p, idx)
        assert L.shape == (p.dim_x, p.dim_x)


class TestLPHI:
    def test_shape_2d(self, setup_2d):
        p, idx, x, y = setup_2d
        Lx = construct_LPHI_x(p, idx)
        Ly = construct_LPHI_y(p, idx)
        Lz = construct_LPHI_z(p, idx)
        for L in [Lx, Ly, Lz]:
            assert L.shape == (p.dim_x, p.dim_x)

    def test_diagonal_values(self, setup_3d):
        p, idx, x, y = setup_3d
        Lx = construct_LPHI_x(p, idx)
        m = idx.interior_to_full
        expected_diag = -2.0 * (p.kappa**2 / p.hy**2 + p.kappa**2 / p.hz**2)
        diag_vals = np.asarray(Lx[m, m]).flatten()
        np.testing.assert_allclose(diag_vals, expected_diag)


class TestFPSI:
    def test_zero_for_uniform(self, setup_2d):
        """(1 - |ψ|²)ψ = 0 when |ψ| = 1."""
        p, idx, x, y = setup_2d
        F = construct_FPSI(x, p, idx)
        np.testing.assert_allclose(F, 0.0, atol=1e-14)

    def test_nonzero_for_half(self, setup_2d):
        """(1 - 0.25)*0.5 = 0.375 when ψ = 0.5."""
        p, idx, x, y = setup_2d
        x_half = x * 0.5
        F = construct_FPSI(x_half, p, idx)
        expected = (1.0 - 0.25) * 0.5
        np.testing.assert_allclose(F, expected, atol=1e-14)


class TestFPHI:
    def test_fphi_x_shape_2d(self, setup_2d):
        p, idx, x, y = setup_2d
        F = construct_FPHI_x(x, y, y, y, p, idx)
        assert F.shape == (p.n_interior,)

    def test_fphi_y_shape_3d(self, setup_3d):
        p, idx, x, y = setup_3d
        F = construct_FPHI_y(x, y, y, y, p, idx)
        assert F.shape == (p.n_interior,)

    def test_fphi_z_zero_for_2d(self, setup_2d):
        p, idx, x, y = setup_2d
        F = construct_FPHI_z(x, y, y, y, p, idx)
        np.testing.assert_allclose(F, 0.0)
