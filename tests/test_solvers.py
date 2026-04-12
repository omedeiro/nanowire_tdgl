"""Tests for the linear solvers (TGCR, Newton)."""

import numpy as np
import pytest
from tdgl3d.solvers.tgcr import tgcr_matrix_free, tgcr_matrix_free_trap
from tdgl3d.solvers.newton import newton_gcr, newton_gcr_trap


class TestTGCR:
    def test_solve_identity(self):
        """Solve I * x = b (identity Jacobian → f(x) = x)."""
        b = np.array([1.0, 2.0, 3.0], dtype=np.complex128)

        def f(x):
            return x  # Jacobian = I

        x = tgcr_matrix_free(f, np.zeros(3, dtype=np.complex128), b, tol=1e-8)
        np.testing.assert_allclose(x, b, atol=1e-6)

    def test_solve_scaled(self):
        """Solve 2I * x = b → x = b/2."""
        b = np.array([4.0, 6.0], dtype=np.complex128)

        def f(x):
            return 2.0 * x

        x = tgcr_matrix_free(f, np.zeros(2, dtype=np.complex128), b, tol=1e-8)
        np.testing.assert_allclose(x, b / 2.0, atol=1e-5)

    def test_nonconvergence_returns_empty(self):
        """With max_iter=0 should not converge."""
        b = np.array([1.0, 2.0], dtype=np.complex128)
        x = tgcr_matrix_free(lambda x: x, np.zeros(2, dtype=np.complex128), b, max_iter=0)
        assert x.size == 0


class TestTGCRTrap:
    def test_trap_identity(self):
        """With f(x)=x and dt=0, system is I*δx = b."""
        b = np.array([1.0, 2.0], dtype=np.complex128)
        x = tgcr_matrix_free_trap(
            lambda x: x, np.zeros(2, dtype=np.complex128), b, dt=0.0, tol=1e-8,
        )
        np.testing.assert_allclose(x, b, atol=1e-5)


class TestNewtonGCR:
    def test_find_root_simple(self):
        """Solve f(x) = x - [1, 2] = 0 → x = [1, 2]."""
        target = np.array([1.0, 2.0], dtype=np.complex128)

        def f(x):
            return x - target

        x0 = np.zeros(2, dtype=np.complex128)
        x, converged, iters = newton_gcr(f, x0, tol_f=1e-6, tol_dx=1e-6)
        assert converged
        np.testing.assert_allclose(x, target, atol=1e-4)

    def test_quadratic_root(self):
        """Solve f(x) = x² - 4 = 0 starting from x0=3 → x≈2."""
        def f(x):
            return x**2 - 4.0

        x0 = np.array([3.0], dtype=np.complex128)
        x, converged, iters = newton_gcr(f, x0, tol_f=1e-8, tol_dx=1e-8)
        assert converged
        np.testing.assert_allclose(np.abs(x), 2.0, atol=1e-3)


class TestNewtonGCRTrap:
    def test_trap_trivial(self):
        """Trapezoidal Newton with f(x)=0 and gamma=x0 → x=x0."""
        x0 = np.array([1.0, 2.0], dtype=np.complex128)
        gamma = x0.copy()

        def f(x):
            return np.zeros_like(x)

        x, converged, iters = newton_gcr_trap(f, x0, gamma, dt=1.0, tol_f=1e-6, tol_dx=1e-6)
        assert converged
        np.testing.assert_allclose(x, x0, atol=1e-4)
