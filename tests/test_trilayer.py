"""Tests for the S/I/S trilayer material system."""

from __future__ import annotations

import numpy as np
import pytest

from tdgl3d import (
    Device,
    Layer,
    MaterialMap,
    SimulationParameters,
    Trilayer,
    AppliedField,
    solve,
)
from tdgl3d.core.material import build_material_map
from tdgl3d.mesh.indices import construct_indices


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

def _make_trilayer(kappa_sc=2.0, kappa_ins=0.0,
                   t_bot=2, t_ins=1, t_top=2):
    """Helper: build a symmetric trilayer."""
    return Trilayer(
        bottom=Layer(thickness_z=t_bot, kappa=kappa_sc),
        insulator=Layer(thickness_z=t_ins, kappa=kappa_ins, is_superconductor=False),
        top=Layer(thickness_z=t_top, kappa=kappa_sc),
    )


@pytest.fixture
def trilayer():
    return _make_trilayer()


@pytest.fixture
def params(trilayer):
    """4×4 lateral grid, Nz from trilayer (=5)."""
    return SimulationParameters(Nx=4, Ny=4, Nz=trilayer.Nz, kappa=2.0)


@pytest.fixture
def idx(params):
    return construct_indices(params)


@pytest.fixture
def material(params, trilayer, idx):
    return build_material_map(params, trilayer, idx)


# ────────────────────────────────────────────────────────────
# Layer / Trilayer
# ────────────────────────────────────────────────────────────

class TestLayerTrilayer:
    def test_trilayer_Nz(self, trilayer):
        assert trilayer.Nz == 5  # 2 + 1 + 2

    def test_z_ranges(self, trilayer):
        r = trilayer.z_ranges()
        assert r["bottom"] == (0, 2)
        assert r["insulator"] == (2, 3)
        assert r["top"] == (3, 5)

    def test_insulator_must_be_non_sc(self):
        with pytest.raises(ValueError, match="is_superconductor"):
            Trilayer(
                bottom=Layer(2, 2.0),
                insulator=Layer(1, 0.0, is_superconductor=True),  # wrong!
                top=Layer(2, 2.0),
            )

    def test_asymmetric_Nz(self):
        tri = _make_trilayer(t_bot=3, t_ins=2, t_top=4)
        assert tri.Nz == 9


# ────────────────────────────────────────────────────────────
# MaterialMap construction
# ────────────────────────────────────────────────────────────

class TestMaterialMap:
    def test_shapes(self, material, params):
        assert material.kappa.shape == (params.dim_x,)
        assert material.sc_mask.shape == (params.dim_x,)

    def test_interior_sc_mask_shape(self, material, idx):
        assert material.interior_sc_mask.shape == (len(idx.interior_to_full),)

    def test_kappa_values(self, material, params, trilayer):
        """κ = 2.0 in SC planes, 0.0 in insulator plane."""
        Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
        mk = (Nx + 1) * (Ny + 1)

        # Bottom SC: k = 0, 1
        for k in range(2):
            plane = material.kappa[k * mk : (k + 1) * mk]
            np.testing.assert_allclose(plane, 2.0)

        # Insulator: k = 2
        plane = material.kappa[2 * mk : 3 * mk]
        np.testing.assert_allclose(plane, 0.0)

        # Top SC: k = 3, 4, 5
        for k in range(3, Nz + 1):
            plane = material.kappa[k * mk : (k + 1) * mk]
            np.testing.assert_allclose(plane, 2.0)

    def test_sc_mask_values(self, material, params):
        """SC planes → 1.0, insulator plane → 0.0."""
        Nx, Ny = params.Nx, params.Ny
        mk = (Nx + 1) * (Ny + 1)

        # Bottom SC
        for k in range(2):
            plane = material.sc_mask[k * mk : (k + 1) * mk]
            np.testing.assert_allclose(plane, 1.0)

        # Insulator
        plane = material.sc_mask[2 * mk : 3 * mk]
        np.testing.assert_allclose(plane, 0.0)

        # Top SC
        for k in range(3, params.Nz + 1):
            plane = material.sc_mask[k * mk : (k + 1) * mk]
            np.testing.assert_allclose(plane, 1.0)

    def test_Nz_mismatch_raises(self, trilayer):
        bad_params = SimulationParameters(Nx=4, Ny=4, Nz=10)
        idx_bad = construct_indices(bad_params)
        with pytest.raises(ValueError, match="does not match"):
            build_material_map(bad_params, trilayer, idx_bad)


# ────────────────────────────────────────────────────────────
# Device with trilayer
# ────────────────────────────────────────────────────────────

class TestTrilayerDevice:
    def test_device_creates_material(self, trilayer):
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, kappa=2.0),
            trilayer=trilayer,
        )
        assert dev.material is not None
        assert isinstance(dev.material, MaterialMap)

    def test_device_auto_sets_Nz(self, trilayer):
        """If params.Nz != trilayer.Nz, device should override it."""
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, Nz=99, kappa=2.0),
            trilayer=trilayer,
        )
        assert dev.params.Nz == trilayer.Nz

    def test_initial_state_insulator_zero(self, trilayer):
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, kappa=2.0),
            trilayer=trilayer,
        )
        sv = dev.initial_state()

        # All interior ψ at insulator nodes should be 0
        mat = dev.material
        psi = sv.psi
        insulator_interior = mat.interior_sc_mask == 0.0
        np.testing.assert_allclose(psi[insulator_interior], 0.0)

        # SC interior ψ should be 1.0
        sc_interior = mat.interior_sc_mask == 1.0
        np.testing.assert_allclose(np.abs(psi[sc_interior]), 1.0)

    def test_repr_contains_trilayer(self, trilayer):
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, kappa=2.0),
            trilayer=trilayer,
        )
        r = repr(dev)
        assert "trilayer" in r
        assert "SC" in r

    def test_no_trilayer_no_material(self):
        dev = Device(params=SimulationParameters(Nx=4, Ny=4, Nz=3))
        assert dev.material is None


# ────────────────────────────────────────────────────────────
# Simulation runs with trilayer
# ────────────────────────────────────────────────────────────

class TestTrilayerSimulation:
    """Smoke tests — just verify the solver chain doesn't crash."""

    def test_euler_trilayer_runs(self, trilayer):
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, kappa=2.0),
            applied_field=AppliedField(Bz=1.0),
            trilayer=trilayer,
        )
        x0 = dev.initial_state()
        sol = solve(
            dev, t_stop=0.1, dt=0.01, method="euler",
            x0=x0, progress=False,
        )
        assert sol.times.shape[0] > 1
        assert np.all(np.isfinite(sol.states))

    def test_trapezoidal_trilayer_runs(self, trilayer):
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, kappa=2.0),
            applied_field=AppliedField(Bz=0.5),
            trilayer=trilayer,
        )
        x0 = dev.initial_state()
        sol = solve(
            dev, t_stop=0.1, dt=0.05, method="trapezoidal",
            x0=x0, progress=False, verbose=False,
        )
        assert sol.times.shape[0] > 1
        assert np.all(np.isfinite(sol.states))

    def test_insulator_psi_stays_small(self, trilayer):
        """After a short Euler run, |ψ| in the insulator should stay near 0."""
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, kappa=2.0),
            applied_field=AppliedField(Bz=0.5),
            trilayer=trilayer,
        )
        x0 = dev.initial_state()
        sol = solve(
            dev, t_stop=0.5, dt=0.01, method="euler",
            x0=x0, progress=False,
        )
        # Get final state ψ
        from tdgl3d.core.state import StateVector
        sv_final = StateVector(sol.states[:, -1], dev.params)
        psi = sv_final.psi

        # Insulator interior indices
        ins_mask = dev.material.interior_sc_mask == 0.0
        psi_ins = np.abs(psi[ins_mask])

        # Should be suppressed (small compared to SC value of ~1.0)
        assert np.all(psi_ins < 0.2), (
            f"Max |ψ| in insulator = {psi_ins.max():.4f}"
        )

    def test_none_material_backward_compat(self):
        """A plain device (no trilayer) should still work fine."""
        dev = Device(
            params=SimulationParameters(Nx=4, Ny=4, Nz=2, kappa=2.0),
            applied_field=AppliedField(Bz=1.0),
        )
        sol = solve(
            dev, t_stop=0.1, dt=0.01, method="euler",
            progress=False,
        )
        assert sol.times.shape[0] > 1
        assert np.all(np.isfinite(sol.states))
