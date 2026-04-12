"""Tests for grid index construction."""

import numpy as np
import pytest
from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.mesh.indices import GridIndices, construct_indices


class TestConstructIndices:
    def test_M2_count_2d(self):
        """M2 should contain exactly (Nx-1)*(Ny-1) interior nodes for 2-D."""
        p = SimulationParameters(Nx=6, Ny=6, Nz=1)
        idx = construct_indices(p)
        assert len(idx.interior_to_full) == (p.Nx - 1) * (p.Ny - 1)

    def test_M2_count_3d(self):
        """M2 should contain exactly (Nx-1)*(Ny-1)*(Nz-1) interior nodes for 3-D."""
        p = SimulationParameters(Nx=5, Ny=5, Nz=5)
        idx = construct_indices(p)
        assert len(idx.interior_to_full) == (p.Nx - 1) * (p.Ny - 1) * (p.Nz - 1)

    def test_M2_values_2d(self):
        """Interior indices should be within [0, dim_x) and unique."""
        p = SimulationParameters(Nx=4, Ny=4, Nz=1)
        idx = construct_indices(p)
        assert np.all(idx.interior_to_full >= 0)
        assert np.all(idx.interior_to_full < p.dim_x)
        assert len(np.unique(idx.interior_to_full)) == len(idx.interior_to_full)

    def test_M2_values_3d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=4)
        idx = construct_indices(p)
        assert np.all(idx.interior_to_full >= 0)
        assert np.all(idx.interior_to_full < p.dim_x)
        assert len(np.unique(idx.interior_to_full)) == len(idx.interior_to_full)

    def test_boundary_index_counts_2d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        idx = construct_indices(p)
        # x-boundary: i==0 happens once per (j,k) → (Ny+1)*klim rows
        assert len(idx.x_face_lo) == (p.Ny + 1) * 1
        # y-boundary: j==0 happens once per (i,k)
        assert len(idx.y_face_lo) == (p.Nx + 1) * 1

    def test_boundary_int_indices_are_subset_2d(self):
        p = SimulationParameters(Nx=6, Ny=6, Nz=1)
        idx = construct_indices(p)
        assert set(idx.x_face_lo_inner).issubset(set(idx.x_face_lo))
        assert set(idx.y_face_lo_inner).issubset(set(idx.y_face_lo))

    def test_z_boundary_indices_3d(self):
        p = SimulationParameters(Nx=4, Ny=4, Nz=4)
        idx = construct_indices(p)
        assert len(idx.z_face_lo) > 0
        assert len(idx.z_face_lo_inner) > 0

    def test_M2B_count_2d(self):
        p = SimulationParameters(Nx=6, Ny=6, Nz=1)
        idx = construct_indices(p)
        assert len(idx.bfield_interior) == (p.Nx - 2) * (p.Ny - 2)

    def test_M2B_count_3d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=5)
        idx = construct_indices(p)
        assert len(idx.bfield_interior) == (p.Nx - 2) * (p.Ny - 2) * (p.Nz - 2)

    def test_small_grid(self):
        """Smallest allowed grid (Nx=Ny=2)."""
        p = SimulationParameters(Nx=2, Ny=2, Nz=1)
        idx = construct_indices(p)
        assert len(idx.interior_to_full) == 1  # (2-1)*(2-1) = 1

    def test_no_z_indices_for_2d(self):
        p = SimulationParameters(Nx=5, Ny=5, Nz=1)
        idx = construct_indices(p)
        assert len(idx.z_face_lo) == 0
        assert len(idx.z_face_lo_inner) == 0
