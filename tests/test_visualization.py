"""Tests for visualization utilities."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless runs

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tdgl3d
from tdgl3d.core.solution import Solution
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.visualization.plotting import (
    _grid_coords_2d,
    animate,
    plot_bfield,
    plot_order_parameter,
    plot_summary,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def solution_2d() -> Solution:
    """Run a short 2-D Forward-Euler simulation and return a Solution."""
    params = tdgl3d.SimulationParameters(
        Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=5.0,
    )
    field = tdgl3d.AppliedField(Bz=0.3, t_on_fraction=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    sol = tdgl3d.solve(
        device, t_start=0.0, t_stop=1.0, dt=0.02,
        method="euler", save_every=5, progress=False,
    )
    return sol


@pytest.fixture()
def solution_3d() -> Solution:
    """Run a short 3-D Forward-Euler simulation and return a Solution."""
    params = tdgl3d.SimulationParameters(
        Nx=5, Ny=5, Nz=3, hx=1.0, hy=1.0, hz=1.0, kappa=5.0,
    )
    field = tdgl3d.AppliedField(Bz=0.2)
    device = tdgl3d.Device(params, applied_field=field)
    sol = tdgl3d.solve(
        device, t_start=0.0, t_stop=0.5, dt=0.02,
        method="euler", save_every=5, progress=False,
    )
    return sol


# ---------------------------------------------------------------------------
# _grid_coords_2d
# ---------------------------------------------------------------------------

class TestGridCoords:
    def test_shape(self):
        params = tdgl3d.SimulationParameters(Nx=6, Ny=8, Nz=1)
        xx, yy = _grid_coords_2d(params)
        assert xx.shape == (5, 7)
        assert yy.shape == (5, 7)

    def test_values(self):
        params = tdgl3d.SimulationParameters(Nx=4, Ny=4, Nz=1, hx=2.0, hy=3.0)
        xx, yy = _grid_coords_2d(params)
        np.testing.assert_allclose(xx[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(yy[0, :], [3.0, 6.0, 9.0])


# ---------------------------------------------------------------------------
# plot_order_parameter
# ---------------------------------------------------------------------------

class TestPlotOrderParameter:
    def test_returns_axes(self, solution_2d):
        ax = plot_order_parameter(solution_2d, step=0)
        assert ax is not None
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_custom_title(self, solution_2d):
        ax = plot_order_parameter(solution_2d, step=0, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close("all")

    def test_custom_cmap(self, solution_2d):
        ax = plot_order_parameter(solution_2d, step=-1, cmap="viridis")
        assert ax is not None
        plt.close("all")

    def test_with_provided_axes(self, solution_2d):
        fig, ax = plt.subplots()
        returned_ax = plot_order_parameter(solution_2d, step=0, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_last_step(self, solution_2d):
        ax = plot_order_parameter(solution_2d, step=-1)
        assert "|ψ|²" in ax.get_title()
        plt.close("all")

    def test_3d_slice(self, solution_3d):
        ax = plot_order_parameter(solution_3d, step=-1, slice_z=0)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_bfield
# ---------------------------------------------------------------------------

class TestPlotBfield:
    def test_returns_axes(self, solution_2d):
        ax = plot_bfield(solution_2d, component="z", step=-1)
        assert ax is not None
        plt.close("all")

    def test_component_in_title(self, solution_2d):
        for comp in ("x", "y", "z"):
            ax = plot_bfield(solution_2d, component=comp, step=-1)
            assert f"B_{comp}" in ax.get_title()
            plt.close("all")

    def test_with_provided_axes(self, solution_2d):
        fig, ax = plt.subplots()
        returned_ax = plot_bfield(solution_2d, component="z", step=-1, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_3d_slice(self, solution_3d):
        ax = plot_bfield(solution_3d, component="z", step=-1, slice_z=0)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_summary
# ---------------------------------------------------------------------------

class TestPlotSummary:
    def test_returns_figure(self, solution_2d):
        fig = plot_summary(solution_2d, step=-1)
        assert fig is not None
        axes = fig.get_axes()
        # Should have 2 main axes + 2 colorbars = 4
        assert len(axes) >= 2
        plt.close("all")

    def test_custom_figsize(self, solution_2d):
        fig = plot_summary(solution_2d, figsize=(10, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1
        assert abs(h - 4) < 0.1
        plt.close("all")

    def test_3d_slice(self, solution_3d):
        fig = plot_summary(solution_3d, step=-1, slice_z=1)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# animate
# ---------------------------------------------------------------------------

class TestAnimate:
    def test_creates_gif(self, solution_2d, tmp_path):
        gif_path = tmp_path / "test_anim.gif"
        result = animate(solution_2d, filename=gif_path, fps=5, step_stride=2)
        assert Path(result).exists()
        assert os.path.getsize(result) > 0
        plt.close("all")


# ---------------------------------------------------------------------------
# Savefig smoke test
# ---------------------------------------------------------------------------

class TestSaveFig:
    def test_save_png(self, solution_2d, tmp_path):
        ax = plot_order_parameter(solution_2d, step=-1)
        out = tmp_path / "test_plot.png"
        ax.get_figure().savefig(str(out), dpi=72, bbox_inches="tight")
        assert out.exists()
        assert os.path.getsize(out) > 100
        plt.close("all")
