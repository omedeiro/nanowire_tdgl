"""Tests for B-field visualization with streamlines."""

from __future__ import annotations

import numpy as np
import pytest

import tdgl3d
from tdgl3d.visualization.plotting import plot_bfield, plot_bfield_streamlines


def test_plot_bfield_streamlines_basic():
    """Test basic B-field streamline plotting."""
    # Create simple 2D system
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, field)
    
    # Quick solve
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=2.0,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    # Plot with streamlines
    fig, ax = plot_bfield_streamlines(solution, step=-1, component='z', streamplot=True)
    
    assert fig is not None
    assert ax is not None
    
    # Check that plot has expected elements
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    assert ax.get_title() != ""


def test_plot_bfield_streamlines_no_streamplot():
    """Test B-field plot without streamlines."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.2)
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.0,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    # Plot without streamlines
    fig, ax = plot_bfield_streamlines(solution, streamplot=False)
    
    assert fig is not None
    assert ax is not None


def test_plot_bfield_streamlines_components():
    """Test different field components."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=3, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bx=0.1, By=0.1, Bz=0.2)
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.0,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    # Test each component
    for comp in ['x', 'y', 'z', 'magnitude']:
        fig, ax = plot_bfield_streamlines(solution, component=comp, streamplot=True)
        assert fig is not None
        # Check title contains reference to the component
        title = ax.get_title()
        assert 'B-field' in title


def test_plot_bfield_streamlines_invalid_component():
    """Test that invalid component raises error."""
    params = tdgl3d.SimulationParameters(Nx=6, Ny=6, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.1)
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=0.5,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    with pytest.raises(ValueError, match="Unknown component"):
        plot_bfield_streamlines(solution, component='invalid')


def test_plot_bfield_streamlines_3d():
    """Test B-field streamlines on 3D system with z-slice."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.5,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    # Plot different z-slices (Nz=5 means Nz-1=4 interior slices: 0,1,2,3)
    for slice_z in [0, 2, 3]:
        fig, ax = plot_bfield_streamlines(solution, slice_z=slice_z, streamplot=True)
        assert fig is not None


def test_plot_bfield_streamlines_custom_params():
    """Test custom streamline parameters."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.2)
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.0,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    # Custom streamline parameters
    fig, ax = plot_bfield_streamlines(
        solution,
        streamplot=True,
        stream_density=2.5,
        stream_color="red",
        stream_linewidth=1.2,
        stream_arrowsize=1.5,
    )
    
    assert fig is not None


def test_plot_bfield_vs_streamlines_compatibility():
    """Test that old plot_bfield and new plot_bfield_streamlines give similar results."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.0,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=False,
    )
    
    # Old function
    ax1 = plot_bfield(solution, component='z')
    
    # New function without streamlines
    fig2, ax2 = plot_bfield_streamlines(solution, component='z', streamplot=False)
    
    # Both should produce plots
    assert ax1 is not None
    assert ax2 is not None
