"""Tests for run logging and metadata tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

import tdgl3d
from tdgl3d.io.logging import RunMetadata, TimingContext, create_run_metadata


def test_timing_context():
    """Test TimingContext measures elapsed time."""
    import time
    
    with TimingContext() as timer:
        time.sleep(0.1)
    
    assert timer.elapsed >= 0.1
    assert timer.elapsed < 0.2  # Should be quick


def test_timing_context_still_running():
    """Test elapsed property works during execution."""
    import time
    
    with TimingContext() as timer:
        time.sleep(0.05)
        elapsed_mid = timer.elapsed
        assert elapsed_mid >= 0.05
    
    assert timer.elapsed > elapsed_mid


def test_run_metadata_creation():
    """Test creating RunMetadata from simulation config."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=2, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.5)
    device = tdgl3d.Device(params, field)
    
    metadata = create_run_metadata(
        params=params,
        device=device,
        method="trapezoidal",
        dt=0.05,
        t_final=10.0,
        wall_time=5.2,
        atol=1e-3,
        rtol=1e-3,
        total_steps=100,
    )
    
    assert metadata.wall_time_seconds == 5.2
    assert metadata.parameters['Nx'] == 10
    assert metadata.parameters['kappa'] == 2.0
    assert metadata.solver_config['method'] == 'trapezoidal'
    assert metadata.solver_config['dt_initial'] == 0.05
    assert metadata.solver_config['atol'] == 1e-3
    assert metadata.performance_metrics['total_timesteps'] == 100


def test_run_metadata_serialization():
    """Test RunMetadata can be converted to/from dict."""
    params = tdgl3d.SimulationParameters(Nx=5, Ny=5, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, field)
    
    metadata = create_run_metadata(
        params=params,
        device=device,
        method="euler",
        dt=0.01,
        t_final=5.0,
        wall_time=2.5,
    )
    
    # Convert to dict
    data = metadata.to_dict()
    assert isinstance(data, dict)
    assert 'timestamp' in data
    assert 'wall_time_seconds' in data
    
    # Reconstruct from dict
    metadata2 = RunMetadata.from_dict(data)
    assert metadata2.wall_time_seconds == metadata.wall_time_seconds
    assert metadata2.parameters == metadata.parameters


def test_run_metadata_json_save_load(tmp_path):
    """Test saving and loading metadata to JSON file."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=3, hx=1.0, hy=1.0, hz=1.0, kappa=3.0)
    field = tdgl3d.AppliedField(Bz=0.4)
    device = tdgl3d.Device(params, field)
    
    metadata = create_run_metadata(
        params=params,
        device=device,
        method="trapezoidal",
        dt=0.02,
        t_final=8.0,
        wall_time=3.7,
    )
    
    # Save to JSON
    json_file = tmp_path / "test_metadata.json"
    metadata.save_json(json_file)
    
    assert json_file.exists()
    
    # Load back
    metadata_loaded = RunMetadata.load_json(json_file)
    
    assert metadata_loaded.wall_time_seconds == 3.7
    assert metadata_loaded.parameters['Nx'] == 8
    assert metadata_loaded.parameters['kappa'] == 3.0
    assert metadata_loaded.solver_config['method'] == 'trapezoidal'


def test_solve_creates_metadata():
    """Test that solve() populates solution.metadata."""
    params = tdgl3d.SimulationParameters(Nx=6, Ny=6, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.0)  # Zero field for fast solve
    device = tdgl3d.Device(params, field)
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.0,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=True,
    )
    
    assert solution.metadata is not None
    assert 'wall_time_seconds' in solution.metadata
    assert solution.metadata['wall_time_seconds'] > 0
    assert 'parameters' in solution.metadata
    assert solution.metadata['parameters']['Nx'] == 6
    assert solution.metadata['parameters']['kappa'] == 2.0
    assert solution.metadata['solver_config']['method'] == 'euler'


def test_solve_metadata_disabled():
    """Test that solve() respects log_metadata=False."""
    params = tdgl3d.SimulationParameters(Nx=4, Ny=4, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.0)
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
    
    assert solution.metadata is None


def test_solve_auto_saves_json(tmp_path):
    """Test that solve() auto-saves metadata JSON file."""
    params = tdgl3d.SimulationParameters(Nx=5, Ny=5, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.0)
    device = tdgl3d.Device(params, field)
    
    log_dir = tmp_path / "test_logs"
    
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=0.5,
        dt=0.1,
        method="euler",
        progress=False,
        log_metadata=True,
        log_dir=log_dir,
    )
    
    # Check that JSON file was created
    assert log_dir.exists()
    json_files = list(log_dir.glob("run_*.json"))
    assert len(json_files) == 1
    
    # Verify content
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    assert data['parameters']['Nx'] == 5
    assert data['solver_config']['method'] == 'euler'


def test_metadata_with_trilayer():
    """Test metadata captures trilayer configuration."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=8, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=3, kappa=2.5),
        insulator=tdgl3d.Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=3, kappa=2.5),
    )
    
    device = tdgl3d.Device(params, field, trilayer=trilayer)
    
    metadata = create_run_metadata(
        params=params,
        device=device,
        method="trapezoidal",
        dt=0.05,
        t_final=10.0,
        wall_time=8.5,
    )
    
    assert metadata.device_config['has_trilayer'] is True
    assert 'trilayer' in metadata.device_config
    assert metadata.device_config['trilayer']['bottom_thickness'] == 3
    assert metadata.device_config['trilayer']['insulator_thickness'] == 2
    assert metadata.device_config['trilayer']['top_kappa'] == 2.5


def test_metadata_git_commit():
    """Test that git commit hash is captured if available."""
    params = tdgl3d.SimulationParameters(Nx=5, Ny=5, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.0)
    device = tdgl3d.Device(params, field)
    
    metadata = create_run_metadata(
        params=params,
        device=device,
        method="euler",
        dt=0.1,
        t_final=1.0,
        wall_time=1.0,
    )
    
    # Git commit may be None if not in a repo, or a hex string
    if metadata.git_commit is not None:
        assert isinstance(metadata.git_commit, str)
        assert len(metadata.git_commit) == 40  # SHA-1 hash length
