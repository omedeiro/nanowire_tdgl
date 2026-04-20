"""Logging and run metadata tracking for TDGL simulations."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tdgl3d.core.device import Device
from tdgl3d.core.parameters import SimulationParameters


@dataclass
class RunMetadata:
    """Complete metadata for a simulation run.
    
    Attributes:
        timestamp: ISO 8601 timestamp when run started
        wall_time_seconds: Total wall-clock time for simulation
        hostname: Machine hostname
        platform_info: OS and Python version info
        git_commit: Git commit hash (if available)
        parameters: Simulation parameters (grid size, kappa, etc.)
        device_config: Device configuration (applied field, trilayer)
        solver_config: Integration method, timestep, tolerances
        convergence_info: Final convergence metrics (optional)
        vortex_count_final: Number of vortices at final time (optional)
        performance_metrics: Additional timing/memory data
    """
    timestamp: str
    wall_time_seconds: float
    hostname: str
    platform_info: dict[str, str]
    git_commit: Optional[str]
    parameters: dict[str, Any]
    device_config: dict[str, Any]
    solver_config: dict[str, Any]
    convergence_info: Optional[dict[str, float]] = None
    vortex_count_final: Optional[int] = None
    performance_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        """Reconstruct from dictionary."""
        return cls(**data)

    def save_json(self, filepath: str | Path) -> None:
        """Save metadata to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_serializer)
    
    @classmethod
    def load_json(cls, filepath: str | Path) -> RunMetadata:
        """Load metadata from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Reconstructed metadata object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash if in a git repo.
    
    Returns:
        Commit hash as hex string, or None if not in git repo
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def create_run_metadata(
    params: SimulationParameters,
    device: Device,
    method: str,
    dt: float,
    t_final: float,
    wall_time: float,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    **extra_metrics: Any
) -> RunMetadata:
    """Create run metadata from simulation configuration.
    
    Args:
        params: Simulation parameters
        device: Device configuration
        method: Integration method ('forward_euler' or 'trapezoidal')
        dt: Initial timestep
        t_final: Final simulation time
        wall_time: Wall-clock time in seconds
        atol: Absolute tolerance (for adaptive methods)
        rtol: Relative tolerance (for adaptive methods)
        **extra_metrics: Additional performance metrics to record
    
    Returns:
        Complete metadata object
    """
    # Platform info
    platform_info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': platform.python_version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
    
    # Simulation parameters
    params_dict = {
        'Nx': params.Nx,
        'Ny': params.Ny,
        'Nz': params.Nz,
        'hx': params.hx,
        'hy': params.hy,
        'hz': params.hz,
        'kappa': params.kappa,
        'periodic_x': params.periodic_x,
        'periodic_y': params.periodic_y,
        'periodic_z': params.periodic_z,
    }
    
    # Device configuration
    device_config = {
        'has_trilayer': device.trilayer is not None,
    }
    
    if device.trilayer is not None:
        device_config['trilayer'] = {
            'top_thickness': device.trilayer.top.thickness_z,
            'top_kappa': device.trilayer.top.kappa,
            'insulator_thickness': device.trilayer.insulator.thickness_z,
            'bottom_thickness': device.trilayer.bottom.thickness_z,
            'bottom_kappa': device.trilayer.bottom.kappa,
        }
    
    # Applied field info
    if device.applied_field is not None:
        from tdgl3d.physics.applied_field import AppliedField
        field = device.applied_field
        if callable(field):
            device_config['applied_field'] = {
                'type': 'callable',
                'description': str(field)
            }
        else:
            # It's an AppliedField instance
            B_initial = field.evaluate(0.0, 1.0)  # Evaluate at t=0
            device_config['applied_field'] = {
                'type': type(field).__name__,
                'Bx': field.Bx,
                'By': field.By,
                'Bz': field.Bz,
                'B_initial': B_initial,
            }
    
    # Solver configuration
    solver_config = {
        'method': method,
        'dt_initial': dt,
        't_final': t_final,
    }
    
    if atol is not None:
        solver_config['atol'] = atol
    if rtol is not None:
        solver_config['rtol'] = rtol
    
    # Performance metrics
    performance_metrics = {
        'wall_time_per_timestep_avg': extra_metrics.get('wall_time_per_step'),
        'total_timesteps': extra_metrics.get('total_steps'),
        'newton_iterations_total': extra_metrics.get('newton_iterations'),
        'dt_reductions': extra_metrics.get('dt_reductions'),
    }
    # Remove None values
    performance_metrics = {k: v for k, v in performance_metrics.items() if v is not None}
    performance_metrics.update({k: v for k, v in extra_metrics.items() 
                               if k not in performance_metrics})
    
    return RunMetadata(
        timestamp=datetime.now().isoformat(),
        wall_time_seconds=wall_time,
        hostname=platform.node(),
        platform_info=platform_info,
        git_commit=_get_git_commit(),
        parameters=params_dict,
        device_config=device_config,
        solver_config=solver_config,
        performance_metrics=performance_metrics,
    )


def setup_file_logger(
    log_dir: str | Path = "logs",
    log_name: Optional[str] = None,
    level: int = logging.INFO
) -> tuple[logging.Logger, Path]:
    """Set up file-based logger for a simulation run.
    
    Args:
        log_dir: Directory for log files
        log_name: Log file name (auto-generated if None)
        level: Logging level
    
    Returns:
        Tuple of (logger instance, log file path)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"tdgl_run_{timestamp}.log"
    
    log_path = log_dir / log_name
    
    # Create logger
    logger = logging.getLogger(f"tdgl3d.run.{log_name}")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, log_path


class TimingContext:
    """Context manager for timing code blocks.
    
    Example:
        >>> with TimingContext() as timer:
        ...     # do work
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed:.3f} seconds")
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> TimingContext:
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        return False  # Don't suppress exceptions
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        if self.end_time is None:
            # Still running
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
