"""Smoke tests for figure generation scripts.

Each test imports a figure script, runs it with small geometry (small=True),
and verifies that the expected PNG files are created and non-empty.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

FIGURES_DIR = Path(__file__).parent


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, FIGURES_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPTS = [
    ("meissner_screening", ["meissner_screening.png"]),
    ("vortex_entry", ["vortex_entry.png"]),
    ("hole_field_penetration", ["hole_field_penetration.png"]),
    ("supercurrent_hole", ["supercurrent_hole.png"]),
    ("trilayer_bfield", ["trilayer_bfield.png"]),
    ("insulator_psi_decay", ["insulator_psi_decay.png"]),
    ("energy_dissipation", ["energy_dissipation.png"]),
    ("phase_winding", ["phase_winding.png"]),
    ("cfl_instability", ["cfl_instability.png"]),
    ("hole_bc_verification", [
        "hole_bc_verification_time_evolution.png",
        "hole_bc_verification_current.png",
        "hole_bc_verification_psi.png",
        "hole_bc_verification_crosssection.png",
    ]),
    ("vortex_entry_dynamics", ["vortex_entry_dynamics.gif"]),
    ("analytic_cross_sections", ["analytic_cross_sections.png"]),
]


@pytest.mark.parametrize("script_name,expected_files", SCRIPTS,
                         ids=[s[0] for s in SCRIPTS])
def test_figure_smoke(script_name, expected_files, tmp_path):
    mod = _load_module(script_name)
    saved = mod.main(output_dir=tmp_path, small=True)
    saved_names = [p.name for p in saved]
    for fname in expected_files:
        assert fname in saved_names, f"{fname} not produced by {script_name}"
        p = tmp_path / fname
        assert p.exists(), f"{p} does not exist"
        assert p.stat().st_size > 0, f"{p} is empty"
