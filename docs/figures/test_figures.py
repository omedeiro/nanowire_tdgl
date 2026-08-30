"""Smoke tests for figure generation scripts.

Each test imports a figure script, runs it with small geometry (small=True),
and verifies that the expected PNG files are created and non-empty.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

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
    ("nb_hole_array", [
        "nb_hole_array_entry.png",
        "nb_hole_array_trapped.png",
        "nb_hole_array_trapped.gif",
    ]),
    ("sis_vortex_trapping_3d", [
        "sis_vortex_trapping_3d.png",
        "sis_vortex_trapping_sweep.png",
    ]),
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


def test_isometric_sheet_is_drawn_symmetrically():
    """A mirror-symmetric film must be painted as a mirror-symmetric sheet.

    ``plot_surface`` colours the quad *between* nodes ``i`` and ``i+1`` with
    ``facecolors[i, j]``, turning an n x m grid into an (n-1) x (m-1) quad
    mesh.  Handing it node coordinates therefore shifts every colour half a
    cell towards +x and +y and drops the last row and column, which on a
    symmetric film draws the vacuum-adjacent dark band in full on the low
    edge and one cell short on the high edge.

    Both halves of that are checked here: the painted extent must straddle
    the film centre, and every data cell must reach the canvas.
    """
    mod = _load_module("sis_vortex_trapping_3d")
    params, device, trilayer = mod._build(3.0, 1.0, width=8.0, margin=2.0, pad=2.0)
    mod._carve_hole(params, device, trilayer)
    slice_z = mod._layer_midplanes(trilayer)[0]

    nx, ny, nz = mod._interior_shape(params)

    class _Stub:
        """Only ``psi(-1)`` is read by the painter."""

        def psi(self, step: int = -1):
            return np.ones(nx * ny * nz, dtype=complex)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    try:
        x0, x1, y0, y1 = mod._paint_layer(
            ax, params, device, _Stub(), slice_z,
            plt.get_cmap("inferno"), Normalize(0.0, 1.0),
        )
        centre_x = params.Nx * params.hx / 2.0
        centre_y = params.Ny * params.hy / 2.0
        assert x0 < centre_x < x1 and y0 < centre_y < y1
        assert (x0 + x1) / 2 == pytest.approx(centre_x, abs=1e-12), (
            f"sheet spans {x0}..{x1}, not centred on the film at {centre_x}"
        )
        assert (y0 + y1) / 2 == pytest.approx(centre_y, abs=1e-12), (
            f"sheet spans {y0}..{y1}, not centred on the film at {centre_y}"
        )

        i0, i1, j0, j1 = mod._film_extent(params, device, slice_z)
        expected = (i1 - i0) * (j1 - j0)
        assert expected > 0
        drawn = len(ax.collections[0].get_facecolors())
        assert drawn == expected, (
            f"{drawn} quads drawn for {expected} data cells — "
            "plot_surface is dropping the high edge"
        )
    finally:
        plt.close(fig)
