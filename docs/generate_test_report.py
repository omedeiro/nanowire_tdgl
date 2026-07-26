"""Generate a markdown report from per-test physics log files.

Scans logs/test_*.json (individual files written by PhysicsTestLogger)
and writes docs/physics_test_report.md with a summary table and per-test details.

Usage:
    python docs/generate_test_report.py
    python docs/generate_test_report.py --input logs/ --output docs/physics_test_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable


def _fmt(val: object) -> str:
    """Format a value for display in the report table."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if math.isinf(val):
            return "inf"
        if math.isnan(val):
            return "nan"
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def _status_icon(status: str) -> str:
    return "PASS" if status == "passed" else "FAIL"


# ---------------------------------------------------------------------------
# Per-test metric extractors
#
# Each returns (display_name, metric_value_str, detail_str | None) where
# metric_value_str is shown in the summary table and detail_str is a
# human-readable explanation of what was measured.
# ---------------------------------------------------------------------------


def _metric_uniform_zero_rhs(diag: dict) -> tuple[str, str, str]:
    v = diag.get("max_rhs", 0)
    return "Uniform state zero RHS", _fmt(v), "max|RHS| should be 0"


def _metric_c4_symmetry(diag: dict) -> tuple[str, str, str]:
    v = diag.get("max_symmetry_violation", 0)
    return "C4 symmetry preserved", _fmt(v), "max|φ_x + φ_y^T| should be 0"


def _metric_boundary_current(diag: dict) -> tuple[str, str, str]:
    v = diag.get("max_boundary_link_phi", 0)
    return "Supercurrent zero at boundary", _fmt(v), "max|φ_boundary| should be 0"


def _metric_cfl_stable(diag: dict) -> tuple[str, str, str]:
    mx = diag.get("max_psi2", 0)
    return "CFL stable (below limit)", _fmt(mx), "max|ψ|² should stay near 1"


def _metric_cfl_unstable(diag: dict) -> tuple[str, str, str]:
    v = diag.get("mean_psi_final", 0)
    return "CFL unstable (above limit)", _fmt(v), "mean|ψ| should collapse to ~0"


def _metric_div_b(diag: dict) -> tuple[str, str, str]:
    v = diag.get("div_to_B_ratio", diag.get("max_div_b", 0))
    return "B-field div-free", _fmt(v), "max|∇·B|/max|B| should be ~0"


def _metric_energy_dissipation(diag: dict) -> tuple[str, str, str]:
    inc = diag.get("max_energy_increase", 0)
    tol = diag.get("tolerance", 0)
    return (
        "Energy dissipation",
        f"{_fmt(inc)} (tol {_fmt(tol)})",
        "max relative energy increase must stay below tolerance",
    )


def _metric_insulator_decay(diag: dict) -> tuple[str, str, str]:
    tau_fit = diag.get("tau_fit", 0)
    tau_exp = diag.get("tau_expected", 0)
    if tau_exp:
        err = abs(tau_fit - tau_exp) / tau_exp
        return (
            "Insulator |ψ| decay",
            f"τ={_fmt(tau_fit)} (expected {tau_exp})",
            f"relative error = {err:.1%}",
        )
    return "Insulator |ψ| decay", _fmt(tau_fit), "τ should be ~0.1"


def _metric_bfield_uniform(diag: dict) -> tuple[str, str, str]:
    v = diag.get("bz_x_lo_std", 0)
    return "B-field uniform at boundary", _fmt(v), "std(Bz) at boundary should be 0"


def _metric_reversal_symmetry(diag: dict) -> tuple[str, str, str]:
    v = diag.get("max_asymmetry", 0)
    return "B-field reversal symmetry", _fmt(v), "max|Bz(+B) + Bz(-B)| should be 0"


def _metric_kappa_discontinuity(diag: dict) -> tuple[str, str, str]:
    sc = diag.get("sc_diag_mean")
    ins = diag.get("ins_diag_mean")
    exp_sc = diag.get("expected_sc")
    parts = []
    if sc is not None and exp_sc is not None:
        parts.append(f"SC={_fmt(sc)} (expected {exp_sc})")
    if ins is not None:
        parts.append(f"Ins={_fmt(ins)}")
    return "Trilayer κ discontinuity", ", ".join(parts) if parts else "—", (
        "SC diagonal should match κ² stencil, insulator should be 0"
    )


def _metric_z_boundary_jn(diag: dict) -> tuple[str, str, str]:
    lo = diag.get("max_jn_z_lo", 0)
    hi = diag.get("max_jn_z_hi", 0)
    mx = max(lo, hi)
    return "Trilayer z-boundary J_n", _fmt(mx), "J_n at z-faces should be 0"


def _metric_meissner(diag: dict) -> tuple[str, str, str]:
    lam = diag.get("lambda_fit")
    kappa = diag.get("lambda_expected")
    converged = diag.get("fit_converged", False)
    rel_err = diag.get("relative_error")
    if not converged or lam is None:
        return "Meissner screening", "fit failed", "cosh fit did not converge"
    if kappa and rel_err is not None:
        return (
            "Meissner screening",
            f"λ={_fmt(lam)} vs κ={kappa}",
            f"relative error = {rel_err:.1%}",
        )
    return "Meissner screening", f"λ={_fmt(lam)}", "λ should equal κ"


def _metric_trilayer_penetration(diag: dict) -> tuple[str, str, str]:
    bz_bot = diag.get("bz_bottom", 0)
    bz_ins = diag.get("bz_insulator", 0)
    bz_top = diag.get("bz_top", 0)
    bz_app = diag.get("bz_applied", 0)
    screened = diag.get("sc_screened")
    penetrated = diag.get("insulator_penetrated")
    if bz_app > 0:
        ins_ratio = bz_ins / bz_app
        parts = []
        if screened is not None:
            parts.append("SC✓" if screened else "SC✗")
        if penetrated is not None:
            parts.append("ins✓" if penetrated else "ins✗")
        status = ", ".join(parts) if parts else "—"
        return (
            "Trilayer B penetration",
            f"Bz(ins)={_fmt(bz_ins)} ({ins_ratio:.1%} of applied)",
            f"Bz(Nb)={_fmt(bz_bot)}/{_fmt(bz_top)}, Bz(app)={bz_app}, {status}",
        )
    return "Trilayer B penetration", "—", "no applied field"


def _metric_vortex_entry(diag: dict) -> tuple[str, str, str]:
    n = diag.get("n_vortices", 0)
    expected = diag.get("expected_approx", 0)
    winds = diag.get("winding_numbers", [])
    if expected > 0:
        return (
            "Vortex entry & counting",
            f"n={n} (expected ≈{expected:.0f})",
            f"detected {n/expected:.0%} of expected, winding={winds}",
        )
    return (
        "Vortex entry & counting",
        f"n={n}",
        "n > 0 and winding ≈ ±1 expected above H_c1",
    )


# Map test_name -> extractor function
METRIC_EXTRACTORS: dict[str, Callable[[dict], tuple[str, str, str]]] = {
    "test_uniform_state_zero_rhs": _metric_uniform_zero_rhs,
    "test_c4_symmetry_preserved_over_time": _metric_c4_symmetry,
    "test_supercurrent_zero_at_boundaries": _metric_boundary_current,
    "test_cfl_stability_below_limit": _metric_cfl_stable,
    "test_cfl_instability_above_limit": _metric_cfl_unstable,
    "test_bfield_divergence_free": _metric_div_b,
    "test_energy_dissipation_monotonic": _metric_energy_dissipation,
    "test_insulator_psi_exponential_decay": _metric_insulator_decay,
    "test_bfield_uniform_at_boundary": _metric_bfield_uniform,
    "test_bfield_reversal_symmetry": _metric_reversal_symmetry,
    "test_trilayer_kappa_discontinuity": _metric_kappa_discontinuity,
    "test_trilayer_external_z_boundary_jn": _metric_z_boundary_jn,
    "test_meissner_screening_exponential": _metric_meissner,
    "test_trilayer_bfield_penetration_profile": _metric_trilayer_penetration,
    "test_vortex_entry_and_counting": _metric_vortex_entry,
}


def generate_report(log_dir: Path) -> str:
    """Generate a markdown report from per-test JSON files in log_dir."""
    log_files = sorted(log_dir.glob("test_*.json"))
    if not log_files:
        return "# Physics Test Results Report\n\nNo test log files found.\n"

    results: list[dict] = []
    for lf in log_files:
        with open(lf) as f:
            results.append(json.load(f))

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")

    lines: list[str] = []

    # Header
    lines.append("# Physics Test Results Report")
    lines.append("")
    if results:
        lines.append(f"**Run timestamp:** {results[0].get('timestamp', 'unknown')}")
    lines.append(f"**Results:** {passed}/{total} passed, {failed} failed")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Test | Metric | Details | Status | Duration |")
    lines.append("|------|--------|---------|--------|----------|")

    for result in results:
        name = result["test_name"]
        diag = result.get("diagnostics", {})
        status = result.get("status", "unknown")
        duration = result.get("duration_s", 0)

        extractor = METRIC_EXTRACTORS.get(name)
        if extractor:
            display, metric, detail = extractor(diag)
        else:
            display = name
            metric = "—"
            detail = name

        icon = _status_icon(status)
        lines.append(
            f"| {display} | {metric} | {detail} | {icon} | {duration:.3f}s |"
        )

    lines.append("")

    # Detailed sections
    lines.append("## Detailed Results")
    lines.append("")

    for result in results:
        name = result["test_name"]
        diag = result.get("diagnostics", {})
        status = result.get("status", "unknown")
        params = result.get("params", {})
        error_msg = result.get("error")
        duration = result.get("duration_s", 0)

        extractor = METRIC_EXTRACTORS.get(name)
        display = extractor(diag)[0] if extractor else name

        lines.append(f"### {display}")
        lines.append("")
        lines.append(f"- **Status:** {_status_icon(status)}")
        lines.append(f"- **Duration:** {duration:.3f}s")

        if params:
            param_strs = [f"{k}={v}" for k, v in params.items()]
            lines.append(f"- **Parameters:** {', '.join(param_strs)}")

        if error_msg:
            lines.append(f"- **Error:** `{error_msg}`")

        # Key diagnostics (skip large array fields)
        skip_keys = {
            "energies",
            "psi_insulator_vs_time",
            "bfield_profile",
            "bz_profile",
            "vortex_positions",
            "winding_numbers",
        }
        key_diag = {
            k: v for k, v in diag.items() if k not in skip_keys and v is not None
        }
        if key_diag:
            lines.append("- **Diagnostics:**")
            for k, v in key_diag.items():
                lines.append(f"  - `{k}`: {_fmt(v)}")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate physics test results report from per-test log files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs"),
        help="Directory containing test_*.json log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "physics_test_report.md",
        help="Path to write the markdown report.",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: log directory not found at {args.input}")
        print(
            "Run the physics tests first: "
            "pytest packages/tdgl3d/tests/test_physics_validation.py -q"
        )
        return

    report = generate_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
