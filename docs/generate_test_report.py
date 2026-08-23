"""Generate a markdown report from per-test physics log files.

Scans ``logs/test_*.json`` (written by the ``phys_log`` fixture in
``packages/tdgl3d/tests/conftest.py``) and writes
``docs/physics_test_report.md``.

Every row of the report comes from a *structured check* recorded by the test —
what was measured, what was expected and how much slack was allowed — so the
report can only claim a physical statement that the suite actually asserted.
Tests that record no checks are listed separately rather than being reported as
if they had verified something.

Usage::

    python3 docs/generate_test_report.py
    python3 docs/generate_test_report.py --input logs/ --output docs/physics_test_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


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
        if val == 0:
            return "0"
        if abs(val) < 1e-3 or abs(val) >= 1e5:
            return f"{val:.3e}"
        return f"{val:.6g}"
    if isinstance(val, (list, tuple)):
        if len(val) > 8:
            body = ", ".join(_fmt(v) for v in list(val)[:8])
            return f"[{body}, … ({len(val)} values)]"
        return "[" + ", ".join(_fmt(v) for v in val) + "]"
    if isinstance(val, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in val.items())
    return str(val)


def _status_icon(status: str) -> str:
    return "PASS" if status == "passed" else "FAIL"


def _check_status(check: dict[str, Any]) -> str:
    return "PASS" if check.get("passed") else "FAIL"


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|")


MODULE_GROUPS = {
    "test_verification_gauge": "Gauge invariance",
    "test_verification_conservation": "Conservation laws and identities",
    "test_verification_symmetry": "Symmetry and boundary conditions",
    "test_verification_analytic": "Analytic limits",
    "test_verification_vortex": "Vortices and flux quantisation",
    "test_physics_validation": "Heterostructures",
    "test_verification_expulsion": "Flux expulsion by a ring",
}


def _group_of(result: dict[str, Any]) -> str:
    """Group a result by the suite that produced it."""
    return MODULE_GROUPS.get(result.get("module", ""), "Other")


GROUP_ORDER = [
    "Gauge invariance",
    "Conservation laws and identities",
    "Symmetry and boundary conditions",
    "Analytic limits",
    "Vortices and flux quantisation",
    "Heterostructures",
    "Flux expulsion by a ring",
    "Other",
]


def generate_report(log_dir: Path) -> str:
    """Generate a markdown report from per-test JSON files in *log_dir*."""
    log_files = sorted(log_dir.glob("test_*.json"))
    if not log_files:
        return "# Physics Verification Report\n\nNo test log files found.\n"

    results: list[dict] = []
    for log_file in log_files:
        with open(log_file) as handle:
            results.append(json.load(handle))

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = total - passed
    all_checks = [c for r in results for c in r.get("checks", [])]
    checks_passed = sum(1 for c in all_checks if c.get("passed"))
    unchecked = [r for r in results if not r.get("checks")]

    lines: list[str] = ["# Physics Verification Report", ""]
    if results:
        newest = max(r.get("timestamp", "") for r in results)
        lines.append(f"**Run timestamp:** {newest}")
    lines.append(f"**Tests:** {passed}/{total} passed, {failed} failed")
    lines.append(
        f"**Checks:** {checks_passed}/{len(all_checks)} passed "
        f"({len(all_checks) - checks_passed} failed)"
    )
    lines.append("")
    lines.append(
        "Each check records the measured value, the value physics requires and the "
        "tolerance allowed, so every line below is falsifiable. Tolerances near "
        "machine precision mark exact discrete identities; the wider ones are "
        "discretisation error bounds stated up front rather than fitted to the "
        "measurement."
    )
    lines.append("")

    # -- summary by group ---------------------------------------------------
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(_group_of(result), []).append(result)

    lines.append("## Checks")
    lines.append("")
    for group in GROUP_ORDER:
        entries = grouped.get(group)
        if not entries:
            continue
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Check | Measured | Expected | Tolerance | Status |")
        lines.append("|-------|----------|----------|-----------|--------|")
        for result in sorted(entries, key=lambda r: r["test_name"]):
            checks = result.get("checks", [])
            if not checks:
                continue
            lines.append(f"| **{_escape(result['test_name'])}** | | | | |")
            for check in checks:
                lines.append(
                    f"| {_escape(check['label'])} "
                    f"| {_fmt(check['measured'])} "
                    f"| {_escape(_fmt(check['expected']))} "
                    f"| {_escape(_fmt(check['tolerance']))} "
                    f"| {_check_status(check)} |"
                )
        lines.append("")

    # -- per-test detail ----------------------------------------------------
    lines.append("## Test details")
    lines.append("")
    for result in sorted(results, key=lambda r: r["test_name"]):
        name = result["test_name"]
        lines.append(f"### {name}")
        lines.append("")
        if result.get("description"):
            lines.append(f"_{result['description']}_")
            lines.append("")
        lines.append(f"- **Status:** {_status_icon(result.get('status', 'unknown'))}")
        lines.append(f"- **Duration:** {result.get('duration_s', 0):.3f}s")

        params = result.get("params", {})
        if params:
            lines.append(
                "- **Parameters:** " + ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())
            )
        if result.get("error"):
            lines.append(f"- **Error:** `{result['error']}`")

        for check in result.get("checks", []):
            detail = f" — {check['detail']}" if check.get("detail") else ""
            lines.append(
                f"- **{_check_status(check)}** {check['label']}: "
                f"measured {_fmt(check['measured'])}, "
                f"expected {_fmt(check['expected'])}{detail}"
            )

        diagnostics = {
            k: v for k, v in result.get("diagnostics", {}).items() if v is not None
        }
        if diagnostics:
            lines.append("- **Diagnostics:**")
            for key, value in diagnostics.items():
                lines.append(f"  - `{key}`: {_fmt(value)}")
        lines.append("")

    if unchecked:
        lines.append("## Tests recording no structured checks")
        lines.append("")
        lines.append(
            "These logged diagnostics but asserted nothing through the check API, so "
            "the report cannot say what they verified."
        )
        lines.append("")
        for result in unchecked:
            lines.append(f"- `{result['test_name']}` ({_status_icon(result.get('status', ''))})")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the physics verification report from per-test log files."
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
            "Run the physics tests first, from packages/tdgl3d/:\n"
            "    python3 -m pytest tests/test_verification_*.py "
            "tests/test_physics_validation.py -q"
        )
        return

    report = generate_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
