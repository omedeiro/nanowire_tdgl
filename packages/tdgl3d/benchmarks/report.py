"""Turn ``results.json`` into the error tables.

Three kinds of number come out of this, and they answer different
questions:

* **Against the closed form** — only meaningful where a closed form
  applies, so the weak-screening column is reported only for
  ``Λ/R ≥ 5`` and the complete-screening column only for ``Λ/R ≤ 0.03``.
  Reporting them everywhere would dress the physical crossover up as
  numerical error.
* **Against each other** — ``|μ_a - μ_b| / μ_b`` at matched ``Λ/R``,
  which is defined over the whole sweep including the crossover, where
  no closed form exists.  This is the number that says whether two
  independent implementations of the same equation agree.
* **Under refinement** — the same quantity at two mesh sizes, which is
  what distinguishes discretisation error from a modelling difference:
  the first shrinks, the second does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WEAK_MIN = 5.0        # Λ/R above which the London asymptote is worth quoting
IDEAL_MAX = 0.03      # Λ/R below which the complete-screening one is


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}g}"


def _rows(data, tool):
    return sorted(data.get(tool, []), key=lambda r: -r["lambda_over_r"])


def weak_limit(rows) -> tuple[float, int] | None:
    r"""Extrapolate ``μ`` to ``R/Λ → 0``, where the London form is exact.

    ``μ`` is not ``1`` at any finite ``Λ``: screening reduces it, and the
    reduction is first order in ``R/Λ`` — about ``0.15 R/Λ`` for a disk.
    Quoting ``|μ - 1|`` at, say, ``Λ/R = 30`` therefore reports a 0.5%
    *physical* correction as if it were solver error.  A straight-line
    fit in ``R/Λ`` through the weakly screening points removes exactly
    that term, and its intercept is a quantity the closed form does fix
    at ``1``, so the distance from ``1`` is the code's own error.

    Returns ``(intercept, number of points used)``.
    """
    points = [(1.0 / r["lambda_over_r"], r["mu"]) for r in rows
              if r["lambda_over_r"] >= WEAK_MIN]
    if len(points) < 2:
        return None
    x, y = np.array(points).T
    slope, intercept = np.polyfit(x, y, 1)
    del slope
    return float(intercept), len(points)


def profile_rms(row) -> float:
    """rms of ``K_φ(r)/K_London(r) - 1`` over the sampled radii.

    A field-level error rather than an integrated one, so a code whose
    moment happens to come out right by cancellation does not pass it.
    The three codes sample different radii — the Cartesian one has nodes
    only where its grid puts them — but an rms over each code's own
    radii is still comparable between them.
    """
    ratio = np.asarray(row["sheet_current"], dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((ratio - 1.0) ** 2)))


def weak_limit_table(data) -> str:
    """The one place the closed form is exact, and each code's error there."""
    lines = [
        "| tool | points fitted (Λ/R ≥ 5) | μ extrapolated to Λ/R → ∞ | error in μ "
        "| rms of K_φ/K_London - 1 at the weakest screening |",
        "|---|---|---|---|---|",
    ]
    for tool in ("superscreen", "pytdgl", "tdgl3d"):
        rows = _rows(data, tool)
        fit = weak_limit(rows)
        if fit is None:
            continue
        intercept, count = fit
        weakest = max(rows, key=lambda r: r["lambda_over_r"])
        lines.append(
            f"| {tool} | {count} | {_fmt(intercept, 6)} "
            f"| {_fmt(abs(intercept - 1.0), 3)} | {_fmt(profile_rms(weakest), 3)} |"
        )
    return "\n".join(lines)


def closed_form_table(data) -> str:
    lines = [
        "| tool | Λ/R | μ = m/m_London | \\|μ-1\\| | m/m_ideal | \\|m/m_ideal - 1\\| |",
        "|---|---|---|---|---|---|",
    ]
    for tool in ("superscreen", "pytdgl", "tdgl3d"):
        for row in _rows(data, tool):
            x = row["lambda_over_r"]
            weak = abs(row["mu"] - 1.0) if x >= WEAK_MIN else None
            ideal = abs(row["mu_ideal"] - 1.0) if x <= IDEAL_MAX else None
            lines.append(
                f"| {tool} | {_fmt(x)} | {_fmt(row['mu'], 5)} | {_fmt(weak, 3)} "
                f"| {_fmt(row['mu_ideal'], 5)} | {_fmt(ideal, 3)} |"
            )
    return "\n".join(lines)


def pairwise_table(data) -> str:
    """``|μ_a - μ_b|/μ_b`` at matched Λ/R, over the whole sweep."""
    a = {r["lambda_over_r"]: r["mu"] for r in data.get("pytdgl", [])}
    b = {r["lambda_over_r"]: r["mu"] for r in data.get("superscreen", [])}
    shared = sorted(set(a) & set(b), reverse=True)
    lines = ["| Λ/R | μ (SuperScreen) | μ (pyTDGL) | relative difference |", "|---|---|---|---|"]
    for x in shared:
        lines.append(
            f"| {_fmt(x)} | {_fmt(b[x], 5)} | {_fmt(a[x], 5)} "
            f"| {_fmt(abs(a[x] - b[x]) / abs(b[x]), 3)} |"
        )

    t3d = data.get("tdgl3d", [])
    if t3d and b:
        xs = np.array(sorted(b))
        mus = np.array([b[x] for x in xs])
        lines.append("")
        lines.append(
            "| Λ_eff/R_eff | μ (tdgl3d) | μ (SuperScreen, interp.) | relative difference |"
        )
        lines.append("|---|---|---|---|")
        for row in sorted(t3d, key=lambda r: -r["lambda_over_r"]):
            x = row["lambda_over_r"]
            reference = float(np.interp(np.log(x), np.log(xs), mus))
            lines.append(
                f"| {_fmt(x)} | {_fmt(row['mu'], 5)} | {_fmt(reference, 5)} "
                f"| {_fmt(abs(row['mu'] - reference) / abs(reference), 3)} |"
            )
    return "\n".join(lines)


def convergence_table(data) -> str:
    """The two approximations the tdgl3d sweep makes, changed one at a time."""
    rows = data.get("tdgl3d_convergence", [])
    if not rows:
        return "_Not run._"
    reference = {r["lambda_over_r"]: r["mu"] for r in data.get("superscreen", [])}
    xs = np.array(sorted(reference))
    mus = np.array([reference[x] for x in xs])
    lines = [
        "| h (ξ) | box (ξ) | interior nodes | Λ_eff/R_eff | ∫\\|ψ\\|²dz | μ "
        "| distance from SuperScreen | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        box = f"{row['lateral_cells'] * row['spacing']:.0f}×{row['z_cells'] * row['spacing']:.0f}"
        gap = float("nan")
        if xs.size:
            interpolated = float(
                np.interp(np.log(row["lambda_over_r"]), np.log(xs), mus)
            )
            gap = abs(row["mu"] - interpolated) / abs(interpolated)
        lines.append(
            f"| {_fmt(row['spacing'])} | {box} | {row['interior_nodes']} "
            f"| {_fmt(row['lambda_over_r'], 4)} | {_fmt(row['sheet_ns'], 4)} "
            f"| {_fmt(row['mu'], 5)} | {_fmt(gap, 3)} | {row['seconds']:.0f} |"
        )
    return "\n".join(lines)


def wall_table(data) -> str:
    lines = [
        "| tool | h (ξ) | rms of ψ' - (1-ψ²)/√2 | fitted healing length (exact √2 = 1.41421) |",
        "|---|---|---|---|",
    ]
    for row in data.get("gl_wall", []):
        lines.append(
            f"| {row['tool']} | {_fmt(row['spacing'])} | {_fmt(row['rms'], 3)} "
            f"| {_fmt(row['healing_length'], 6)} |"
        )
    return "\n".join(lines)


def write(results: Path) -> str:
    data = json.loads(Path(results).read_text())
    text = "\n\n".join([
        "# Cross-tool benchmark results",
        "Generated by `python3 -m benchmarks.run report`.",
        "## Thin disk: the weak-screening limit, where the London form is exact",
        weak_limit_table(data),
        "## Thin disk: every point, against both closed forms",
        closed_form_table(data),
        "## Thin disk: the codes against each other",
        pairwise_table(data),
        "## Thin disk: what the tdgl3d approximations cost",
        convergence_table(data),
        "## Pair-breaking wall: the order-parameter equation",
        wall_table(data),
        "",
    ])
    out = Path(results).with_name("REPORT.md")
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
    return text
