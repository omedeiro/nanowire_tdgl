"""tdgl3d, pyTDGL and SuperScreen on the same problems, against exact answers.

Reads ``packages/tdgl3d/benchmarks/results.json`` — produced by
``python3 -m benchmarks.run all`` from ``packages/tdgl3d`` — and draws
the three things the benchmark measures:

* **Left** — the normalised moment ``μ = m/m_London`` of a thin disk
  against ``Λ/R``, with the two closed forms it has to meet: ``μ = 1``
  where screening is weak, and the complete-screening line
  ``(64/3π)(Λ/R)`` where it is strong.  One dimensionless curve, so the
  three codes are on it without any unit conversion between them.
* **Middle** — the error each code makes where a closed form exists, and
  the difference between codes where none does.
* **Right** — the order-parameter equation on its own: measured ``ψ'``
  against ``(1 - ψ²)/√2`` at a pair-breaking wall, for the two codes
  that have a condensate at all.

Regenerate with ``python3 docs/figures/cross_tool_benchmark.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "tdgl3d"))

from benchmarks.closed_form import gl_wall_first_integral, ideal_over_london  # noqa: E402
from benchmarks.report import weak_limit  # noqa: E402

RESULTS = ROOT / "packages" / "tdgl3d" / "benchmarks" / "results.json"
OUTPUT = Path(__file__).with_suffix(".png")

STYLE = {
    "superscreen": {"color": "C0", "marker": "o", "label": "SuperScreen"},
    "pytdgl": {"color": "C1", "marker": "s", "label": "pyTDGL"},
    "tdgl3d": {"color": "C2", "marker": "^", "label": "tdgl3d"},
}


def _sorted(data, tool):
    rows = sorted(data.get(tool, []), key=lambda r: r["lambda_over_r"])
    return (
        np.array([r["lambda_over_r"] for r in rows]),
        np.array([r["mu"] for r in rows]),
    )


def crossover_panel(ax, data):
    # Each asymptote is drawn only where it is an asymptote.  Continuing
    # them across the whole axis would suggest the codes are being measured
    # against them in the crossover, where neither holds.
    weak = np.geomspace(1.0, 1e3, 100)
    ideal = np.geomspace(3e-3, 2.0, 100)
    ax.plot(weak, np.ones_like(weak), color="k", ls="--", lw=1.0)
    ax.plot(ideal, ideal_over_london(ideal), color="k", ls=":", lw=1.0)
    ax.text(6e2, 1.25, r"$\mu \to 1$" "\n" r"(London, $\Lambda \gg R$)",
            fontsize=8, ha="right")
    ax.text(4e-3, 0.05, r"$\frac{64}{3\pi}\Lambda/R$" "\n(complete screening)",
            fontsize=8)

    for tool, style in STYLE.items():
        xs, mus = _sorted(data, tool)
        if xs.size:
            ax.plot(xs, mus, lw=1.2, ms=4.5, **style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e-3, 1e3)
    ax.set_ylim(2e-2, 2.5)
    ax.set_xlabel(r"$\Lambda / R$")
    ax.set_ylabel(r"$\mu = m\,/\,m_{\mathrm{London}}$")
    ax.set_title("Thin disk in a perpendicular field")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)


def error_panel(ax, data):
    """What the codes cost each other, and what they cost against the exact answer.

    The lines are code-to-code, defined over the whole sweep including
    the crossover where nothing can be checked analytically.  The
    horizontal marks are each code's distance from ``μ = 1`` after the
    physical ``O(R/Λ)`` screening term has been fitted out — the one
    place on this axis where the closed form fixes a value, so the one
    place a "distance from exact" is a solver error rather than physics.
    """
    reference = {r["lambda_over_r"]: r["mu"] for r in data.get("superscreen", [])}
    other = {r["lambda_over_r"]: r["mu"] for r in data.get("pytdgl", [])}
    shared = sorted(set(reference) & set(other))
    if shared:
        ax.plot(
            shared,
            [abs(other[x] - reference[x]) / abs(reference[x]) for x in shared],
            color=STYLE["pytdgl"]["color"], lw=1.2, marker="s", ms=4,
            label="pyTDGL vs SuperScreen",
        )

    rows = data.get("tdgl3d", [])
    if rows and reference:
        xs = np.array(sorted(reference))
        mus = np.array([reference[x] for x in xs])
        points = sorted(r["lambda_over_r"] for r in rows)
        interpolated = np.interp(np.log(points), np.log(xs), mus)
        measured = np.array([r["mu"] for r in sorted(rows, key=lambda r: r["lambda_over_r"])])
        ax.plot(points, np.abs(measured - interpolated) / np.abs(interpolated),
                color=STYLE["tdgl3d"]["color"], lw=1.2, marker="^", ms=5,
                label="tdgl3d vs SuperScreen")

    for tool, style in STYLE.items():
        fit = weak_limit(sorted(data.get(tool, []), key=lambda r: -r["lambda_over_r"]))
        if fit is None:
            continue
        ax.axhline(abs(fit[0] - 1.0), color=style["color"], ls=":", lw=1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Lambda / R$")
    ax.set_ylabel(r"relative difference in $\mu$")
    ax.set_title(
        "Codes against each other (lines);\n"
        r"dotted: each code's own error at $\Lambda/R \to \infty$"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)


def wall_panel(ax, data):
    rows = data.get("gl_wall", [])
    if not rows:
        ax.set_axis_off()
        return
    finest = min(r["spacing"] for r in rows)
    psi = np.linspace(0.0, 1.0, 200)
    ax.plot(psi, gl_wall_first_integral(psi), "k-", lw=1.2,
            label=r"$(1-\psi^2)/\sqrt{2}$")
    for row in rows:
        if row["spacing"] != finest:
            continue
        style = dict(STYLE[row["tool"]])
        style.pop("marker")
        ax.plot(row["psi"], row["dpsi_dx"], ls="none", marker=STYLE[row["tool"]]["marker"],
                ms=4, alpha=0.8, **style)
    ax.set_xlabel(r"$\psi$")
    ax.set_ylabel(r"$\mathrm{d}\psi/\mathrm{d}x$   ($\xi^{-1}$)")
    ax.set_title(f"Pair-breaking wall, $h = {finest}\\,\\xi$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    """Draw the figure from the stored results.

    Unlike the other figure scripts this one runs no simulation — the
    runs it plots need pyTDGL and SuperScreen installed and take the
    better part of an hour — so *small* has nothing to shrink and is
    accepted only so the smoke test can call every script the same way.
    """
    del small
    data = json.loads(RESULTS.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    crossover_panel(axes[0], data)
    error_panel(axes[1], data)
    wall_panel(axes[2], data)
    fig.tight_layout()
    out = Path(output_dir) / OUTPUT.name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return [out]


if __name__ == "__main__":
    main()
