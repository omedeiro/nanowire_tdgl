"""Driver: run the sweeps, write the results to JSON.

Each tool is a separate subcommand because they have very different run
times and different optional dependencies, and because a sweep that dies
half way through should not lose the half that finished.  Results
accumulate into one file, keyed by tool.

::

    cd packages/tdgl3d
    python3 -m benchmarks.run superscreen
    python3 -m benchmarks.run pytdgl
    python3 -m benchmarks.run tdgl3d
    python3 -m benchmarks.run wall
    python3 -m benchmarks.run report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).with_name("results.json")

#: Λ/R values for the thin-film codes.  Spans both closed forms: the
#: weak-screening London limit at the top and complete screening at the
#: bottom, with the crossover (no closed form) in between.
SWEEP = [300.0, 100.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.3, 0.1, 0.03, 0.01]

#: pyTDGL is two orders of magnitude slower per point, so it gets the
#: same axis at fewer points rather than a different axis.
PYTDGL_SWEEP = [300.0, 100.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.3, 0.1]


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _store(path: Path, key: str, payload) -> None:
    """Merge *payload* under *key* and rewrite the file.

    Called after every point rather than after every sweep: the runs are
    minutes each and the long ones are hours in total, so a sweep that
    dies or is interrupted half way through should keep the half that
    finished.
    """
    data = _load(path)
    data[key] = payload
    path.write_text(json.dumps(data, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "what",
        choices=["superscreen", "pytdgl", "tdgl3d", "tdgl3d-convergence",
                 "wall", "report", "all"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mesh", type=float, default=None,
        help="Mesh size for the thin-film codes (µm); repeat runs to refine.",
    )
    args = parser.parse_args(argv)
    run(args.what, args.output, args.mesh)


def run(what: str, output: Path, mesh: float | None = None) -> None:
    from . import pearl_disk

    if what in ("superscreen", "all"):
        runs = []
        for x in SWEEP:
            kwargs = {"max_edge": mesh} if mesh else {}
            result = pearl_disk.run_superscreen(x, **kwargs)
            print(_line(result))
            runs.append(result.as_dict())
            _store(output, "superscreen", runs)
        print(f"wrote superscreen -> {output}")

    if what in ("pytdgl", "all"):
        runs = []
        for x in PYTDGL_SWEEP:
            kwargs = {"max_edge": mesh} if mesh else {}
            result = pearl_disk.run_pytdgl(x, **kwargs)
            print(_line(result))
            runs.append(result.as_dict())
            _store(output, "pytdgl", runs)
        print(f"wrote pytdgl -> {output}")

    if what in ("tdgl3d", "all"):
        from .tdgl3d_disk import SWEEP as T3D_SWEEP
        runs = []
        for case in T3D_SWEEP:
            result = pearl_disk.run_tdgl3d(**case)
            print(_line(result))
            runs.append(result.as_dict())
            _store(output, "tdgl3d", runs)
        print(f"wrote tdgl3d -> {output}")

    if what == "tdgl3d-convergence":
        from .tdgl3d_disk import CONVERGENCE
        runs = []
        for case in CONVERGENCE:
            result = pearl_disk.run_tdgl3d(**case)
            print(_line(result))
            runs.append(result.as_dict())
            _store(output, "tdgl3d_convergence", runs)
        print(f"wrote tdgl3d_convergence -> {output}")

    if what in ("wall", "all"):
        from . import gl_wall
        _store(output, "gl_wall", gl_wall.run_all())
        print(f"wrote gl_wall -> {output}")

    if what in ("report", "all"):
        from . import report
        report.write(output)


def _line(result) -> str:
    return (
        f"{result.tool:12s} Λ/R={result.lambda_over_r:8.4g}  "
        f"μ={result.mu:8.5f}  m/m_ideal={result.mu_ideal:8.5f}  "
        f"{result.meta.get('seconds', float('nan')):6.1f}s"
    )


if __name__ == "__main__":
    main()
