#!/usr/bin/env python3
"""Agent driver for the tdgl3d platform.

Two surfaces, one script:

* ``solve`` / ``inspect``  -- drive the solver **in-process**. This is the path
  most changes need: it imports ``tdgl3d``, runs a real (small) simulation,
  prints physics diagnostics, and renders a PNG you can actually look at.
* ``serve`` / ``api`` / ``stop`` -- drive the FastAPI job service over HTTP:
  start uvicorn in the background, POST a POM project, launch a job, follow the
  SSE stream, download the HDF5 artifact, summarise it.

Run from the repo root.  ``python3 .claude/skills/run-nanowire-tdgl/driver.py -h``
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless container: no display, must be set before pyplot

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / ".tdgl3d-run"
DEFAULT_PORT = 8787
TOKEN = os.environ.get("TDGL3D_API_TOKEN", "dev-token")


def _out_dir(path: str | None) -> Path:
    d = Path(path) if path else DEFAULT_OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# presets -- each returns a tdgl3d.Device
# --------------------------------------------------------------------------- #
def build_device(args):
    import tdgl3d

    field = tdgl3d.AppliedField(Bx=args.bx, By=args.by, Bz=args.bz, t_on_fraction=args.t_on)

    if args.preset == "trilayer":
        # S/I/S stack. kappa on a non-superconducting layer carries no
        # physics: the Maxwell coefficient takes params.kappa everywhere.
        k = args.kappa
        tri = tdgl3d.Trilayer(
            bottom=tdgl3d.Layer(thickness_z=args.nz, kappa=k, is_superconductor=True),
            insulator=tdgl3d.Layer(thickness_z=1, kappa=k, is_superconductor=False),
            top=tdgl3d.Layer(thickness_z=args.nz, kappa=k, is_superconductor=True),
        )
        params = tdgl3d.SimulationParameters(
            Nx=args.nx, Ny=args.ny, Nz=tri.Nz, hx=args.hx, hy=args.hy, hz=args.hz, kappa=k
        )
        return tdgl3d.Device(params, applied_field=field, trilayer=tri)

    params = tdgl3d.SimulationParameters(
        Nx=args.nx, Ny=args.ny, Nz=args.nz,
        hx=args.hx, hy=args.hy, hz=args.hz, kappa=args.kappa,
    )
    device = tdgl3d.Device(params, applied_field=field)

    if args.preset == "hole":
        cx, cy = args.nx * args.hx / 2.0, args.ny * args.hy / 2.0
        r = max(1.5, min(args.nx, args.ny) * 0.15)
        device.add_hole(
            [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        )
    return device


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def report(solution, device, *, label="solution"):
    """Print physics diagnostics. Returns a dict; raises on a non-physical run."""
    import tdgl3d

    psiN = solution.psi(-1)
    p2_0, p2_N = solution.psi_squared(0), solution.psi_squared(-1)
    Bx, By, Bz = solution.bfield(step=-1)

    d = {
        "label": label,
        "n_steps": int(solution.n_steps),
        "t": [float(solution.times[0]), float(solution.times[-1])],
        "psi_abs_max_final": float(np.abs(psiN).max()),
        "psi2_mean_initial": float(p2_0.mean()),
        "psi2_mean_final": float(p2_N.mean()),
        "psi2_min_final": float(p2_N.min()),
        "Bz_mean_final": float(np.mean(Bz)),
        "Bz_absmax_final": float(np.max(np.abs(Bz))),
    }

    # Vortex count at the LAST step alone is misleading: with t_on_fraction < 1
    # the applied field switches off and the vortices leave, while |psi|^2 still
    # shows their cores for a while. Report the whole trace and the peak.
    try:
        import tdgl3d.analysis as _an

        trace = []
        for k in range(solution.n_steps):
            vort, psi2min = _an.plaquette_vorticity(solution, step=k)
            trace.append(int(((np.abs(vort) > 0.8) & (psi2min >= 1e-6)).sum()))
        d["vortices_final"] = trace[-1]
        d["vortices_peak"] = max(trace)
        d["vortices_peak_t"] = float(solution.times[int(np.argmax(trace))])
        # compress the trace to "t=<time>:<count>" transitions, one line
        d["vortices_trace"] = " ".join(
            f"t={solution.times[k]:.2f}:{n}"
            for k, n in enumerate(trace)
            if k == 0 or n != trace[k - 1]
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the run
        d["vortices_final"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        # gl_free_energy takes a FLAT interior state vector, not a Solution.
        d["free_energy_final"] = float(
            tdgl3d.gl_free_energy(solution.states[:, -1], device.params, device.idx)
        )
    except Exception as exc:  # noqa: BLE001
        d["free_energy_final"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        steady, step, _ = solution.check_steady_state(device)
        d["steady_state"] = [bool(steady), int(step)]
    except Exception as exc:  # noqa: BLE001
        d["steady_state"] = f"unavailable: {type(exc).__name__}: {exc}"

    print(json.dumps(d, indent=2))

    # Sanity gates. These are the failures that produce plausible-looking but
    # meaningless output (see AGENTS.md "Known WIP").
    problems = []
    if not np.all(np.isfinite(solution.states.view(np.float64))):
        problems.append("state contains NaN/Inf -- CFL violated? dt < h^2/(4*kappa^2)")
    if d["psi_abs_max_final"] < 1e-2:
        problems.append(
            f"max|psi|={d['psi_abs_max_final']:.2e} -- fully pair-broken, "
            "nothing phase-derived is trustworthy"
        )
    # A CFL violation does not always produce NaN: forward Euler can also just
    # diverge into finite garbage (B ~ 1e19 while |psi| stays bounded).
    # In GL units B is O(1), so anything past 1e3 is numerical blow-up.
    if d["Bz_absmax_final"] > 1e3:
        problems.append(
            f"max|B_z|={d['Bz_absmax_final']:.3e} -- diverged, not physical. "
            "For method=euler check dt < h^2/(4*kappa^2)."
        )
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        raise SystemExit(1)
    return d


def render(solution, out: Path, *, gif=False, stride=2, name="summary"):
    from tdgl3d.visualization.plotting import animate, plot_summary

    png = out / f"{name}.png"
    fig = plot_summary(solution, step=-1)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    if gif:
        g = out / f"{name.replace('summary', 'vortices')}.gif"
        animate(solution, str(g), fps=10, step_stride=stride)
        print(f"wrote {g}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_solve(args):
    import tdgl3d

    out = _out_dir(args.out)
    device = build_device(args)
    print(device)

    cfl = args.hx ** 2 / (4 * args.kappa ** 2)
    if args.method == "euler" and args.dt >= cfl:
        print(
            f"WARNING: dt={args.dt} >= CFL limit h^2/(4*kappa^2)={cfl:.4g}; "
            "forward Euler will blow up",
            file=sys.stderr,
        )

    t0 = time.time()
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=args.t_stop,
        dt=args.dt,
        method=args.method,
        save_every=args.save_every,
        noise_amplitude=args.noise,
        noise_seed=args.seed,
        progress=args.progress,
        log_metadata=True,
        log_dir=out / "logs",  # keep run metadata out of the repo tree
    )
    print(f"solve wall time: {time.time() - t0:.2f}s")

    h5 = out / "solution.h5"
    solution.save(str(h5))
    print(f"wrote {h5}")
    render(solution, out, gif=args.gif)
    report(solution, device, label=args.preset)


def cmd_inspect(args):
    from tdgl3d import Solution

    # NOTE: use Solution.load(), NOT tdgl3d.io.hdf5.load_solution() -- they read
    # different on-disk layouts and load_solution() cannot read Solution.save()
    # output (it looks for states_real/states_imag; save() writes complex states).
    sol = Solution.load(args.path)
    print(f"file: {args.path}")
    print(f"times: {sol.times.shape} [{sol.times[0]:.3f} .. {sol.times[-1]:.3f}]")
    print(f"states: {sol.states.shape}")
    print(f"params: {sol.params}")
    p2 = sol.psi_squared(-1)
    print(f"|psi|^2 final: mean={p2.mean():.4f} min={p2.min():.4f} max={p2.max():.4f}")
    print("|psi|^2 (z-slice 0):")
    print(np.round(sol.psi_squared_2d(-1), 3))
    if args.png:
        out = _out_dir(args.out)
        render(sol, out)


# -- server ------------------------------------------------------------------ #
def _pidfile(out: Path) -> Path:
    return out / "uvicorn.pid"


def _get(url, token=TOKEN, timeout=10):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    # Defensive: force proxy-off so a container-wide HTTPS_PROXY whose no_proxy
    # does not list 127.0.0.1 can never swallow these localhost calls.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return r.status, r.read()


def _post(url, payload, token=TOKEN, timeout=30, method="POST"):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _wait_health(port, tries=40):
    for _ in range(tries):
        try:
            status, body = _get(f"http://127.0.0.1:{port}/health", timeout=2)
            if status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    return False


def cmd_serve(args, quiet=False):
    out = _out_dir(args.out)
    pf = _pidfile(out)
    if _wait_health(args.port, tries=1):
        if not quiet:
            print(f"server already healthy on :{args.port}")
        return None
    log = out / "uvicorn.log"
    env = dict(os.environ)
    env["TDGL3D_DATA_DIR"] = str(out / "serverdata")
    env["TDGL3D_API_TOKEN"] = TOKEN
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tdgl3d_server.app:app",
         "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=log.open("w"), stderr=subprocess.STDOUT, env=env, cwd=str(REPO),
    )
    pf.write_text(str(proc.pid))
    if not _wait_health(args.port):
        print(f"server did not come up; see {log}", file=sys.stderr)
        print(log.read_text()[-2000:], file=sys.stderr)
        raise SystemExit(1)
    print(f"server pid={proc.pid} on http://127.0.0.1:{args.port} (log: {log})")
    print(f"data dir: {env['TDGL3D_DATA_DIR']}  token: {TOKEN}")
    return proc


def cmd_stop(args):
    out = _out_dir(args.out)
    pf = _pidfile(out)
    if not pf.exists():
        print("no pidfile; nothing to stop")
        return
    pid = int(pf.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"sent SIGTERM to {pid}")
    except ProcessLookupError:
        print(f"pid {pid} already gone")
    pf.unlink()


def cmd_api(args):
    out = _out_dir(args.out)
    if args.start:
        cmd_serve(args, quiet=True)
    base = f"http://127.0.0.1:{args.port}"
    if not _wait_health(args.port, tries=2):
        print(f"no server on :{args.port}; run 'serve' first or pass --start", file=sys.stderr)
        raise SystemExit(1)

    project = {
        "id": args.project_id,
        "name": "driver smoke",
        "description": "driver.py api round-trip",
        "devices": [{
            "id": "dev1", "name": "film",
            "grid": {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz,
                     "hx": args.hx, "hy": args.hy, "hz": args.hz},
            "kappa": args.kappa,
            "applied_field": {"id": "fld1", "name": "Bz", "Bz": args.bz},
        }],
        "simulations": [{
            "id": "sim1", "name": "quick", "device_id": "dev1",
            "solver": {"method": args.method, "t_stop": args.t_stop, "dt": args.dt,
                       "save_every": args.save_every,
                       "initial_noise_amplitude": args.noise,
                       "initial_noise_seed": args.seed if args.seed is not None else 42},
        }],
    }

    # POST /projects upserts (store.put_project does ON CONFLICT DO UPDATE), so
    # re-running the driver with the same --project-id is safe and stays 201.
    status, body = _post(f"{base}/projects", project)
    print(f"POST /projects -> {status} id={body['id']}")

    status, job = _post(f"{base}/projects/{args.project_id}/jobs", {"simulation_id": "sim1"})
    job_id = job["id"]
    print(f"POST /projects/{args.project_id}/jobs -> {status} job={job_id} status={job['status']}")

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        _, raw = _get(f"{base}/jobs/{job_id}")
        job = json.loads(raw)
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.5)
    print(f"job finished: status={job['status']} after {time.time() - t0:.1f}s")
    if job["status"] != "completed":
        print(job.get("error") or "(no error recorded)", file=sys.stderr)
        raise SystemExit(1)

    art = out / f"{job_id}.h5"
    _, blob = _get(f"{base}/jobs/{job_id}/result", timeout=60)
    art.write_bytes(blob)
    print(f"GET /jobs/{job_id}/result -> {len(blob)} bytes -> {art}")

    from tdgl3d import Solution

    sol = Solution.load(str(art))
    p2 = sol.psi_squared(-1)
    print(f"artifact: {sol.states.shape} states, {sol.n_steps} frames, "
          f"t=[{sol.times[0]:.2f}, {sol.times[-1]:.2f}]")
    print(f"|psi|^2 final: mean={p2.mean():.4f} min={p2.min():.4f}")
    render(sol, out, name="api-summary")
    print("API round-trip OK")


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def physics_args(sp, *, nx=15, ny=15, nz=1, kappa=5.0, bz=0.5,
                     method="euler", t_stop=5.0, dt=0.005):
        sp.add_argument("--nx", type=int, default=nx)
        sp.add_argument("--ny", type=int, default=ny)
        sp.add_argument("--nz", type=int, default=nz)
        sp.add_argument("--hx", type=float, default=1.0)
        sp.add_argument("--hy", type=float, default=1.0)
        sp.add_argument("--hz", type=float, default=1.0)
        sp.add_argument("--kappa", type=float, default=kappa)
        sp.add_argument("--bx", type=float, default=0.0)
        sp.add_argument("--by", type=float, default=0.0)
        sp.add_argument("--bz", type=float, default=bz)
        sp.add_argument("--t-on", type=float, default=1.0, dest="t_on",
                        help="fraction of t_stop the field stays on "
                             "(tdgl3d's own default is 2/3, which switches it off)")
        sp.add_argument("--method", choices=["euler", "trapezoidal"], default=method)
        sp.add_argument("--t-stop", type=float, default=t_stop, dest="t_stop")
        sp.add_argument("--dt", type=float, default=dt)
        sp.add_argument("--save-every", type=int, default=5, dest="save_every")
        sp.add_argument("--noise", type=float, default=0.01)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", default=None)

    s = sub.add_parser("solve", help="in-process solve + diagnostics + PNG")
    s.add_argument("--preset", choices=["film2d", "trilayer", "hole"], default="film2d")
    s.add_argument("--gif", action="store_true", help="also render an animated GIF")
    s.add_argument("--progress", action="store_true", help="show the tqdm bar")
    physics_args(s)
    s.set_defaults(func=cmd_solve)

    s = sub.add_parser("inspect", help="summarise a saved .h5 artifact")
    s.add_argument("path")
    s.add_argument("--png", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("serve", help="start uvicorn in the background, wait for /health")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("stop", help="stop the background server")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("api", help="full HTTP round-trip against the job service")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--start", action="store_true", help="start the server if it isn't up")
    s.add_argument("--project-id", default="driver-demo", dest="project_id")
    s.add_argument("--timeout", type=float, default=300)
    physics_args(s, nx=8, ny=8, kappa=2.0, t_stop=2.0, dt=0.02)
    s.set_defaults(func=cmd_api)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
