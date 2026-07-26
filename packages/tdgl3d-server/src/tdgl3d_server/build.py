"""Translate POM specs into tdgl3d runtime objects."""

from __future__ import annotations

from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d_schema import DeviceSpec, SolverSettings


def build_device(spec: DeviceSpec) -> Device:
    """Construct a runnable :class:`tdgl3d.Device` from a POM ``DeviceSpec``."""
    params = SimulationParameters(
        Nx=spec.grid.Nx,
        Ny=spec.grid.Ny,
        Nz=spec.grid.Nz,
        hx=spec.grid.hx,
        hy=spec.grid.hy,
        hz=spec.grid.hz,
        kappa=spec.kappa,
    )
    trilayer = None
    if spec.trilayer is not None:
        t = spec.trilayer
        trilayer = Trilayer(
            bottom=Layer(t.bottom.thickness_z, t.bottom.kappa, t.bottom.is_superconductor),
            insulator=Layer(
                t.insulator.thickness_z, t.insulator.kappa, t.insulator.is_superconductor
            ),
            top=Layer(t.top.thickness_z, t.top.kappa, t.top.is_superconductor),
        )
    field = AppliedField(
        Bx=spec.applied_field.Bx,
        By=spec.applied_field.By,
        Bz=spec.applied_field.Bz,
        t_on_fraction=spec.applied_field.t_on_fraction,
        ramp=spec.applied_field.ramp,
        ramp_fraction=spec.applied_field.ramp_fraction,
    )
    device = Device(params=params, applied_field=field, trilayer=trilayer)
    for hole in spec.holes:
        device.add_hole([tuple(v) for v in hole.vertices], z_range=hole.z_range)
    return device


def solve_kwargs(settings: SolverSettings) -> dict:
    """Map POM ``SolverSettings`` onto :func:`tdgl3d.solve` keyword arguments."""
    return dict(
        t_start=settings.t_start,
        t_stop=settings.t_stop,
        dt=settings.dt,
        method=settings.method,
        save_every=settings.save_every,
        newton_tol_f=settings.newton_tol_f,
        newton_tol_dx=settings.newton_tol_dx,
        newton_max_iter=settings.newton_max_iter,
        tol_gcr=settings.tol_gcr,
        eps_mf=settings.eps_mf,
        adaptive=settings.adaptive,
        noise_amplitude=settings.initial_noise_amplitude,
        noise_seed=settings.initial_noise_seed,
        progress=False,
        log_metadata=False,
    )
