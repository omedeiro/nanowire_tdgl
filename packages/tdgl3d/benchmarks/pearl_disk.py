r"""A thin superconducting disk in a perpendicular field, in three codes.

The disk is the one geometry where the weak-screening London solution is
exact rather than approximate (see :mod:`benchmarks.closed_form`), and it
is also the geometry whose complete-screening limit is classical.  That
makes ``Λ/R`` a single axis with an exact answer at each end, and the
three codes can be laid on it together.

What each code is asked for
---------------------------
``superscreen``
    Solves the thin-film London equation directly, for the stream
    function ``g``.  It has no order parameter, so it *is* the
    ``|ψ| = 1`` model — the closed forms are statements about the
    equation it solves, and its error is mesh error and nothing else.
``tdgl``  (pyTDGL)
    Solves 2-D TDGL on a triangular mesh with the same thin-film
    screening model, relaxed to a steady state.  At the fields used here
    ``|ψ| ≈ 1``, so it should land on the same curve; the difference from
    ``superscreen`` measures the two discretisations plus whatever the
    condensate does at finite field.
``tdgl3d``
    Solves 3-D TDGL on a Cartesian grid, with the film given a real
    thickness and vacuum around it.  It is not a thin-film code: the
    Pearl length is an outcome of ``κ`` and the thickness rather than an
    input, and its non-superconducting regions are pair-breaking, so a
    film a few ξ thick has ``|ψ| < 1`` through its whole thickness.
    Both effects are measured here rather than assumed away — see
    ``moment_london_effective`` and ``lambda_over_r_effective`` on the
    result.

The reported number
-------------------
``mu = m / m_London``, with ``m_London`` computed in each code's own
units.  It is dimensionless, it is ``1`` in the weak-screening limit for
every code, and it needs no unit conversion between them, so the three
sets of runs are directly comparable and directly comparable to the
closed forms.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .closed_form import ideal_disk_moment, london_disk_moment

__all__ = [
    "DiskRun",
    "MU0",
    "run_pytdgl",
    "run_superscreen",
    "run_tdgl3d",
]

#: Vacuum permeability, T·m/A.  ``B/μ₀`` in A/m is numerically ``μA/μm``,
#: which is why the SI-unit runners need no conversion factors.
MU0 = 4.0e-7 * np.pi


@dataclass
class DiskRun:
    """One disk solved by one code.

    ``moment`` and ``moment_london`` are in whatever units that code
    works in; only their ratio ``mu`` is compared across codes.
    """

    tool: str
    lambda_over_r: float
    moment: float
    moment_london: float
    #: ``r/R`` and ``K_φ(r)/K_London(r)`` along a radius, for the profile check.
    radius: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    sheet_current: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    meta: dict = field(default_factory=dict)

    @property
    def mu(self) -> float:
        return self.moment / self.moment_london

    @property
    def mu_ideal(self) -> float:
        """``m / m_ideal`` — the other end's reference, ``1`` at ``Λ/R → 0``."""
        return self.moment / self.meta["moment_ideal"]

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "lambda_over_r": self.lambda_over_r,
            "moment": self.moment,
            "moment_london": self.moment_london,
            "mu": self.mu,
            "mu_ideal": self.mu_ideal,
            "radius": self.radius.tolist(),
            "sheet_current": self.sheet_current.tolist(),
            **{k: v for k, v in self.meta.items()},
        }


def _profile_radii(n: int = 36, r_min: float = 0.2, r_max: float = 0.9) -> NDArray[np.float64]:
    """Radii to sample ``K_φ`` at, as a fraction of ``R``.

    Stops short of both ends.  At the rim the complete-screening solution
    diverges and every code regularises the divergence with its own mesh,
    so the last tenth of the radius compares discretisations rather than
    physics; at the centre ``K_φ`` goes to zero, and the ratio to the
    closed form there is a small number over a small number.
    """
    return np.linspace(r_min, r_max, n)


# ---------------------------------------------------------------------------
# SuperScreen
# ---------------------------------------------------------------------------

def run_superscreen(
    lambda_over_r: float,
    *,
    radius: float = 5.0,
    field_mT: float = 1.0,
    max_edge: float = 0.25,
    boundary_points: int = 401,
    dtype: str = "float64",
) -> DiskRun:
    """Solve the disk with SuperScreen.

    The magnetic moment is read off the stream function as
    ``m = ∫ g dA``: for a sheet current ``K = ∇×(g ẑ)`` with ``g = 0`` on
    the boundary, integrating ``½∫(r×K)_z dA`` by parts leaves exactly
    that, with no derivative of ``g`` taken numerically.
    """
    import superscreen as sc

    Lambda = lambda_over_r * radius
    layer = sc.Layer("film", Lambda=Lambda, z0=0.0)
    film = sc.Polygon(
        "film", layer="film", points=sc.geometry.circle(radius, points=boundary_points)
    )
    device = sc.Device(
        "disk", layers=[layer], films=[film], length_units="um", solve_dtype=dtype
    )
    # buffer=0.0 matters: make_mesh otherwise pads the film polygon by 5% of
    # its bounding box and solves on the *padded* disk, which is 10% larger in
    # radius here and reads 1.5% high on the moment at every Lambda — a
    # systematic that does not shrink under mesh refinement, so it cannot be
    # mistaken for discretisation error.
    device.make_mesh(max_edge_length=max_edge, buffer=0.0, smooth=20)

    start = time.perf_counter()
    solution = sc.solve(
        device,
        applied_field=sc.sources.ConstantField(field_mT),
        field_units="mT",
        current_units="uA",
        progress_bar=False,
    )[-1]
    elapsed = time.perf_counter() - start

    mesh = device.meshes["film"]
    film_solution = solution.film_solutions["film"]
    moment = float(np.sum(film_solution.stream * mesh.vertex_areas))   # µA·µm²

    H_a = field_mT * 1e-3 / MU0                                        # µA/µm
    frac = _profile_radii()
    positions = np.column_stack([frac * radius, np.zeros_like(frac)])
    K = solution.interp_current_density(
        positions, film="film", units="uA/um", with_units=False
    )
    # K_φ on the +x axis is +K_y.
    k_london = -0.5 * H_a * (frac * radius) / Lambda
    k_ratio = K[:, 1] / k_london

    return DiskRun(
        tool="superscreen",
        lambda_over_r=lambda_over_r,
        moment=moment,
        moment_london=london_disk_moment(H_a, radius, Lambda),
        radius=frac,
        sheet_current=k_ratio,
        meta={
            "moment_ideal": ideal_disk_moment(H_a, radius),
            "sites": int(mesh.sites.shape[0]),
            "max_edge": max_edge,
            "seconds": elapsed,
            "Lambda": Lambda,
            "radius_um": radius,
            "field_mT": field_mT,
        },
    )


# ---------------------------------------------------------------------------
# pyTDGL
# ---------------------------------------------------------------------------

def run_pytdgl(
    lambda_over_r: float,
    *,
    radius: float = 5.0,
    xi: float = 0.5,
    thickness: float = 0.05,
    field_mT: float = 0.01,
    max_edge: Optional[float] = None,
    solve_time: float = 30.0,
    boundary_points: int = 301,
) -> DiskRun:
    """Solve the disk with pyTDGL, with self-consistent screening on.

    ``Λ = λ²/d`` is set by choosing ``λ`` for the requested ``Λ/R`` at
    fixed thickness.  The field is kept well below ``B_c2 = Φ₀/2πξ²`` so
    the run stays in the linear London regime the closed forms describe;
    ``psi_min`` on the result records how far ``|ψ|`` actually moved.
    """
    import tdgl
    from tdgl.geometry import circle

    Lambda = lambda_over_r * radius
    london_lambda = float(np.sqrt(Lambda * thickness))
    layer = tdgl.Layer(
        coherence_length=xi,
        london_lambda=london_lambda,
        thickness=thickness,
        gamma=1.0,
    )
    film = tdgl.Polygon("film", points=circle(radius, points=boundary_points))
    device = tdgl.Device("disk", layer=layer, film=film, length_units="um")
    device.make_mesh(max_edge_length=max_edge or 0.5 * xi, smooth=20)

    # pyTDGL writes its own scratch files alongside the output — a
    # ``.h5.tmp`` while solving, and a ``-1`` variant if the name is taken —
    # so it gets a directory of its own rather than the working tree.
    workspace = tempfile.mkdtemp(prefix="tdgl3d-benchmark-")
    output = os.path.join(workspace, "disk.h5")
    options = tdgl.SolverOptions(
        solve_time=solve_time,
        output_file=output,
        field_units="mT",
        current_units="uA",
        include_screening=True,
        save_every=10**6,
    )
    start = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()):
        solution = tdgl.solve(device, options, applied_vector_potential=field_mT)
    elapsed = time.perf_counter() - start

    moment = float(solution.magnetic_moment(units="uA * um ** 2").magnitude)

    # The London reference is evaluated on the condensate this run actually
    # has, not on |ψ| = 1.  A finite probe field depletes the condensate by
    # about (B/B_c2)²R²/8ξ², which at the field and radius used here is under
    # 0.1% — but measuring it is cheaper than arguing that it is small, and it
    # makes ``mu`` mean the same thing as it does for the 3-D code, where the
    # depletion is not small.
    density = np.abs(solution.tdgl_data.psi) ** 2
    psi_min = float(np.sqrt(density.min()))
    x_site, y_site = device.points.T
    areas = device.areas
    second_moment = float(np.sum(density * (x_site**2 + y_site**2) * areas))

    H_a = field_mT * 1e-3 / MU0
    frac = _profile_radii()
    positions = np.column_stack([frac * radius, np.zeros_like(frac)])
    K = solution.interp_current_density(
        positions, units="uA/um", with_units=False
    )
    k_london = -0.5 * H_a * (frac * radius) / Lambda
    k_ratio = K[:, 1] / k_london

    sites = int(device.points.shape[0])
    del solution
    shutil.rmtree(workspace, ignore_errors=True)

    return DiskRun(
        tool="pytdgl",
        lambda_over_r=lambda_over_r,
        moment=moment,
        moment_london=-0.25 * H_a * second_moment / Lambda,
        radius=frac,
        sheet_current=k_ratio,
        meta={
            "moment_ideal": ideal_disk_moment(H_a, radius),
            "moment_london_geometric": london_disk_moment(H_a, radius, Lambda),
            "sites": sites,
            "max_edge": max_edge or 0.5 * xi,
            "seconds": elapsed,
            "Lambda": Lambda,
            "radius_um": radius,
            "field_mT": field_mT,
            "field_over_bc2": field_mT / _bc2_mT(xi),
            "psi_min": psi_min,
            "london_lambda": london_lambda,
            "thickness": thickness,
            "xi": xi,
        },
    )


def _bc2_mT(xi_um: float) -> float:
    """``B_c2 = Φ₀/(2πξ²)`` in mT, for ξ in µm."""
    phi0 = 2.067833848e-15
    return phi0 / (2.0 * np.pi * (xi_um * 1e-6) ** 2) * 1e3


# ---------------------------------------------------------------------------
# tdgl3d
# ---------------------------------------------------------------------------

def _disk_material(params, radius: float, k_lo: int, k_hi: int):
    """A :class:`MaterialMap` holding a disk of metal, vacuum everywhere else.

    Built directly rather than through :class:`~tdgl3d.Trilayer`, which
    describes stacks of full-width layers; here the film has to be a
    disk in x-y so the closed forms apply.
    """
    from tdgl3d.core.material import MaterialMap

    centre_x = 0.5 * params.Nx * params.hx
    centre_y = 0.5 * params.Ny * params.hy
    x = np.arange(params.Nx + 1) * params.hx - centre_x
    y = np.arange(params.Ny + 1) * params.hy - centre_y
    X, Y = np.meshgrid(x, y, indexing="ij")
    inside = (X**2 + Y**2) <= radius**2
    # The full grid is i-fastest, so the (i, j) plane is transposed before ravel.
    plane = inside.T.ravel()

    sc_mask = np.zeros(params.dim_x)
    for k in range(k_lo, k_hi + 1):
        sc_mask[k * params.mk:(k + 1) * params.mk] = plane
    return MaterialMap(
        kappa=np.full(params.dim_x, params.kappa),
        sc_mask=sc_mask,
        interior_sc_mask=None,   # filled by the caller, which holds the indices
    )


def _cfl(params) -> float:
    """Forward-Euler limit, ``h²/(4κ²(d-1))`` — see ``tests/physics_helpers.py``."""
    h = min(params.hx, params.hy, params.hz if params.is_3d else np.inf)
    return h**2 / (4.0 * params.kappa**2 * (2.0 if params.is_3d else 1.0))


def _relaxed_psi(params, device, material, t_stop: float = 15.0):
    """The zero-field ψ profile, relaxed at an arbitrary κ.

    At ``B = 0`` with ``φ = 0`` the whole φ-block of the right-hand side
    vanishes identically — the curl-curl term because **A** is zero, the
    supercurrent term because ψ is real — so ``φ ≡ 0`` is an exact
    solution and the ψ-equation, which contains no κ, relaxes on its own.
    Doing this pass at ``κ = 1`` therefore gives *the same* profile as at
    the real κ while lifting the Courant limit by ``κ²``: the pair-breaking
    that the vacuum imposes on a film a few ξ thick is settled in a few
    hundred steps instead of tens of thousands.
    """
    from tdgl3d.physics.rhs import BoundaryVectors
    from tdgl3d.solvers.integrators import forward_euler

    cheap = params.copy()
    cheap.kappa = 1.0
    zero = np.zeros(params.dim_x)
    boundary = BoundaryVectors(zero, zero.copy(), zero.copy())
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data,
        cheap, device.idx, lambda t, X: boundary,
        0.0, t_stop, 0.4 * _cfl(cheap),
        save_every=10**9, progress=False, material=material,
    )
    return states[:, -1][:params.n_interior]


def run_tdgl3d(
    *,
    kappa: float = 8.0,
    radius: float = 6.0,
    thickness_cells: int = 4,
    spacing: float = 1.0,
    lateral_cells: int = 28,
    z_cells: int = 20,
    field: float = 0.005,
    diffusion_times: float = 2.5,
    dt_fraction: float = 0.8,
) -> DiskRun:
    """Solve the disk with tdgl3d: a real film of metal in a box of vacuum.

    ``Λ/R`` is not an input here.  The Pearl length comes out as
    ``κ²/∫|ψ|²dz``, and the pair-breaking that the surrounding vacuum
    imposes makes ``∫|ψ|²dz`` smaller than the geometric thickness — for
    a film a few ξ thick, substantially so.  Both the nominal
    ``κ²/d`` and the measured value are reported; ``lambda_over_r`` is
    the measured one, so the point lands where the physics it is actually
    solving puts it.

    The run time is set by the diffusion time of **A** across the box,
    ``L²/κ²``, which is why ``diffusion_times`` rather than an absolute
    time is the knob.  Steps needed for that scale as ``L²/h²`` and are
    independent of κ; what does scale with κ² is relaxing ψ, and
    :func:`_relaxed_psi` takes that out of the loop.
    """
    from tdgl3d import AppliedField, Device, SimulationParameters
    from tdgl3d.core.solution import Solution
    from tdgl3d.physics.applied_field import build_boundary_field_vectors
    from tdgl3d.physics.rhs import BoundaryVectors, eval_f
    from tdgl3d.solvers.integrators import forward_euler

    params = SimulationParameters(
        Nx=lateral_cells, Ny=lateral_cells, Nz=z_cells,
        hx=spacing, hy=spacing, hz=spacing, kappa=kappa,
    )
    device = Device(params, applied_field=AppliedField(Bz=field, t_on_fraction=1.0))
    k_lo = (z_cells - thickness_cells) // 2
    k_hi = k_lo + thickness_cells
    material = _disk_material(params, radius, k_lo, k_hi)
    material.interior_sc_mask = material.sc_mask[device.idx.interior_to_full]
    device._material = material

    state0 = np.zeros(params.n_state, dtype=np.complex128)
    state0[:params.n_interior] = _relaxed_psi(params, device, material)

    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, field, params, device.idx)
    )
    box = lateral_cells * spacing
    # A few diffusion times of **A** across the box, but never less than a
    # dozen ψ time constants: at large κ the field settles in well under one,
    # and it is then ψ's adjustment to the field that decides when the state
    # has stopped moving.
    t_stop = max(diffusion_times * box * box / kappa**2, 12.0)
    dt = dt_fraction * _cfl(params)
    start = time.perf_counter()
    _, states = forward_euler(
        state0, params, device.idx, lambda t, X: boundary,
        0.0, t_stop, dt, save_every=10**9, progress=False, material=material,
    )
    elapsed = time.perf_counter() - start
    state = states[:, -1]
    residual = float(np.abs(eval_f(state, params, device.idx, boundary, material)).max())

    solution = Solution(
        times=np.array([0.0, t_stop]),
        states=np.stack([state0, state], axis=1),
        params=params, idx=device.idx, device=device,
    )
    return _measure_tdgl3d(
        solution, params, material, radius, field, thickness_cells,
        {"seconds": elapsed, "residual": residual, "steps": int(t_stop / dt),
         "t_stop": t_stop, "dt": dt, "kappa": kappa, "spacing": spacing,
         "lateral_cells": lateral_cells, "z_cells": z_cells,
         "interior_nodes": params.n_interior},
    )


def _measure_tdgl3d(solution, params, material, radius, field, thickness_cells, meta):
    r"""Reduce a relaxed tdgl3d state to the same numbers the other codes give.

    The subtlety is that ``tdgl3d`` does not have a sheet current or a
    Pearl length; it has a current density in a film of finite thickness
    whose condensate the surrounding vacuum has partly broken.  What
    plays the role of the sheet superfluid density is
    :math:`n_s d = \int |\psi|^2\,\mathrm{d}z`, and everything else
    follows from the two moments of it,

    .. math:: S_0 = \int |\psi|^2\,\mathrm{d}V, \qquad
              S_2 = \int |\psi|^2 r^2\,\mathrm{d}V,

    as :math:`R_{\text{eff}} = \sqrt{2S_2/S_0}` (exact for a uniform
    disk), :math:`n_s d = S_0/\pi R_{\text{eff}}^2` and
    :math:`\Lambda = \kappa^2/n_s d`.  The London moment
    :math:`-\tfrac14 B S_2` is then the *same* closed form as the other
    two codes use, evaluated on the condensate this code actually has,
    so ``mu`` still means "how far below the weak-screening limit".

    ``|ψ|²`` is taken on the links rather than the nodes, because that is
    where ``J = Im[ψ* e^{-iφ} ψ']`` lives; using node values instead
    displaces the reference by half a cell exactly where ``|ψ|`` varies
    fastest, at the film's surfaces and rim.
    """
    Jx, Jy, _ = solution.supercurrent_density(step=-1)
    shape = (params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))
    Jx = Jx.reshape(shape)
    Jy = Jy.reshape(shape)
    psi = np.abs(solution.psi(step=-1).reshape(shape))

    dV = params.hx * params.hy * params.hz
    centre = 0.5 * params.Nx * params.hx
    x = (np.arange(1, params.Nx) * params.hx - centre)[:, None, None]
    y = (np.arange(1, params.Ny) * params.hy - centre)[None, :, None]

    # |ψ|² on the same links the current sits on.  ψ vanishes on the
    # boundary nodes the interior array stops short of.
    pad_i = np.concatenate([psi, np.zeros((1,) + shape[1:])], axis=0)
    pad_j = np.concatenate([psi, np.zeros((shape[0], 1, shape[2]))], axis=1)
    n_x = psi * pad_i[1:]
    n_y = psi * pad_j[:, 1:]

    moment = 0.5 * float(np.sum(x * Jy - y * Jx)) * dV
    S0 = 0.5 * float(np.sum(n_x + n_y)) * dV
    S2 = float(np.sum(x**2 * n_y + y**2 * n_x)) * dV
    radius_eff = float(np.sqrt(2.0 * S2 / S0))
    sheet_ns = S0 / (np.pi * radius_eff**2)
    Lambda_eff = params.kappa**2 / sheet_ns
    H_a = params.kappa**2 * field                     # μ₀ → 1/κ² in these units

    # K_φ(r) = ∫ J_φ dz, one value per column of the grid rather than
    # resampled onto the radii the mesh codes are read at: a Cartesian grid
    # has no nodes at arbitrary radii, and interpolating to pretend it does
    # would smooth the staircase this benchmark is trying to measure.  The
    # comparison downstream is the rms of the ratio, which does not need the
    # three codes to share a radial grid.
    r_node = np.sqrt(x**2 + y**2)[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        j_phi = np.where(r_node[:, :, None] > 0,
                         (x * Jy - y * Jx) / np.maximum(r_node[:, :, None], 1e-30), 0.0)
    K_phi = (np.sum(j_phi, axis=2) * params.hz).ravel()
    # Compared against the *local* sheet superfluid density, not the disk
    # average: the vacuum pair-breaks the rim, so ``n_s d`` falls away over
    # the last ξ or two of the radius.  Dividing by the average would fold
    # that into the profile error and read it as a screening disagreement,
    # which it is not — it is the same condensate the moment was normalised
    # by, resolved in r.
    local_ns = (np.sum(0.5 * (n_x + n_y), axis=2) * params.hz).ravel()
    frac = (r_node / radius_eff).ravel()
    limits = _profile_radii()
    keep = (frac >= limits[0]) & (frac <= limits[-1]) & (local_ns > 0.05 * sheet_ns)
    frac, K_phi, local_ns = frac[keep], K_phi[keep], local_ns[keep]
    order_by_radius = np.argsort(frac)
    frac, K_phi, local_ns = (a[order_by_radius] for a in (frac, K_phi, local_ns))
    k_ratio = K_phi / (-0.5 * local_ns * field * (frac * radius_eff))

    thickness = thickness_cells * params.hz
    meta.update({
        "moment_ideal": ideal_disk_moment(H_a, radius_eff),
        "psi_max": float(psi.max()),
        "sheet_ns": sheet_ns,
        "radius_eff": radius_eff,
        "radius_nominal": radius,
        "Lambda": Lambda_eff,
        "Lambda_nominal": params.kappa**2 / thickness,
        "lambda_over_r_nominal": (params.kappa**2 / thickness) / radius,
        "moment_london_nominal": london_disk_moment(
            H_a, radius, params.kappa**2 / thickness
        ),
        "thickness": thickness,
        "field": field,
    })
    return DiskRun(
        tool="tdgl3d",
        lambda_over_r=Lambda_eff / radius_eff,
        moment=moment,
        moment_london=-0.25 * field * S2,
        radius=frac,
        sheet_current=k_ratio,
        meta=meta,
    )
