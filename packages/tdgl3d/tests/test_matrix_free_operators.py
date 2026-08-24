"""The fast paths must agree with the readable ones they replaced.

Three rewrites in the solver's hot path trade a readable implementation for a
faster one that is supposed to compute exactly the same thing:

* ``apply_LPSI`` / ``apply_LPHI_*`` against the assembled ``construct_*``
  matrices they stand in for,
* the vectorised hole geometry against a node-at-a-time reference,
* the Krylov solver's cached ``f_base`` against recomputing it.

Each is checked here against the thing it is meant to reproduce, on geometry
that exercises the parts the fast version could plausibly get wrong: anisotropic
grid spacing, a trilayer's non-uniform κ, and 2-D as well as 3-D.
"""

from __future__ import annotations

import numpy as np
import pytest
from tdgl3d.core import parallel
from tdgl3d.core.device import Device
from tdgl3d.core.material import Layer, Trilayer
from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.mesh.holes import (
    identify_boundary_links,
    identify_hole_nodes,
    identify_normal_boundary_links,
    point_in_polygon,
)
from tdgl3d.operators.sparse_operators import (
    apply_LPHI_x,
    apply_LPHI_y,
    apply_LPHI_z,
    apply_LPSI,
    construct_LPHI_x,
    construct_LPHI_y,
    construct_LPHI_z,
    construct_LPSI_x,
    construct_LPSI_y,
    construct_LPSI_z,
    kappa_sq_interior,
)
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.rhs import BoundaryVectors, eval_f
from tdgl3d.solvers.tgcr import tgcr_matrix_free, tgcr_matrix_free_trap

# Both paths do the same arithmetic in a different order, so they agree to
# round-off rather than exactly; 1e-13 relative is two orders above the noise.
RTOL = 1e-13


def _device(Nx, Ny, Nz, hx, hy, hz, trilayer=False):
    if trilayer:
        tri = Trilayer(
            bottom=Layer(thickness_z=2, kappa=2.0, is_superconductor=True),
            insulator=Layer(thickness_z=2, kappa=3.5, is_superconductor=False),
            top=Layer(thickness_z=2, kappa=2.0, is_superconductor=True),
        )
        params = SimulationParameters(
            Nx=Nx, Ny=Ny, Nz=tri.Nz, hx=hx, hy=hy, hz=hz, kappa=2.0
        )
        return Device(params, trilayer=tri)
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, hx=hx, hy=hy, hz=hz, kappa=2.7)
    return Device(params)


def _random_full(params, rng):
    return (
        rng.standard_normal(params.dim_x) + 1j * rng.standard_normal(params.dim_x)
    )


GEOMETRIES = [
    pytest.param((7, 9, 1, 1.0, 0.7, 1.0, False), id="2d-anisotropic"),
    pytest.param((6, 5, 4, 0.5, 0.8, 1.3, False), id="3d-anisotropic"),
    pytest.param((5, 6, 6, 1.0, 1.0, 1.0, True), id="trilayer"),
]


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_apply_matches_assembled_operators(geometry):
    """The matrix-free operators reproduce ``L[interior, :] @ vec``."""
    dev = _device(*geometry)
    params, idx, material = dev.params, dev.idx, dev.material
    m = idx.interior_to_full
    hx, hy, hz = params.hx, params.hy, params.hz

    rng = np.random.default_rng(20260823)
    x, y1, y2 = (_random_full(params, rng) for _ in range(3))
    y3 = (
        _random_full(params, rng)
        if params.is_3d
        else np.zeros(params.dim_x, dtype=np.complex128)
    )

    expected_psi = (
        construct_LPSI_x(y1, params, idx) / hx**2
        + construct_LPSI_y(y2, params, idx) / hy**2
        + construct_LPSI_z(y3, params, idx) / hz**2
    )[m, :] @ x
    got_psi, (fx, fy, fz) = apply_LPSI(x, y1, y2, y3, params, idx)
    assert np.allclose(got_psi, expected_psi, rtol=RTOL, atol=0.0)
    # Non-vacuous: a zero operator would also "agree" with itself.
    assert np.max(np.abs(expected_psi)) > 1.0

    for construct, apply_fn, vec in (
        (construct_LPHI_x, apply_LPHI_x, y1),
        (construct_LPHI_y, apply_LPHI_y, y2),
        (construct_LPHI_z, apply_LPHI_z, y3),
    ):
        expected = construct(params, idx, material)[m, :] @ vec
        got = apply_fn(vec, params, idx, material)
        assert np.allclose(got, expected, rtol=RTOL, atol=0.0)

    # The link factors handed back are the on-site Peierls phases.
    assert np.allclose(fx, np.exp(-1j * y1[m]), rtol=RTOL, atol=0.0)
    assert np.allclose(fy, np.exp(-1j * y2[m]), rtol=RTOL, atol=0.0)
    if params.is_3d:
        assert np.allclose(fz, np.exp(-1j * y3[m]), rtol=RTOL, atol=0.0)
    else:
        assert fz is None


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_chunked_rhs_matches_the_whole_array_operators(geometry):
    """``rhs_rows`` — the path ``eval_f`` takes — matches ``apply_*`` row for row.

    ``rhs_rows`` re-derives the same stencil in one pass over a block of nodes
    so each thread reads one neighbourhood.  That is a second implementation of
    arithmetic already checked against the assembled matrices above, so it has
    to be held against the first one or the two can drift apart silently.
    """
    from tdgl3d.operators.sparse_operators import (
        construct_FPHI_x,
        construct_FPHI_y,
        construct_FPHI_z,
        construct_FPSI,
        rhs_rows,
    )

    dev = _device(*geometry)
    params, idx, material = dev.params, dev.idx, dev.material
    n = params.n_interior

    rng = np.random.default_rng(555)
    x, y1, y2 = (_random_full(params, rng) for _ in range(3))
    y3 = (
        _random_full(params, rng)
        if params.is_3d
        else np.zeros(params.dim_x, dtype=np.complex128)
    )

    lpsi, (fx, fy, fz) = apply_LPSI(x, y1, y2, y3, params, idx)
    expected = [
        lpsi + construct_FPSI(x, params, idx, material),
        apply_LPHI_x(y1, params, idx, material)
        + construct_FPHI_x(x, y1, y2, y3, params, idx, material, link_factor=fx),
        apply_LPHI_y(y2, params, idx, material)
        + construct_FPHI_y(x, y1, y2, y3, params, idx, material, link_factor=fy),
        apply_LPHI_z(y3, params, idx, material)
        + construct_FPHI_z(x, y1, y2, y3, params, idx, material, link_factor=fz),
    ]

    # Split into uneven blocks so a chunk-boundary bug cannot hide.
    got = [np.zeros(n, dtype=np.complex128) for _ in range(4)]
    for rows in (slice(0, 1), slice(1, n // 3), slice(n // 3, n)):
        rhs_rows(x, y1, y2, y3, params, idx, material, rows, tuple(got))

    for block, (want, have) in enumerate(zip(expected, got)):
        if block == 3 and not params.is_3d:
            continue
        assert np.allclose(have, want, rtol=RTOL, atol=1e-12), f"block {block}"
    assert np.max(np.abs(expected[0])) > 1.0


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_eval_f_is_independent_of_thread_count(geometry):
    """Splitting the interior across threads must not change a single bit.

    The chunks are disjoint and each writes only its own rows, so the result is
    exactly the serial one — not merely close to it.
    """
    dev = _device(*geometry)
    params, idx = dev.params, dev.idx
    rng = np.random.default_rng(99)
    state = (
        rng.standard_normal(params.n_state) + 1j * rng.standard_normal(params.n_state)
    )
    u = BoundaryVectors(*build_boundary_field_vectors(0.02, -0.01, 0.05, params, idx))

    original_threads = parallel.get_num_threads()
    original_min = parallel.MIN_NODES_PER_THREAD
    try:
        # Force real splitting on grids far smaller than the production cutoff.
        parallel.MIN_NODES_PER_THREAD = 1
        parallel.set_num_threads(1)
        serial = eval_f(state, params, idx, u, material=dev.material)
        results = []
        for n_threads in (2, 3, 5):
            parallel.set_num_threads(n_threads)
            results.append(eval_f(state, params, idx, u, material=dev.material))
    finally:
        parallel.MIN_NODES_PER_THREAD = original_min
        parallel.set_num_threads(original_threads)

    assert np.isfinite(serial).all()
    assert np.max(np.abs(serial)) > 0.0
    for threaded in results:
        assert np.array_equal(serial, threaded)


def test_kappa_cache_follows_the_material():
    """The cached κ² is invalidated when the device's material map changes."""
    dev = _device(5, 6, 6, 1.0, 1.0, 1.0, trilayer=True)
    params, idx = dev.params, dev.idx

    uniform = kappa_sq_interior(params, idx, None)
    assert np.allclose(uniform, params.kappa**2)

    layered = kappa_sq_interior(params, idx, dev.material)
    assert np.allclose(layered, dev.material.kappa[idx.interior_to_full] ** 2)
    # The trilayer really is non-uniform, so the two answers must differ.
    assert not np.allclose(layered, uniform)

    # Asking again with the original argument must not return the other's value.
    assert np.allclose(kappa_sq_interior(params, idx, None), uniform)


def _hole_nodes_reference(vertices, z_range, hx, hy, Nx, Ny, Nz, edge_tolerance):
    """Node-at-a-time reference for :func:`identify_hole_nodes`."""
    mask = np.zeros((Nx + 1, Ny + 1, Nz + 1), dtype=bool)
    z_min, z_max = z_range
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if point_in_polygon((i * hx, j * hy), vertices, edge_tolerance):
                for k in range(z_min, min(z_max + 1, Nz + 1)):
                    mask[i, j, k] = True
    return mask


SQUARE = [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]
CONCAVE = [(1.0, 1.0), (9.0, 1.0), (9.0, 9.0), (5.0, 5.0), (1.0, 9.0)]
SLANTED = [(2.5, 1.5), (9.0, 2.0), (6.0, 8.5), (1.0, 6.0)]


@pytest.mark.parametrize("vertices", [SQUARE, CONCAVE, SLANTED])
@pytest.mark.parametrize("spacing", [(1.0, 1.0), (0.5, 0.75), (1.3, 0.4)])
@pytest.mark.parametrize("edge_tolerance", [1e-9, 0.0])
def test_vectorised_hole_nodes_match_node_at_a_time(vertices, spacing, edge_tolerance):
    """Carving a polygon gives the same nodes as testing them one by one."""
    hx, hy = spacing
    got = identify_hole_nodes(
        vertices, (1, 4), hx, hy, 12, 11, 6, edge_tolerance=edge_tolerance
    )
    expected = _hole_nodes_reference(
        vertices, (1, 4), hx, hy, 12, 11, 6, edge_tolerance
    )
    assert np.array_equal(got, expected)
    # Non-vacuous: the polygon must actually carve something.
    assert expected.any()


def _boundary_links_reference(hole_mask, direction, is_3d):
    """Loop-at-a-time reference for :func:`identify_boundary_links`."""
    Nx, Ny, Nz = (n - 1 for n in hole_mask.shape)
    mj, mk = Nx + 1, (Nx + 1) * (Ny + 1)
    links = []

    def linear(i, j, k):
        return k * mk + j * mj + i if is_3d else j * mj + i

    if direction == "x":
        for k in range(Nz + 1):
            for j in range(Ny + 1):
                for i in range(Nx):
                    if hole_mask[i, j, k] != hole_mask[i + 1, j, k]:
                        links.append(linear(i, j, k))
    elif direction == "y":
        for k in range(Nz + 1):
            for i in range(Nx + 1):
                for j in range(Ny):
                    if hole_mask[i, j, k] != hole_mask[i, j + 1, k]:
                        links.append(linear(i, j, k))
    else:
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                for k in range(Nz):
                    if hole_mask[i, j, k] != hole_mask[i, j, k + 1]:
                        links.append(linear(i, j, k))
    return np.array(links, dtype=np.int64)


@pytest.mark.parametrize("direction", ["x", "y", "z"])
@pytest.mark.parametrize("is_3d", [True, False])
def test_boundary_links_match_reference_including_order(direction, is_3d):
    """Same links, in the same order, as the loop-based implementation."""
    rng = np.random.default_rng(7)
    for fill in (0.15, 0.5):
        mask = rng.random((9, 8, 6)) < fill
        got = identify_boundary_links(mask, direction, is_3d)
        expected = _boundary_links_reference(mask, direction, is_3d)
        assert np.array_equal(got, expected)
        assert got.size > 0


@pytest.mark.parametrize("direction", ["x", "y"])
def test_normal_links_of_a_square_hole_are_its_two_faces(direction):
    """For a square hole the normal x-links are exactly the left/right faces.

    Every x-link that crosses the hole boundary does so at ``i = 6`` or
    ``i = 12`` for the rows the hole spans, and none of those has a crossing
    x-link beside it — so the classification must return all of them and
    nothing else.  The y-direction is the same statement rotated.
    """
    nx = ny = 20
    mask = np.zeros((nx + 1, ny + 1, 4), dtype=bool)
    mask[7:13, 7:13, :] = True  # 6x6 square hole through the stack
    mj, mk = nx + 1, (nx + 1) * (ny + 1)

    expected = set()
    for k in range(4):
        for other in range(7, 13):
            for face in (6, 12):
                i, j = (face, other) if direction == "x" else (other, face)
                expected.add(k * mk + j * mj + i)

    normal = identify_normal_boundary_links(mask, direction)
    boundary = set(identify_boundary_links(mask, direction).tolist())

    assert set(normal.tolist()) == expected
    assert set(normal.tolist()) <= boundary
    assert normal.size == 2 * 6 * 4


def test_normal_links_separate_normal_from_tangential():
    """Only crossings with no crossing neighbour along the link axis are normal.

    A void three cells deep is entered and left at well-separated k, so both of
    its z-crossings are isolated and count as normal.  A void one cell deep has
    its two crossings adjacent — each is the other's neighbour — so neither
    does.  With both present the normal links are a proper subset.
    """
    mask = np.zeros((13, 13, 9), dtype=bool)
    mask[4:7, 4:7, 3:6] = True   # three cells deep -> isolated crossings
    mask[9:11, 9:11, 4:5] = True  # one cell deep   -> adjacent crossings

    boundary = set(identify_boundary_links(mask, "z").tolist())
    normal = set(identify_normal_boundary_links(mask, "z").tolist())

    assert normal, "the deep void must contribute normal z-links"
    assert normal < boundary, "the one-cell void's crossings must be excluded"

    # Every excluded link belongs to the shallow void, and none of the deep
    # void's crossings was dropped.
    mj, mk = 13, 13 * 13
    deep = {
        k * mk + j * mj + i
        for k in (2, 5)
        for j in range(4, 7)
        for i in range(4, 7)
    }
    assert deep <= normal
    assert not (normal & (boundary - deep))


def test_gcr_cached_f_base_matches_recomputing_it():
    """Passing ``f_base`` in changes nothing but the number of ``eval_f`` calls."""
    rng = np.random.default_rng(11)
    n = 40
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A += np.eye(n) * n  # keep it well conditioned

    calls = {"n": 0}

    def f(v):
        calls["n"] += 1
        return A @ v

    x_lin = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    calls["n"] = 0
    without = tgcr_matrix_free(f, x_lin, b, tol=1e-10)
    n_without = calls["n"]

    base = f(x_lin)
    calls["n"] = 0  # count only what the solver itself spends
    with_base = tgcr_matrix_free(f, x_lin, b, tol=1e-10, f_base=base)
    n_with = calls["n"]

    assert without.size == n and with_base.size == n
    assert np.allclose(with_base, without, rtol=1e-12, atol=1e-12)
    # One evaluation per Krylov iteration is saved, so this must be a real cut.
    assert n_with < n_without

    calls["n"] = 0
    trap_without = tgcr_matrix_free_trap(f, x_lin, b, 0.1, tol=1e-10)
    calls["n"] = 0
    trap_with = tgcr_matrix_free_trap(f, x_lin, b, 0.1, tol=1e-10, f_base=f(x_lin))
    assert np.allclose(trap_with, trap_without, rtol=1e-12, atol=1e-12)
