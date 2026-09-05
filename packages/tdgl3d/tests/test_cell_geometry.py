"""Cell geometry for node-sampled data: the helpers that keep a picture honest.

Every field this solver produces is sampled *at* nodes, but the two matplotlib
calls that draw a filled grid want cell *boundaries*, and both fail quietly
when handed node coordinates:

* ``imshow(extent=...)`` reads the extent as the outer edge of the image, so n
  nodes get squeezed into n-1 cells' worth of axis and every pixel shifts half
  a cell;
* ``plot_surface(facecolors=...)`` colours the quad *between* nodes i and i+1,
  making an (n-1) x (m-1) mesh out of an n x m grid -- the last row and column
  are never drawn.

Neither raises.  Both show up as a device that is mirror-symmetric being drawn
lopsided, with the boundary band full width on the low edge and short on the
high one, which is what these helpers exist to prevent.  The tests below check
the geometry directly and then check it survives a real matplotlib call.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from tdgl3d.visualization.plotting import (  # noqa: E402
    cell_edges,
    imshow_extent,
    pad_facecolors,
    surface_facecolors,
)


@pytest.mark.parametrize("start,step,n", [(1.0, 1.0, 5), (0.0, 0.5, 8), (-3.0, 2.0, 3)])
def test_cell_edges_centre_every_node(start, step, n):
    """Each node must land exactly at the middle of the cell that carries it."""
    coords = start + step * np.arange(n)
    edges = cell_edges(coords)

    assert edges.shape == (n + 1,)
    np.testing.assert_allclose(0.5 * (edges[:-1] + edges[1:]), coords, atol=1e-12)
    np.testing.assert_allclose(np.diff(edges), step, atol=1e-12)


def test_cell_edges_limits_clamp_without_moving_the_interior():
    """Clamping trims the overhang and leaves every interior boundary alone.

    Four faces of one box are drawn separately, so their outermost cells have
    to stop at the corners instead of hanging half a cell past them.
    """
    coords = np.arange(1.0, 6.0)
    plain = cell_edges(coords)
    clamped = cell_edges(coords, limits=(coords[0], coords[-1]))

    assert clamped[0] == pytest.approx(coords[0])
    assert clamped[-1] == pytest.approx(coords[-1])
    np.testing.assert_allclose(clamped[1:-1], plain[1:-1], atol=1e-12)
    assert clamped.shape == plain.shape


def test_cell_edges_rejects_inverted_limits():
    with pytest.raises(ValueError):
        cell_edges(np.arange(3.0), limits=(2.0, 1.0))


def test_cell_edges_single_node():
    np.testing.assert_allclose(cell_edges(np.array([2.0])), [1.5, 2.5])


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.array([])])
def test_cell_edges_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        cell_edges(bad)


def test_imshow_extent_is_symmetric_about_the_data():
    """A symmetric set of nodes must produce a symmetric extent.

    This is the property that failed: with ``extent=[xs[0], xs[-1], ...]`` the
    image is narrower than the data it represents, so it cannot be centred on
    it and every pixel is offset.
    """
    xs = np.arange(1.0, 21.0)          # 20 nodes, centre 10.5
    ys = np.arange(1.0, 11.0)          # 10 nodes, centre 5.5
    x0, x1, y0, y1 = imshow_extent(xs, ys)

    assert (x0 + x1) / 2 == pytest.approx(xs.mean())
    assert (y0 + y1) / 2 == pytest.approx(ys.mean())
    # One cell per node, not one per gap.
    assert (x1 - x0) == pytest.approx(len(xs) * 1.0)
    assert (y1 - y0) == pytest.approx(len(ys) * 1.0)


def test_imshow_extent_places_pixels_on_their_nodes():
    """Round-trip through a real imshow: pixel k must sit on node k."""
    xs = np.arange(1.0, 6.0)
    ys = np.arange(2.0, 5.0)
    data = np.arange(15.0).reshape(len(xs), len(ys))

    fig, ax = plt.subplots()
    try:
        im = ax.imshow(data.T, origin="lower", extent=imshow_extent(xs, ys))
        x0, x1, y0, y1 = im.get_extent()
        pixel_w = (x1 - x0) / len(xs)
        pixel_h = (y1 - y0) / len(ys)
        centres_x = x0 + pixel_w * (0.5 + np.arange(len(xs)))
        centres_y = y0 + pixel_h * (0.5 + np.arange(len(ys)))
        np.testing.assert_allclose(centres_x, xs, atol=1e-12)
        np.testing.assert_allclose(centres_y, ys, atol=1e-12)
    finally:
        plt.close(fig)


def test_pad_facecolors_keeps_the_colours_and_adds_the_margin():
    rgba = np.random.default_rng(0).random((4, 6, 4))
    padded = pad_facecolors(rgba)

    assert padded.shape == (5, 7, 4)
    np.testing.assert_array_equal(padded[:-1, :-1], rgba)


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.zeros((3, 3, 3))])
def test_pad_facecolors_rejects_non_rgba(bad):
    with pytest.raises(ValueError):
        pad_facecolors(bad)


def test_surface_facecolors_rejects_non_2d():
    with pytest.raises(ValueError):
        surface_facecolors(np.zeros(4), plt.get_cmap("inferno"), Normalize(0, 1))


def test_plot_surface_draws_every_node_on_cell_edges():
    """The whole point: no data cell may go missing, and none may move.

    Passing node coordinates instead draws (n-1) x (m-1) quads -- for the 5 x 4
    grid here that is 12 of the 20 cells -- and offsets them half a cell.
    """
    xs = np.arange(1.0, 6.0)           # 5 nodes
    ys = np.arange(1.0, 5.0)           # 4 nodes
    data = np.arange(20.0).reshape(5, 4)
    cmap, norm = plt.get_cmap("inferno"), Normalize(0.0, 19.0)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    try:
        X, Y = np.meshgrid(cell_edges(xs), cell_edges(ys), indexing="ij")
        surf = ax.plot_surface(
            X, Y, np.zeros_like(X),
            facecolors=surface_facecolors(data, cmap, norm),
            shade=False, rstride=1, cstride=1,
        )
        fig.canvas.draw()
        assert len(surf.get_facecolors()) == data.size

        # The painted region spans one cell per node, centred on the data.
        assert X.min() == pytest.approx(xs[0] - 0.5)
        assert X.max() == pytest.approx(xs[-1] + 0.5)
        assert (X.min() + X.max()) / 2 == pytest.approx(xs.mean())
        assert (Y.min() + Y.max()) / 2 == pytest.approx(ys.mean())
    finally:
        plt.close(fig)


def test_node_coordinates_would_have_dropped_data():
    """Pin the behaviour the helpers exist to work around.

    If a matplotlib release ever makes ``plot_surface`` read the last row and
    column, this test fails and the helpers can be revisited -- rather than
    the workaround quietly outliving its reason.
    """
    xs = np.arange(1.0, 6.0)
    ys = np.arange(1.0, 5.0)
    data = np.arange(20.0).reshape(5, 4)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    try:
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        surf = ax.plot_surface(
            X, Y, np.zeros_like(X),
            facecolors=plt.get_cmap("inferno")(Normalize(0.0, 19.0)(data)),
            shade=False, rstride=1, cstride=1,
        )
        fig.canvas.draw()
        assert len(surf.get_facecolors()) == (len(xs) - 1) * (len(ys) - 1)
    finally:
        plt.close(fig)
