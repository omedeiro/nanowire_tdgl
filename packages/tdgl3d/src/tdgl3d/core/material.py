"""Material map and layer definitions for multi-layer devices.

A :class:`Trilayer` stacks two superconducting films (``bottom``, ``top``)
separated by a dielectric ``insulator`` along the z-axis.  The helper
:func:`build_material_map` converts that description into a
:class:`MaterialMap` — flat per-node arrays that the operators read at
every time-step evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .parameters import SimulationParameters

# ---------------------------------------------------------------------------
# Per-node material arrays
# ---------------------------------------------------------------------------

@dataclass
class MaterialMap:
    """Per-node material properties on the full ``(Nx+1)×(Ny+1)×(Nz+1)`` grid.

    Attributes
    ----------
    kappa : ndarray, shape (dim_x,)
        Ginzburg-Landau parameter κ *as declared* for the layer each node
        belongs to.  This is a record of the device description; it is not
        what multiplies the Maxwell term — see ``magnetic_kappa``.
    sc_mask : ndarray, shape (dim_x,)
        1.0 for superconductor nodes, 0.0 for insulator/vacuum nodes.
    interior_sc_mask : ndarray, shape (n_interior,)
        Same mask but only for interior nodes (in interior ordering).
    magnetic_kappa : ndarray, shape (dim_x,), optional
        Coefficient κ multiplying the ``κ²∇×(∇×A)`` term at each node.
        ``None`` means "uniform ``params.kappa`` everywhere", which is the
        physically correct default — see the note below.

    Notes
    -----
    **κ in the Maxwell term is a property of the vacuum, not of the
    material.**  In the Ginzburg-Landau functional

    ::

        F = ∫ [ -|ψ|² + ½|ψ|⁴ + |(∇ - iA)ψ|² + κ²|∇×A|² ] dV

    the last term is the field energy B²/2μ₀ written in the units set by
    the *reference* material's λ and ξ.  It is carried by the field
    itself, so it has the same coefficient in the superconductor, in an
    oxide, in a hole and in vacuum; what distinguishes the materials is
    ψ — and hence the supercurrent — not the field energy.

    Two things follow.  A layer declared with ``kappa=0.0`` must *not*
    hand that zero to the Maxwell term: the φ-equation degenerates and
    **A** is frozen at its initial value, so the layer neither screens
    nor transmits.  And a non-uniform coefficient makes the discrete
    curl-curl operator non-self-adjoint, so it stops being the gradient
    of any energy and the free energy stops being a Lyapunov functional.

    :func:`build_material_map` therefore leaves ``magnetic_kappa`` at
    ``None`` unless a layer explicitly sets
    :attr:`Layer.magnetic_kappa`.  That override exists for the
    occasional model that wants a spatially varying coefficient; where it
    is used, the operators evaluate the coefficient on *plaquettes*
    rather than at nodes, which keeps the operator self-adjoint.
    """

    kappa: NDArray[np.float64]
    sc_mask: NDArray[np.float64]
    interior_sc_mask: NDArray[np.float64]
    magnetic_kappa: Optional[NDArray[np.float64]] = None

    def carve_hole_polygon(
        self,
        vertices: list[tuple[float, float]],
        z_range: tuple[int, int],
        params: SimulationParameters,
        idx,  # GridIndices — avoid circular import
    ) -> None:
        """Carve a polygon-shaped hole by marking nodes as non-superconducting.

        This modifies the material map to treat hole interior as insulator
        (sc_mask = 0.0).  The hole region is defined by a polygon in the
        x-y plane, extruded through the specified z-range.

        Parameters
        ----------
        vertices : list of (x, y) tuples
            Polygon vertices in physical coordinates (ξ units)
        z_range : (k_min, k_max)
            Z-layer extent (grid indices, inclusive)
        params : SimulationParameters
            Grid parameters (Nx, Ny, Nz, hx, hy, hz)
        idx : GridIndices
            Grid index mapping (for interior_to_full)

        Notes
        -----
        - Hole nodes have sc_mask set to 0.0 (non-superconducting)
        - This method is called automatically by Device.add_hole()
        - Can be called multiple times to create multiple holes

        Examples
        --------
        >>> square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        >>> material_map.carve_hole_polygon(square, (0, 5), params, idx)
        """
        from ..mesh.holes import identify_hole_nodes

        # Get hole mask (boolean array on full grid)
        hole_mask_3d = identify_hole_nodes(
            vertices=vertices,
            z_range=z_range,
            grid_spacing_x=params.hx,
            grid_spacing_y=params.hy,
            Nx=params.Nx,
            Ny=params.Ny,
            Nz=params.Nz,
        )

        # Convert 3D mask to linear indices
        # The full grid uses linear index m = i + mj*j + mk*k
        # But for 2D (Nz=1), we only use m = i + mj*j (no z component)
        Nx, Ny, _Nz = params.Nx, params.Ny, params.Nz
        mj = Nx + 1

        # Find all (i, j, k) where hole_mask_3d[i, j, k] == True
        ii, jj, kk = np.where(hole_mask_3d)

        # Compute linear indices based on dimensionality
        if params.is_3d:
            mk = (Nx + 1) * (Ny + 1)
            hole_linear_indices = ii + mj * jj + mk * kk
        else:
            # For 2D, ignore k (all nodes are at k=0)
            hole_linear_indices = ii + mj * jj

        # Mark hole nodes as non-superconducting
        self.sc_mask[hole_linear_indices] = 0.0

        # Update interior mask
        self.interior_sc_mask = self.sc_mask[idx.interior_to_full]


# ---------------------------------------------------------------------------
# Layer / Trilayer descriptors
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """Description of a single material layer.

    Parameters
    ----------
    thickness_z : int
        Number of grid **cells** along z occupied by this layer.
    kappa : float
        Ginzburg-Landau parameter κ = λ/ξ for this material.  For a
        superconducting layer this sets the screening length.  For a
        non-superconducting layer (oxide, vacuum) it is recorded but
        carries no physics: such a layer has no supercurrent, and the
        field energy in it is the vacuum one — see
        :attr:`magnetic_kappa` and the note on :class:`MaterialMap`.
    is_superconductor : bool
        ``True`` for superconducting layers, ``False`` for insulators
        and vacuum.
    magnetic_kappa : float, optional
        Override for the coefficient multiplying ``κ²∇×(∇×A)`` in this
        layer.  Leave it ``None`` (the default) unless you specifically
        want a spatially varying magnetic coefficient: the term is the
        field energy B²/2μ₀, which does not vary between materials.
        ``0.0`` freezes **A** in the layer and is never physical.
    """

    thickness_z: int
    kappa: float
    is_superconductor: bool = True
    magnetic_kappa: Optional[float] = None


@dataclass
class Trilayer:
    """Superconductor / Insulator / Superconductor stack.

    The three layers are stacked along z starting from k = 0:

    ::

        z = 0 ─── bottom SC ─── insulator ─── top SC ─── z = Nz

    Parameters
    ----------
    bottom, insulator, top : Layer
        The three layers.  ``insulator.is_superconductor`` must be ``False``.
    vacuum_below, vacuum_above : int
        Cells of vacuum to place under and over the stack.  With no
        padding the stack fills the box and the applied-field boundary
        condition is imposed **on the superconductor's own surface**,
        which pins the field there and leaves the film no room to expel
        flux.  Padding moves that condition out into vacuum, where it is
        the correct far-field statement, and lets the field outside the
        stack be solved for rather than prescribed.
    lateral_margin : int
        Cells of vacuum around the stack in x and y, so the stack is a
        *finite* slab with edges rather than one that runs into the wall
        of the box.  Flux expelled from the film has to go somewhere;
        without a lateral margin it has nowhere to go.
    vacuum_kappa : float, optional
        κ recorded for the vacuum padding.  Defaults to the reference
        ``params.kappa``.  Like any non-superconducting layer this
        carries no physics — vacuum has no supercurrent — and it does
        not change the Maxwell term.

    Notes
    -----
    Padding costs grid points, and the vacuum region has to be a few
    penetration depths thick before the boundary stops being felt: the
    field a distance d outside a screening film still differs from the
    applied field by roughly the film's own perturbation at that
    distance.  :func:`suggested_vacuum_cells` gives a starting value.
    """

    bottom: Layer
    insulator: Layer
    top: Layer
    vacuum_below: int = 0
    vacuum_above: int = 0
    lateral_margin: int = 0
    vacuum_kappa: Optional[float] = None

    def __post_init__(self) -> None:
        if self.insulator.is_superconductor:
            raise ValueError("The insulator layer must have is_superconductor=False.")
        for name in ("vacuum_below", "vacuum_above", "lateral_margin"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")

    def vacuum_layer(self, thickness_z: int, reference_kappa: float) -> Layer:
        """A vacuum :class:`Layer` of the given thickness."""
        kappa = self.vacuum_kappa if self.vacuum_kappa is not None else reference_kappa
        return Layer(thickness_z=thickness_z, kappa=kappa, is_superconductor=False)

    @property
    def Nz(self) -> int:
        """Total number of z-cells required, padding included."""
        return (
            self.vacuum_below
            + self.bottom.thickness_z
            + self.insulator.thickness_z
            + self.top.thickness_z
            + self.vacuum_above
        )

    def z_ranges(self) -> dict[str, tuple[int, int]]:
        """Return ``{name: (k_start, k_end)}`` cell ranges (0-based, exclusive end).

        Always carries ``"bottom"``, ``"insulator"`` and ``"top"``.  When
        the stack is padded it also carries ``"vacuum_below"`` and
        ``"vacuum_above"``; those keys are absent when the corresponding
        padding is zero, so ``"bottom"`` starting at 0 still means
        "the stack starts at the floor of the box".
        """
        v0 = self.vacuum_below
        b = v0 + self.bottom.thickness_z
        i = b + self.insulator.thickness_z
        t = i + self.top.thickness_z
        ranges = {"bottom": (v0, b), "insulator": (b, i), "top": (i, t)}
        if v0:
            ranges["vacuum_below"] = (0, v0)
        if self.vacuum_above:
            ranges["vacuum_above"] = (t, t + self.vacuum_above)
        return ranges

    @property
    def stack_z_range(self) -> tuple[int, int]:
        """Cell range spanned by the S/I/S stack itself, padding excluded."""
        r = self.z_ranges()
        return r["bottom"][0], r["top"][1]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def suggested_vacuum_cells(kappa: float, hz: float = 1.0) -> int:
    """Cells of vacuum padding worth putting either side of a stack.

    The field a distance *d* outside a screening film relaxes back to
    the applied field over the film's own scale, so the padding has to
    be a few λ = κ ξ thick before the boundary stops being felt.  Three
    penetration depths leaves a residual of order e⁻³ ≈ 5%; that is the
    value returned here, and it is a starting point rather than a
    converged answer — refine it against
    ``tests/test_verification_vacuum.py::test_far_field_converges``.
    """
    return max(1, int(np.ceil(3.0 * kappa / hz)))


def _lateral_vacuum_mask(
    params: SimulationParameters,
    margin: int,
) -> NDArray[np.bool_]:
    """Full-grid mask that is ``True`` on the outer *margin* cells in x and y."""
    Nx, Ny = params.Nx, params.Ny
    i = np.arange(Nx + 1)
    j = np.arange(Ny + 1)
    edge_i = (i < margin) | (i > Nx - margin)
    edge_j = (j < margin) | (j > Ny - margin)
    plane = edge_i[None, :] | edge_j[:, None]          # (j, i)
    nz = (params.Nz + 1) if params.is_3d else 1
    return np.broadcast_to(plane.ravel(), (nz, plane.size)).ravel()


def build_material_map(
    params: SimulationParameters,
    trilayer: Trilayer,
    idx,  # GridIndices — avoid circular import
) -> MaterialMap:
    """Construct per-node material arrays for a :class:`Trilayer`.

    The full grid has ``(Nx+1)×(Ny+1)×(Nz+1)`` nodes.  Node ``(i, j, k)``
    belongs to the layer whose z-cell range contains ``k``, and — when
    ``trilayer.lateral_margin`` is set — to vacuum if it lies within
    that many cells of the x or y wall of the box.

    Parameters
    ----------
    params : SimulationParameters
        Must have ``Nz == trilayer.Nz``.
    trilayer : Trilayer
    idx : GridIndices

    Returns
    -------
    MaterialMap

    Notes
    -----
    ``magnetic_kappa`` is left at ``None`` — meaning the uniform
    ``params.kappa``, the physically correct choice — unless some layer
    sets :attr:`Layer.magnetic_kappa`.  See :class:`MaterialMap` for why
    a per-layer κ must not reach the Maxwell term by default.
    """
    if params.Nz != trilayer.Nz:
        raise ValueError(
            f"params.Nz={params.Nz} does not match trilayer.Nz={trilayer.Nz}"
        )

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    mk = (Nx + 1) * (Ny + 1)

    # Full-grid arrays
    kappa_full = np.empty(params.dim_x, dtype=np.float64)
    sc_mask_full = np.empty(params.dim_x, dtype=np.float64)
    magnetic_full = np.empty(params.dim_x, dtype=np.float64)

    ranges = trilayer.z_ranges()
    layers = {
        "bottom": trilayer.bottom,
        "insulator": trilayer.insulator,
        "top": trilayer.top,
        "vacuum_below": trilayer.vacuum_layer(trilayer.vacuum_below, params.kappa),
        "vacuum_above": trilayer.vacuum_layer(trilayer.vacuum_above, params.kappa),
    }

    # Fill per-node by z-plane.
    #
    # Layer thicknesses are given in *cells*, but material properties live on
    # *nodes*, and the two interface nodes (k = b and k = b + i) are shared
    # between layers.  Assigning each node to the cell range [k_start, k_end)
    # that contains it hands the lower interface to the insulator and the upper
    # one to the top layer, which leaves the two superconducting layers with
    # different node counts — a symmetric stack that is not symmetric.
    #
    # Both interfaces go to the insulator instead.  A stack with equal
    # superconducting thicknesses then has equal superconducting node counts and
    # is exactly mirror-symmetric about its mid-plane; the oxide occupies
    # ``insulator.thickness_z + 1`` nodes, one per cell boundary it spans.
    #
    # The same rule applies at the vacuum/superconductor interfaces, so a
    # padded stack keeps the same node counts in its metal layers as an
    # unpadded one.
    insulator_start, insulator_end = ranges["insulator"]
    stack_start, stack_end = trilayer.stack_z_range

    for k in range(Nz + 1):
        if k < stack_start:
            layer_name = "vacuum_below"
        elif k > stack_end:
            layer_name = "vacuum_above"
        elif k < insulator_start:
            layer_name = "bottom"
        elif k <= insulator_end:
            layer_name = "insulator"
        else:
            layer_name = "top"

        layer = layers[layer_name]

        # Linear indices for this z-plane (all i, j at this k)
        plane_start = k * mk
        plane_end = plane_start + mk
        kappa_full[plane_start:plane_end] = layer.kappa
        sc_mask_full[plane_start:plane_end] = 1.0 if layer.is_superconductor else 0.0
        magnetic_full[plane_start:plane_end] = (
            params.kappa if layer.magnetic_kappa is None else layer.magnetic_kappa
        )

    # Lateral vacuum: a frame of non-superconducting nodes around the stack.
    if trilayer.lateral_margin > 0:
        frame = _lateral_vacuum_mask(params, trilayer.lateral_margin)
        vacuum = trilayer.vacuum_layer(0, params.kappa)
        sc_mask_full[frame] = 0.0
        kappa_full[frame] = vacuum.kappa

    # Only carry an explicit magnetic coefficient when some layer asked for
    # one; otherwise leave it None so the operators take the uniform fast path.
    declared = [layer.magnetic_kappa for layer in layers.values()]
    magnetic_kappa = None if all(v is None for v in declared) else magnetic_full

    # Interior-only mask
    interior_sc_mask = sc_mask_full[idx.interior_to_full]

    return MaterialMap(
        kappa=kappa_full,
        sc_mask=sc_mask_full,
        interior_sc_mask=interior_sc_mask,
        magnetic_kappa=magnetic_kappa,
    )
