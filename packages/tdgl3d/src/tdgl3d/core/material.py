"""Material map and layer definitions for multi-layer devices.

A :class:`Trilayer` stacks two superconducting films (``bottom``, ``top``)
separated by a dielectric ``insulator`` along the z-axis.  The helper
:func:`build_material_map` converts that description into a
:class:`MaterialMap` — flat per-node arrays that the operators read at
every time-step evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        Ginzburg–Landau parameter κ at every grid node.
    sc_mask : ndarray, shape (dim_x,)
        1.0 for superconductor nodes, 0.0 for insulator nodes.
    interior_sc_mask : ndarray, shape (n_interior,)
        Same mask but only for interior nodes (in interior ordering).
    """

    kappa: NDArray[np.float64]
    sc_mask: NDArray[np.float64]
    interior_sc_mask: NDArray[np.float64]

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
        Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
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
        Ginzburg–Landau parameter κ = λ/ξ for this material.
    is_superconductor : bool
        ``True`` for superconducting layers, ``False`` for insulators.
    """

    thickness_z: int
    kappa: float
    is_superconductor: bool = True


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
    """

    bottom: Layer
    insulator: Layer
    top: Layer

    def __post_init__(self) -> None:
        if self.insulator.is_superconductor:
            raise ValueError("The insulator layer must have is_superconductor=False.")

    @property
    def Nz(self) -> int:
        """Total number of z-cells required."""
        return self.bottom.thickness_z + self.insulator.thickness_z + self.top.thickness_z

    def z_ranges(self) -> dict[str, tuple[int, int]]:
        """Return ``{name: (k_start, k_end)}`` cell ranges (0-based, exclusive end)."""
        b = self.bottom.thickness_z
        i = self.insulator.thickness_z
        return {
            "bottom": (0, b),
            "insulator": (b, b + i),
            "top": (b + i, b + i + self.top.thickness_z),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_material_map(
    params: SimulationParameters,
    trilayer: Trilayer,
    idx,  # GridIndices — avoid circular import
) -> MaterialMap:
    """Construct per-node material arrays for a :class:`Trilayer`.

    The full grid has ``(Nx+1)×(Ny+1)×(Nz+1)`` nodes.  Node ``(i, j, k)``
    belongs to the layer whose z-cell range contains ``k``.  Boundary nodes
    at ``k = 0`` are part of the bottom layer and nodes at ``k = Nz`` are
    part of the top layer.

    Parameters
    ----------
    params : SimulationParameters
        Must have ``Nz == trilayer.Nz``.
    trilayer : Trilayer
    idx : GridIndices

    Returns
    -------
    MaterialMap
    """
    if params.Nz != trilayer.Nz:
        raise ValueError(
            f"params.Nz={params.Nz} does not match trilayer.Nz={trilayer.Nz}"
        )

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    mj = Nx + 1
    mk = (Nx + 1) * (Ny + 1)

    # Full-grid arrays
    kappa_full = np.empty(params.dim_x, dtype=np.float64)
    sc_mask_full = np.empty(params.dim_x, dtype=np.float64)

    ranges = trilayer.z_ranges()
    layers = {
        "bottom": trilayer.bottom,
        "insulator": trilayer.insulator,
        "top": trilayer.top,
    }

    # Fill per-node by z-plane
    for k in range(Nz + 1):
        # Determine which layer this z-plane belongs to
        # Node at k belongs to the layer whose cell range [k_start, k_end)
        # contains k (where k_end is the node boundary).
        layer_name = "top"  # default for last node
        for name, (k_start, k_end) in ranges.items():
            if k_start <= k < k_end:
                layer_name = name
                break
            # The very last node (k == Nz) belongs to the top layer
            if k == Nz:
                layer_name = "top"

        layer = layers[layer_name]

        # Linear indices for this z-plane (all i, j at this k)
        plane_start = k * mk
        plane_end = plane_start + mk
        kappa_full[plane_start:plane_end] = layer.kappa
        sc_mask_full[plane_start:plane_end] = 1.0 if layer.is_superconductor else 0.0

    # Interior-only mask
    interior_sc_mask = sc_mask_full[idx.interior_to_full]

    return MaterialMap(
        kappa=kappa_full,
        sc_mask=sc_mask_full,
        interior_sc_mask=interior_sc_mask,
    )
