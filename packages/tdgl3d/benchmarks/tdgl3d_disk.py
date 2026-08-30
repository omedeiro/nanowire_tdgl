"""Which disks tdgl3d is asked to solve, and why those.

``Λ/R`` is an input for the two thin-film codes and an outcome for this
one, so the sweep is over ``κ`` at fixed geometry: the Pearl length is
``κ²`` divided by the sheet superfluid density, so raising ``κ`` walks
the same axis the other two codes are swept along.  Everything else is
held fixed, which keeps the staircase approximation to the disk's rim
and the pair-breaking at its surfaces identical from point to point —
so what moves between points is screening and nothing else.

The cost does not grow with ``κ``.  The run has to last a few diffusion
times of **A** across the box, ``L²/κ²``, and the Courant limit is
``h²/8κ²``, so the step count is ``∝ L²/h²`` either way.  What does cost
``κ²`` is relaxing ψ, and :func:`~benchmarks.pearl_disk._relaxed_psi`
takes that out of the sweep by doing it once at ``κ = 1``, where it is
the same profile and forty times cheaper.

Reaching complete screening (``Λ/R ≪ 1``) is not affordable and is not
attempted: it needs ``κ² ≪ R·n_s d`` at a thickness large enough that the
vacuum has not pair-broken the film, in a box large enough that its walls
are far compared with ``R``.  The two thin-film codes cover that end.
"""

from __future__ import annotations

#: Each entry is keyword arguments for :func:`benchmarks.pearl_disk.run_tdgl3d`.
#:
#: κ starts at 6 rather than at 2.  The thin-film equation the other two
#: codes solve assumes ``d ≪ λ``, and with the four-cell film that a
#: pair-breaking vacuum leaves any condensate in at all, ``d/λ = 4/κ``.
#: Below κ ≈ 6 the film is not thin and the disagreement that follows is
#: a difference of model, not of discretisation — worth stating, not
#: worth sweeping.
SWEEP = [
    {"kappa": kappa, "radius": 6.0, "thickness_cells": 4,
     "lateral_cells": 28, "z_cells": 20, "spacing": 1.0}
    for kappa in (6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
]

#: Three runs that separate the two approximations the sweep above makes,
#: all at κ = 8 on the same physical film (R = 6 ξ, 4 ξ thick).
#:
#: The first pair differ only in how far the pinned walls are from the
#: film — 28 ξ across in the sweep, 40 ξ here — and so measure the
#: far-field boundary condition.  The second pair differ only in grid
#: spacing on a fixed 20 ξ box, and so measure the Cartesian staircase
#: around the rim; the box is shrunk for that pair because halving h on
#: the 28 ξ box is a nine-fold cost and would take about two hours.
#: Changing one thing at a time is the point: a single run at both a
#: bigger box and a finer grid would not say which of them moved the
#: answer.
CONVERGENCE = [
    {"kappa": 8.0, "radius": 6.0, "thickness_cells": 4,
     "lateral_cells": 40, "z_cells": 28, "spacing": 1.0},
    {"kappa": 8.0, "radius": 6.0, "thickness_cells": 4,
     "lateral_cells": 20, "z_cells": 14, "spacing": 1.0},
    {"kappa": 8.0, "radius": 6.0, "thickness_cells": 8,
     "lateral_cells": 40, "z_cells": 28, "spacing": 0.5},
]
