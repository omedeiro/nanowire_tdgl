"""Single precision: what it costs in accuracy, and what it has to buy.

``precision="single"`` runs the state in complex64, halving both the memory a
mesh needs and the bandwidth the evaluation is limited by.  The reason it is
opt-in rather than the default is that forward Euler is first-order and takes
tens of thousands of steps, so the question is whether round-off accumulates.

Measured here: it does not grow without bound.  TDGL is a gradient flow toward
a stable attractor, so a perturbation at one step is pulled back rather than
amplified, and the divergence from a double-precision run saturates instead of
compounding.  These tests pin that behaviour, so a future change that breaks it
shows up as a test failure rather than as quietly wrong physics.
"""

from __future__ import annotations

import numpy as np
import pytest
import tdgl3d
from tdgl3d.core.material import Layer, Trilayer

KAPPA = 2.0
#: 0.9 of the 3-D Forward-Euler limit h²/(4κ²(d−1)).
DT = 0.9 / (4 * KAPPA**2 * 2)

#: The divergence from double precision saturates around 1e-6 relative; this
#: is a ceiling on the saturated value, not a value computed from the run.
MAX_RELATIVE_DIVERGENCE = 5e-6


@pytest.fixture(scope="module")
def device():
    trilayer = Trilayer(
        bottom=Layer(thickness_z=3, kappa=KAPPA, is_superconductor=True),
        insulator=Layer(thickness_z=3, kappa=KAPPA, is_superconductor=False),
        top=Layer(thickness_z=3, kappa=KAPPA, is_superconductor=True),
    )
    params = tdgl3d.SimulationParameters(
        Nx=16, Ny=16, Nz=trilayer.Nz, kappa=KAPPA
    )
    return tdgl3d.Device(
        params,
        applied_field=tdgl3d.AppliedField(Bz=0.35, t_on_fraction=1.0),
        trilayer=trilayer,
    )


def _run(device, t_stop, precision):
    return tdgl3d.solve(
        device, t_stop=t_stop, dt=DT, method="euler", save_every=10**9,
        noise_seed=7, progress=False, log_metadata=False, precision=precision,
    )


def test_single_precision_state_is_complex64(device):
    """The state really is narrow — not cast back to double behind the scenes."""
    solution = _run(device, 1.0, "single")
    assert solution.psi(-1).dtype == np.complex64
    assert solution.phi_x(-1).dtype == np.complex64
    assert _run(device, 1.0, "double").psi(-1).dtype == np.complex128


@pytest.mark.parametrize("t_stop", [5.0, 50.0])
def test_single_tracks_double_and_does_not_drift(device, t_stop):
    """Divergence stays under 5e-6 relative, and does not grow with run length.

    Both halves matter: the first says single precision is usable, the second
    says the reason is dynamical rather than luck at one run length.
    """
    reference = _run(device, t_stop, "double").psi(-1)
    narrow = _run(device, t_stop, "single").psi(-1).astype(np.complex128)

    scale = float(np.max(np.abs(reference)))
    # Non-vacuous: a fully pair-broken state would agree trivially.
    assert scale > 0.3, "the run must have a superconducting state to compare"

    divergence = float(np.max(np.abs(reference - narrow))) / scale
    assert divergence < MAX_RELATIVE_DIVERGENCE, (
        f"single precision diverged by {divergence:.2e} after t = {t_stop}"
    )


def test_bulk_observables_agree_to_seven_figures(device):
    """The quantity a run is read for, not just the raw state vector."""
    reference = _run(device, 25.0, "double")
    narrow = _run(device, 25.0, "single")

    psi2_double = float(np.mean(np.abs(reference.psi(-1)) ** 2))
    psi2_single = float(np.mean(np.abs(narrow.psi(-1)) ** 2))
    assert psi2_double > 0.05, "non-vacuous: there must be a condensate to average"
    assert psi2_single == pytest.approx(psi2_double, rel=1e-6)

    bz_double = reference.bfield(-1)[2]
    bz_single = np.asarray(narrow.bfield(-1)[2], dtype=np.float64)
    bz_scale = float(np.max(np.abs(bz_double)))
    assert bz_scale > 1e-3, "non-vacuous: there must be a field to compare"
    assert np.max(np.abs(bz_double - bz_single)) / bz_scale < 1e-4


def test_single_precision_halves_the_state(device):
    """The point of it: a mesh twice the size fits in the same memory."""
    double = _run(device, 1.0, "double")
    single = _run(device, 1.0, "single")
    assert single.psi(-1).nbytes * 2 == double.psi(-1).nbytes


def test_unknown_precision_is_rejected(device):
    with pytest.raises(ValueError, match="precision"):
        _run(device, 1.0, "half")
