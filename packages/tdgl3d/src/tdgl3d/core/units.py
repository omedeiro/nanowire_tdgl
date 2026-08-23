"""Conversion between SI and the solver's Ginzburg-Landau units.

The solver is dimensionless: lengths in ξ, fields in Φ₀/(2πξ²), time in
τ_GL = ξ²/D.  A device specified in nanometres and millitesla therefore needs
one number to become a simulation — the coherence length ξ at the temperature
of interest — and that number is not a detail: it fixes the grid size, the field
scale, and whether the problem is tractable at all.

Ginzburg-Landau theory is a near-T_c expansion, and both characteristic lengths
diverge there as ``(1 - T/T_c)^{-1/2}``:

.. math::
    \\xi(T) = \\frac{\\xi_0}{\\sqrt{1 - T/T_c}}, \\qquad
    \\lambda(T) = \\frac{\\lambda_0}{\\sqrt{1 - T/T_c}} .

Their ratio κ does not, which is why κ is a material constant here while ξ is a
temperature-dependent input.  A micron-scale device is only a few tens of ξ
across near T_c and a few hundred well below it — the same geometry can be a
routine simulation or an impossible one depending on the temperature it is
posed at.

Examples
--------
>>> units = GLUnits(xi_nm=100.0, kappa=2.0)
>>> units.length(1000.0)                       # 1 µm in units of ξ
10.0
>>> round(units.field_unit_mT, 2)              # one unit of B, in mT
32.91
>>> round(units.field_to_mT(0.0433), 3)
1.425
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GLUnits", "PHI0_WB"]

#: Magnetic flux quantum, Φ₀ = h/2e, in webers.
PHI0_WB = 2.067833848e-15


@dataclass(frozen=True)
class GLUnits:
    """SI ↔ Ginzburg-Landau unit conversion for one material at one temperature.

    Parameters
    ----------
    xi_nm : float
        Coherence length ξ(T) in nanometres.  This is the temperature-dependent
        value, not ξ₀ — see :meth:`from_xi0_and_temperature`.
    kappa : float
        Ginzburg-Landau parameter κ = λ/ξ, temperature-independent.
    """

    xi_nm: float
    kappa: float

    def __post_init__(self) -> None:
        if self.xi_nm <= 0:
            raise ValueError("xi_nm must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")

    @classmethod
    def from_xi0_and_temperature(
        cls, xi0_nm: float, kappa: float, t_over_tc: float
    ) -> GLUnits:
        """Build from the zero-temperature ξ₀ and a reduced temperature.

        ``ξ(T) = ξ₀ / sqrt(1 - T/T_c)``.  Ginzburg-Landau theory is only valid
        for ``T`` close to ``T_c``, so ``t_over_tc`` well below ~0.5 should be
        read as an extrapolation.
        """
        if not 0.0 <= t_over_tc < 1.0:
            raise ValueError("t_over_tc must lie in [0, 1).")
        return cls(xi_nm=xi0_nm / (1.0 - t_over_tc) ** 0.5, kappa=kappa)

    # -- lengths -----------------------------------------------------------
    def length(self, nm: float) -> float:
        """Convert a length in nanometres to units of ξ."""
        return nm / self.xi_nm

    def length_nm(self, xi_units: float) -> float:
        """Convert a length in units of ξ back to nanometres."""
        return xi_units * self.xi_nm

    @property
    def lambda_nm(self) -> float:
        """London penetration depth λ = κξ, in nanometres."""
        return self.kappa * self.xi_nm

    # -- fields ------------------------------------------------------------
    @property
    def field_unit_T(self) -> float:
        """One unit of the solver's field, Φ₀/(2πξ²), in tesla."""
        xi_m = self.xi_nm * 1e-9
        return PHI0_WB / (2.0 * 3.141592653589793 * xi_m * xi_m)

    @property
    def field_unit_mT(self) -> float:
        """One unit of the solver's field, in millitesla."""
        return self.field_unit_T * 1e3

    def field(self, mT: float) -> float:
        """Convert a field in millitesla to solver units."""
        return mT / self.field_unit_mT

    def field_to_mT(self, gl_units: float) -> float:
        """Convert a field in solver units to millitesla."""
        return gl_units * self.field_unit_mT

    @property
    def hc2_mT(self) -> float:
        """Upper critical field, which is 1 in solver units, in millitesla."""
        return self.field_unit_mT

    # -- flux --------------------------------------------------------------
    def flux_quanta(self, area_xi2: float, field_gl: float) -> float:
        """Flux through *area_xi2* at *field_gl*, in units of Φ₀.

        In solver units Φ₀ = 2π, so this is ``B·A / 2π``.
        """
        return field_gl * area_xi2 / (2.0 * 3.141592653589793)

    def summary(self) -> str:
        return (
            f"ξ = {self.xi_nm:g} nm, κ = {self.kappa:g} "
            f"(λ = {self.lambda_nm:g} nm); "
            f"B unit = {self.field_unit_mT:.3g} mT, H_c2 = {self.hc2_mT:.3g} mT"
        )
