"""Vessel geometry and state, adapted from INSIGNEO openBF (Apache-2.0)."""

from dataclasses import dataclass
import math

import numpy as np


def pressure(area, reference_area, beta, external_pressure=0.0):
    """Elastic pressure in Pa in the same reference frame as external pressure."""
    return external_pressure + beta * (np.sqrt(area / reference_area) - 1.0)


def wave_speed(area, gamma):
    """Characteristic wave speed in m/s."""
    return np.sqrt(1.5 * gamma * np.sqrt(area))


def finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def positive(value, name):
    value = finite_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Blood:
    mu: float
    rho: float

    def __post_init__(self):
        object.__setattr__(self, "mu", finite_float(self.mu, "mu"))
        object.__setattr__(self, "rho", positive(self.rho, "rho"))
        if self.mu < 0:
            raise ValueError("mu must be nonnegative")

    @property
    def Cf(self):
        return 8.0 * np.pi * self.mu / self.rho


class Vessel:
    """Fixed vessel geometry and mutable cell-centered Float64 state.

    Rp/Rd denote physical face radii. An explicit initial_pressure uses the
    same pressure reference as Pext; omission initializes A to A0.
    """

    def __init__(self, config, blood):
        self.label = str(config["label"])
        if not self.label or any(c in self.label for c in '/\\:*?"<>|') or self.label in (".", ".."):
            raise ValueError("vessel label must be a safe filename component")
        self.sn, self.tn = config["sn"], config["tn"]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (self.sn, self.tn)):
            raise ValueError("node identifiers must be positive integers")
        self.tosave = bool(config.get("to_save", True))
        self.L = positive(config["L"], "L")
        requested = config.get("M", 5)
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("M must be a positive integer")
        self.M = max(requested, 5, math.ceil(self.L * 1e3))
        self.dx = self.L / self.M
        self.invDx, self.halfDx = 1.0 / self.dx, 0.5 * self.dx
        self.x = (np.arange(self.M, dtype=float) + 0.5) * self.dx
        rp = positive(config.get("Rp", config.get("R0", 0)), "Rp/R0")
        rd = positive(config.get("Rd", rp), "Rd")
        # Correction to Julia: Rp/Rd are face values, and dR/dx has units m/m.
        # Julia multiplied a per-index increment by dx, so it never reached Rd.
        radius_slope = (rd - rp) / self.L
        self.radius = rp + radius_slope * self.x
        self.tapered = rd != rp
        self.A0 = np.pi * self.radius * self.radius
        self.dA0dx = 2.0 * np.pi * self.radius * radius_slope
        young = positive(config["E"], "E")
        if "h0" in config:
            thickness = np.full(self.M, positive(config["h0"], "h0"))
            dhdx = np.zeros(self.M)
        else:
            r = self.radius
            first, second = 0.2802 * np.exp(-505.3 * r), 0.1324 * np.exp(-11.14 * r)
            thickness = r * (first + second)
            dhdx = radius_slope * (first + second + r * (-505.3 * first - 11.14 * second))
        self.beta = np.sqrt(np.pi / self.A0) * thickness * young / 0.75
        self.dbeta_dx = young / 0.75 * (dhdx / self.radius - thickness * radius_slope / self.radius**2)
        interior_gamma = self.beta / (3.0 * blood.rho * np.sqrt(self.A0))
        self.gamma = np.pad(interior_gamma, 1, mode="edge")
        self.Pext = finite_float(config.get("Pext", 0.0), "Pext")
        self.Pout = finite_float(config.get("Pout", 0.0), "Pout")
        self.A = self.A0.copy()
        if "initial_pressure" in config:
            factor = 1.0 + (finite_float(config["initial_pressure"], "initial_pressure") - self.Pext) / self.beta
            if np.any(factor <= 0):
                raise ValueError("initial_pressure requires nonpositive sqrt(A/A0)")
            self.A *= factor * factor
        self.Q = np.full(self.M, finite_float(config.get("initial_flow", 0.0), "initial_flow"))
        self.u = self.Q / self.A
        self.P = pressure(self.A, self.A0, self.beta, self.Pext)
        self.gamma_profile = finite_float(config.get("gamma_profile", 2), "gamma_profile")
        if self.gamma_profile < 0:
            raise ValueError("gamma_profile must be nonnegative")
        self.viscoelastic = bool(config.get("visco-elastic", False))
        self.Cv = ((0.00150 * 0.5 / self.radius + 0.6) / (blood.rho * np.sqrt(self.A0))
                   if self.viscoelastic else np.zeros(self.M))
        c = wave_speed(self.A[-1], self.gamma[-2])
        self.W1M0, self.W2M0 = self.u[-1] - 4*c, self.u[-1] + 4*c
        self.Rt = finite_float(config.get("Rt", 0.0), "Rt")
        self.R1 = self.R2 = self.Cc = self.Pc = 0.0
        self.total_peripheral_resistance = 0.0
        self.inlet_impedance_matching = bool(config.get("inlet_impedance_matching", False))
        self.outlet_config = config
        self.usewk3 = False
        # Match Julia's ties-to-even spatial sampling, converting to zero indexing.
        self.sample_indices = np.array([0, round(self.M*.25)-1, round(self.M*.5)-1,
                                        round(self.M*.75)-1, self.M-1])
        self.work = np.zeros((12, self.M + 2))
        for row, name in enumerate(("fluxA", "fluxQ", "vA", "vQ", "slopesA", "slopesQ", "Al", "Ar", "Ql", "Qr", "Fl", "Fr")):
            setattr(self, name, self.work[row])
        self.update_ghosts()

    def configure_outlet(self, blood):
        config = self.outlet_config
        self.usewk3 = any(k in config for k in ("R1", "R2", "Cc"))
        if not self.usewk3:
            if not -1 <= self.Rt <= 1:
                raise ValueError("Rt must lie in [-1, 1]")
            return
        if "Rt" in config:
            raise ValueError("specify reflection or Windkessel parameters, not both")
        self.Cc = positive(config.get("Cc", 0), "Cc")
        supplied_r1 = positive(config.get("R1", 0), "R1")
        if "R2" in config and float(config["R2"]) != 0:
            self.R1 = supplied_r1
            self.R2 = positive(config["R2"], "R2")
        else:
            # Legacy two-parameter input denotes TOTAL resistance, split into WK3.
            self.R1 = blood.rho * wave_speed(self.A0[-1], self.gamma[-2]) / self.A0[-1]
            self.R2 = positive(supplied_r1 - self.R1, "total resistance minus impedance")
        self.total_peripheral_resistance = self.R1 + self.R2

    def refresh(self):
        if np.any(self.A <= 0) or not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.Q)):
            raise FloatingPointError(f"nonphysical state in vessel {self.label}")
        np.divide(self.Q, self.A, out=self.u)
        self.P[:] = pressure(self.A, self.A0, self.beta, self.Pext)

    def update_ghosts(self):
        self.U00A, self.UM1A = self.A[0], self.A[-1]
        self.U00Q, self.UM1Q = self.Q[0], self.Q[-1]
