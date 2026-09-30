"""Characteristic and three-element Windkessel boundary conditions."""

import numpy as np
from scipy.optimize import brentq

from .vessel import pressure, wave_speed


def _positive_root(function, guess, lower=None):
    """Bracket a positive root, optionally restricted to a monotonic branch."""
    lo = guess * 1e-8 if lower is None else lower
    hi = max(guess * 2, lo * 2)
    flo = function(lo)
    for _ in range(80):
        fhi = function(hi)
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
            return brentq(function, lo, hi, xtol=np.finfo(float).tiny, rtol=8*np.finfo(float).eps)
        hi *= 2
    raise FloatingPointError("cannot bracket a physical boundary area")


def inlet(v, time, dt, heart):
    c = wave_speed(v.A[:2], v.gamma[1:3])
    w = v.u[:2] - 4*c
    outgoing = w[0] + (w[1]-w[0]) * (c[0]-v.u[0]) * dt/v.dx
    q = heart.flow(time)
    k = np.sqrt(1.5*v.gamma[1])
    if q == 0:
        if outgoing >= 0:
            raise FloatingPointError("zero-flow inlet has no positive subcritical area")
        area = (-outgoing/(4*k))**4
    else:
        residual = lambda a: q/a - 4*k*a**0.25 - outgoing
        # For reverse flow the larger root is the subcritical branch (u+c>0).
        lower = (-q/k)**0.8 if q < 0 else None
        area = _positive_root(residual, v.A[0], lower)
    velocity = q/area
    if abs(velocity) >= wave_speed(area, v.gamma[1]):
        raise FloatingPointError("inlet characteristic boundary requires subcritical flow")
    v.A[0], v.Q[0], v.u[0] = area, q, velocity


def reflection(v, dt):
    c = wave_speed(v.A[-2:], v.gamma[-3:-1])
    wplus = v.u[-2:] + 4*c
    wp = wplus[-1] + (wplus[0]-wplus[1])*(v.u[-1]+c[-1])*dt/v.dx
    wm = v.W1M0 - v.Rt*(wp-v.W2M0)
    # Both invariants prescribe area as well as velocity. Keeping the old area
    # (as Julia did) violates the specified reflection coefficient.
    boundary_speed = (wp-wm)/8
    if not np.isfinite(boundary_speed) or boundary_speed <= 0:
        raise FloatingPointError("reflection boundary has nonpositive wave speed")
    v.A[-1] = (boundary_speed/np.sqrt(1.5*v.gamma[-2]))**4
    v.u[-1] = 0.5*(wm+wp)
    v.Q[-1] = v.A[-1]*v.u[-1]


def windkessel(v, dt, rho):
    if v.inlet_impedance_matching:
        v.R1 = rho*wave_speed(v.A[-1], v.gamma[-2])/v.A[-1]
        v.R2 = v.total_peripheral_resistance-v.R1
        if v.R2 <= 0:
            raise FloatingPointError("matched impedance exceeds total outlet resistance")
    v.Pc += dt/v.Cc*(v.Q[-1]-(v.Pc-v.Pout)/v.R2)
    old_area, old_u = v.A[-1], v.u[-1]
    sgamma = 2*np.sqrt(6*v.gamma[-2])
    fourth = np.sqrt(np.sqrt(old_area))
    bA0 = v.beta[-1]/np.sqrt(v.A0[-1])

    def residual(area):
        return (area*v.R1*(old_u+sgamma*(fourth-np.sqrt(np.sqrt(area))))
                - (v.Pext+bA0*(np.sqrt(area)-np.sqrt(v.A0[-1])))+v.Pc)

    area = old_area
    # Preserve the upstream ten Newton steps; reject invalid iterates explicitly.
    for _ in range(10):
        derivative = v.R1*(old_u+sgamma*(fourth-1.25*np.sqrt(np.sqrt(area)))) - 0.5*bA0/np.sqrt(area)
        if not np.isfinite(derivative) or derivative == 0:
            raise FloatingPointError("singular Windkessel Newton derivative")
        area -= residual(area)/derivative
        if not np.isfinite(area) or area <= 0:
            raise FloatingPointError("Windkessel Newton iteration produced nonpositive area")
    # Pc is the proximal resistor's downstream pressure; Pout belongs across R2.
    q = (pressure(area, v.A0[-1], v.beta[-1], v.Pext)-v.Pc)/v.R1
    v.A[-1], v.Q[-1], v.u[-1] = area, q, q/area


def outlet(v, dt, rho):
    if v.usewk3:
        windkessel(v, dt, rho)
    else:
        reflection(v, dt)
