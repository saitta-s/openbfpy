"""Two-stage MUSCL finite-volume method adapted from INSIGNEO openBF."""

import numpy as np
from scipy.linalg import solve_banded

from ._acceleration import kernel
from .boundaries import inlet, outlet
from .junctions import join
from .vessel import wave_speed


def calculate_dt(network):
    minimum = 1.0
    for v in network.vessels:
        if np.any(v.A <= 0) or not np.all(np.isfinite(v.A)) or not np.all(np.isfinite(v.u)):
            raise FloatingPointError(f"invalid CFL state in {v.label}")
        # Both eigenvalues matter when blood flow reverses.
        maximum_speed = np.max(np.abs(v.u)+wave_speed(v.A, v.gamma[1:-1]))
        minimum = min(minimum, v.dx/maximum_speed)
    return minimum*network.Ccfl


@kernel
def limit_slopes(slopes, values, inv_dx, half_dx):
    a = (values[1:-1]-values[:-2])*inv_dx
    b = (values[2:]-values[1:-1])*inv_dx
    t1 = np.maximum(np.minimum(a, 2*b), np.minimum(2*a, b))
    t2 = np.minimum(np.maximum(a, 2*b), np.maximum(2*a, b))
    slopes[0] = slopes[-1] = 0
    slopes[1:-1] = np.where(a > 0, np.where(b > 0, t1, 0), np.where(b < 0, t2, 0))*half_dx


@kernel
def halfstep_arrays(work, gamma, inv_dx, half_dx, dx_dt):
    flux_a, flux_q, a, q, slope_a, slope_q, al, ar, ql, qr, fl, fr = work
    limit_slopes(slope_a, a, inv_dx, half_dx)
    limit_slopes(slope_q, q, inv_dx, half_dx)
    al[:] = a+slope_a
    ar[:] = a-slope_a
    ql[:] = q+slope_q
    qr[:] = q-slope_q
    if np.any(al <= 0) or np.any(ar <= 0):
        raise FloatingPointError("MUSCL reconstructed nonpositive area")
    fl[:] = ql*ql/al + gamma*al*np.sqrt(al)
    fr[:] = qr*qr/ar + gamma*ar*np.sqrt(ar)
    flux_a[:-1] = .5*(qr[1:]+ql[:-1]-dx_dt*(ar[1:]-al[:-1]))
    flux_q[:-1] = .5*(fr[1:]+fl[:-1]-dx_dt*(qr[1:]-ql[:-1]))


def halfstep(v, dx_dt):
    halfstep_arrays(v.work, v.gamma, v.invDx, v.halfDx, dx_dt)


def tapered_source(v, rho):
    """Geometric source consistent with d(gamma*A**1.5)/dx + A/rho*dP/dx.

    Differentiation holds A fixed in the wall-law derivative. Both beta and
    A0 vary with x; the upstream dTaudx approximation omitted these terms.
    This is consistent but does not make the original flux exactly well balanced.
    """
    root_ratio = np.sqrt(v.A/v.A0)
    return (v.dbeta_dx/rho*v.A*(1-2*root_ratio/3)
            + v.beta/(3*rho)*v.A*root_ratio*v.dA0dx/v.A0)


def viscoelastic_step(v, dt):
    a = v.Cv*dt/(v.dx*v.dx)
    band = np.zeros((3, v.M))
    # scipy band rows are upper/diagonal/lower. Preserve the actual Julia
    # Tridiagonal(lower=-a[1:], diagonal, upper=-a[:-1]) operator, not its names.
    band[0, 1:] = -a[:-1]
    band[1] = 1+2*a
    band[1, 0] -= a[0]
    band[1, -1] -= a[-1]
    band[2, :-1] = -a[1:]
    rhs = (1-2*a)*v.Q
    rhs[0] += a[1]*v.Q[1]+a[0]*v.Q[0]
    rhs[1:-1] += a[:-2]*v.Q[:-2]+a[2:]*v.Q[2:]
    rhs[-1] += a[-2]*v.Q[-2]+a[-1]*v.Q[-1]
    v.Q[:] = solve_banded((1, 1), band, rhs, check_finite=False)
    v.refresh()


def muscl(v, dt, blood):
    dx_dt = v.dx/dt
    inv_dx_dt = 1/dx_dt
    v.vA[0], v.vQ[0] = v.U00A, v.U00Q
    v.vA[1:-1], v.vQ[1:-1] = v.A, v.Q
    v.vA[-1], v.vQ[-1] = v.UM1A, v.UM1Q
    halfstep(v, dx_dt)
    v.vA[1:-1] += inv_dx_dt*(v.fluxA[:v.M]-v.fluxA[1:v.M+1])
    v.vQ[1:-1] += inv_dx_dt*(v.fluxQ[:v.M]-v.fluxQ[1:v.M+1])
    v.vA[0], v.vQ[0] = v.vA[1], v.vQ[1]
    v.vA[-1], v.vQ[-1] = v.vA[-2], v.vQ[-2]
    halfstep(v, dx_dt)
    v.A[:] = .5*(v.A+v.vA[1:-1]+inv_dx_dt*(v.fluxA[:v.M]-v.fluxA[1:v.M+1]))
    v.Q[:] = .5*(v.Q+v.vQ[1:-1]+inv_dx_dt*(v.fluxQ[:v.M]-v.fluxQ[1:v.M+1]))
    if np.any(v.A <= 0):
        raise FloatingPointError(f"MUSCL produced nonpositive area in {v.label}")
    v.Q -= 2*(v.gamma_profile+2)*np.pi*blood.mu*v.Q/(v.A*blood.rho)*dt
    if v.tapered:
        v.Q += dt*tapered_source(v, blood.rho)
    if v.viscoelastic:
        viscoelastic_step(v, dt)
    else:
        v.refresh()


def solve(network, dt, current_time):
    """Apply each node once, then advance all vessels using fixed boundaries."""
    for node in network.node_order:
        parents = [network.vessels[e] for e in network.incoming.get(node, [])]
        children = [network.vessels[e] for e in network.outgoing.get(node, [])]
        if not parents:
            inlet(children[0], current_time, dt, network.heart)
        elif not children:
            outlet(parents[0], dt, network.blood.rho)
        else:
            join(parents, children, network.blood.rho)
    for eid in network.topo_order:
        v = network.vessels[eid]
        muscl(v, dt, network.blood)
        v.update_ghosts()
