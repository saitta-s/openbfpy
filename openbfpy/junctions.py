"""Analytic Newton systems from openBF, using velocity and fourth-root area."""

import warnings

import numpy as np

from ._acceleration import kernel

@kernel
def residual_jacobian(state, signs, k, invariants, beta, reference_area, rho, total_pressure):
    """Return the original conjunction/bifurcation/anastomosis equations."""
    n = len(signs)
    u, root = state[:n], state[n:]
    area = root*root*root*root
    f = np.zeros(2*n)
    j = np.zeros((2*n, 2*n))
    f[:n] = u + signs*4*k*root - invariants
    for i in range(n):
        j[i, i] = 1
        j[i, n+i] = signs[i]*4*k[i]
    f[n] = np.sum(signs*u*area)
    j[n, :n] = signs*area
    j[n, n:] = signs*4*u*root**3
    p = beta*(root*root/np.sqrt(reference_area)-1)
    if total_pressure:
        p += .5*rho*u*u
    # Merge: compare each parent with the child; split: parent with each child.
    pairs = [(i, n-1) for i in range(n-1)] if np.sum(signs == 1) == 2 else [(0, i) for i in range(1, n)]
    for row, (a, b) in enumerate(pairs, n+1):
        f[row] = p[a]-p[b]
        j[row, n+a] = 2*beta[a]*root[a]/np.sqrt(reference_area[a])
        j[row, n+b] = -2*beta[b]*root[b]/np.sqrt(reference_area[b])
        if total_pressure:
            j[row, a], j[row, b] = rho*u[a], -rho*u[b]
    return f, j


def join(parents, children, rho):
    vessels = parents + children
    n = len(vessels)
    if (len(parents), len(children)) not in {(1, 1), (1, 2), (2, 1)}:
        raise ValueError("unsupported junction")
    indices = [-1]*len(parents) + [0]*len(children)
    signs = np.array([1.0]*len(parents) + [-1.0]*len(children))
    u = np.array([v.u[i] for v, i in zip(vessels, indices)])
    root = np.array([np.sqrt(np.sqrt(v.A[i])) for v, i in zip(vessels, indices)])
    beta = np.array([v.beta[i] for v, i in zip(vessels, indices)])
    a0 = np.array([v.A0[i] for v, i in zip(vessels, indices)])
    k = np.array([np.sqrt(1.5*v.gamma[i]) for v, i in zip(vessels, indices)])
    invariants = u+signs*4*k*root
    state = np.concatenate((u, root))
    for iteration in range(31):
        f, j = residual_jacobian(state, signs, k, invariants, beta, a0, rho, n == 2)
        if not np.all(np.isfinite(f)) or np.any(state[n:] <= 0):
            raise FloatingPointError("invalid junction state")
        # A raw norm mixes m/s, m^3/s and Pa, allowing clinically large flow
        # imbalances below 1e-5. Check each equation in its own physical units.
        speed_scale = max(1., float(np.max(np.abs(invariants))))
        flow_scale = float(np.max(np.abs(state[:n]*state[n:]**4)))
        pressure_scale = float(np.max(np.abs(beta*(state[n:]**2/np.sqrt(a0)-1))))
        if (np.max(np.abs(f[:n])) <= 1e-10+1e-10*speed_scale
                and abs(f[n]) <= 1e-14+1e-10*flow_scale
                and np.max(np.abs(f[n+1:])) <= 1e-5+1e-10*pressure_scale):
            break
        if iteration == 30:
            warnings.warn("junction Newton iteration did not converge", RuntimeWarning, stacklevel=2)
            break
        state += np.linalg.solve(j, -f)
    for a, (v, i) in enumerate(zip(vessels, indices)):
        v.u[i], v.A[i] = state[a], state[n+a]**4
        v.Q[i] = v.u[i]*v.A[i]
