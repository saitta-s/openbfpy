# Numerical formulation and approved corrections

The reference is the local openBF 2.8.0 source, whose SHA-256 hashes are recorded
in `tests/fixtures/julia/provenance.json`. The original source remains untouched.

## Preserved equations and algorithms

The elastic law is `P = Pext + beta*(sqrt(A/A0)-1)` with
`gamma = beta/(3*rho*sqrt(A0))` and `c = sqrt(1.5*gamma*sqrt(A))`.
The conservative flux is `(Q, Q^2/A + gamma*A^(3/2))`. The original slope limiter,
two-stage MUSCL update, friction source, and boundary-before-vessel update order
are preserved. The original ghost-cache timing is retained.

Junction unknowns are `(u_i, A_i^(1/4))`. The characteristic signs, flow balance,
and analytic Jacobians match Julia. Two-vessel conjunctions equate total pressure;
three-vessel joins equate elastic static pressure, excluding Pext as in Julia.
Pressure discontinuities from differing external pressures are not a supported
extension of the original junction model.

Waveform spatial samples preserve Julia's ties-to-even integer rounding and its
five sample indices. These are stored-cell positions, not interpolated physical
faces. The minimum mesh size remains `max(5, requested M, ceil(1000*L))`.

The friction key is `gamma_profile`, with default 2, exactly as in the Julia
source. Some bundled models spell it `gamma profile`; that spelling is not an
alias and retains the source's default behavior rather than silently changing
those models' friction law.

## Correction ledger

The user approved items 1–9 in the source audit, plus the junction tolerances,
sorting of out-of-order inlet samples, and reflection-state reconstruction
discovered during testing and review.

| Area | Implemented behavior |
|---|---|
| Outlet selection | Rt-only outlets use characteristics. Windkessel values must be physical. Conflicting Rt/Windkessel keys are rejected. |
| Reflection state | Reconstruct c=(Wplus-Wminus)/8 and A=(c/sqrt(1.5*gamma))^4, then update u and Q. The original unchanged-area update did not enforce both characteristics. |
| Resistance split | With absent or zero R2, supplied R1 means total resistance: proximal R1 becomes resting rho*c/A0, distal R2 is the remainder. This is a WK3 parameterization, not a true two-element RC model. |
| Impedance matching | R2 is total resistance minus the current impedance; a nonpositive remainder raises an error rather than being hidden by abs(). |
| WK3 state | Retain explicit capacitor stepping and ten Newton iterations, then use Q=(P-Pc)/R1 and u=Q/A. Invalid Newton iterates fail explicitly. |
| Taper | Rp/Rd are vessel-face radii. Cell centers are x=(i+1/2)*dx, radius slope is (Rd-Rp)/L, and any nonzero taper is represented. |
| Inlet | Solve Q/A-4c(A)=Wminus on the positive subcritical branch; zero flow has an analytic solution. Supercritical states are rejected. |
| CFL | Use abs(u)+c, including reverse flow; nonfinite states are errors. |
| Viscoelasticity | Preserve the upstream matrix and RHS; refresh u after solving for Q. |
| Initialization | Explicit initial_pressure uses the same reference as Pext; invert the elastic law. Without the key, A=A0. |
| Pressure | Keep the elastic P state current; waveform P is P-Pout. Do not enable the deprecated viscous pressure postprocessing. |
| Graphs | Reject duplicate labels/edges (including case-insensitive label collisions for portable filenames), self-loops, directed cycles, unsupported node degrees, and multiple inlets. Undirected loops with acyclic directed flow, including anastomoses, are supported. |
| Time/output | Record states at physical times; linearly interpolate fields to fixed cycle phases, and clip steps at cycle endpoints. Cycle-limit termination is distinct from convergence. |
| Inlet files | Sort out-of-order samples with a warning, preserving each time/flow pair; reject duplicate/nonfinite times and missing t=0. |

## Taper source derivation

Starting with momentum `Q_t + (Q^2/A)_x + A/rho*P_x = friction`, and putting
`gamma*A^(3/2)` into the numerical flux, the required geometric source is

```
S = gamma_x*A^(3/2) - (A/rho)*(partial P/partial x at fixed A)
  = (beta_x/rho)*A*(1 - (2/3)*sqrt(A/A0))
    + beta/(3*rho)*A*sqrt(A/A0)*(A0_x/A0).
```

`beta_x` follows analytically from beta=E*h/(0.75*R), including the derivative
of the Olufsen thickness when h0 is absent. A supplied h0 has zero derivative.
This replaces the inconsistent upstream dTaudx approximation and its 1.3 factor.
The original flux is not exactly well balanced for tapered resting vessels;
finite-mesh equilibrium drift remains and must be assessed by refinement.

## Newton tolerances

All junction equations must pass independently:

- Characteristics: max absolute residual <= `1e-10 + 1e-10*max(1,max(abs(W)))` m/s.
- Flow: absolute residual <= `1e-14 + 1e-10*max(abs(Q_i))` m^3/s.
- Pressure: max absolute residual <= `1e-5 + 1e-10*max(abs(Pelastic_i))` Pa.

The 30-step iteration cap is retained, with a RuntimeWarning on nonconvergence.
This corrects the upstream mixed-unit norm, which could skip solving small-flow
junctions entirely. Invalid or singular iterates propagate errors.

## Viscoelastic operator limitation

For `a=Cv*dt/dx^2`, the original lower diagonal is `-a[1:]` and upper diagonal
is `-a[:-1]` (zero-based notation). The RHS instead weights neighboring Q by
neighboring coefficients. This asymmetry is retained as approved; the port does
not claim that the variable-Cv operator is a newly verified discretization of
a particular viscoelastic PDE. The Julia reference test verifies its translation.

## Verification scope

Portable fixtures were produced by executing the actual upstream functions in
Julia 1.13.1. Elastic and viscoelastic MUSCL A/Q updates and all three junction
residual/Jacobian systems are compared with `rtol=1e-12`. Absolute allowances
are `1e-18` for area/flow (including junction flow rows), `1e-13` for
characteristic rows, and `1e-10` for pressure rows. The viscoelastic u
comparison explicitly accounts for the approved refresh correction.

Algebraic tests separately verify corrected boundary conditions, wall-law
initialization, source differentiation, conservation, fixed sample times and
termination. Full corrected trajectories are intentionally not identical to
unmodified Julia trajectories. Passing primitive comparisons is not evidence of
full-waveform equivalence or clinical validation.

Numba acceleration compiles the same Python array functions with fast-math off.
It is optional; `OPENBFPY_DISABLE_JIT=1` selects NumPy even if Numba is installed.
