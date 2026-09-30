# openBF Python port: source audit and proposed design

Date: 2026-09-28. Source: local `openBF/`, package version 2.8.0.
Status: proposals approved and implemented in `openbfpy/`. See [numerical notes](numerical-notes.md) for the decisions, derivations, and validation scope. The findings below retain the original audit wording for traceability.

## Objective and scope

Create an independent Python package, distributed as `openbfpy`, with no Julia runtime dependency. Preserve the governing equations and numerical methods except for explicitly agreed corrections. The user has requested discussion and correction of suspected bugs before porting. Therefore agreement with unmodified Julia output cannot be the sole acceptance criterion for corrected behavior.

The root currently contains only a Python project stub. The Julia solver comprises ten source files. Its Streamlit dashboard is already Python but is separate from the numerical solver; dashboard redesign is outside the proposed port.

## Numerical contract

- Float64 vessel area A, flow Q, velocity u, and pressure P.
- Pressure law: P = Pext + beta * (sqrt(A/A0) - 1).
- Wave speed: c = sqrt(1.5 * gamma * sqrt(A)).
- Mesh: M = max(configured M, 5, ceil(1000 L)); dx = L/M.
- Olufsen wall-thickness coefficients and the original velocity-profile friction term.
- Two-stage MUSCL finite-volume update, the existing slope limiter and local Lax-Friedrichs form of the flux, followed by source terms.
- All boundary/junction updates precede all vessel MUSCL advances. Graph topology is fixed and adjacency is precomputed.
- Junction unknowns are velocity and A^(1/4). Conjunctions enforce total-pressure continuity; bifurcations and anastomoses enforce the source's static-pressure continuity. This distinction must not be silently removed.
- Junction Newton iterations use residual norm threshold 1e-5 and a 30-iteration cap. The Windkessel scalar Newton solve currently takes exactly ten iterations.
- Existing YAML keys, inlet data format, five spatial waveform sample positions, and pressure RMSE conversion factor 133.332.

## Findings and proposed corrections

### 1. Outlet selection and resistance conversion

Evidence: `openBF/src/vessel.jl:223-228`.

Whenever R2 is zero, the constructor replaces it with R1 - rho*c/A. For an Rt-only outlet, R1 defaults to zero, so R2 becomes negative and `usewk3` becomes true. Compliance remains zero, leading to a division by zero in the Windkessel update. For the documented two-element configuration, the derived R2 is combined with the original R1, changing the specified total resistance.

Proposal: select the outlet type from explicitly supplied parameters. Reflection outlets must stay on the characteristic path. Reject nonphysical Windkessel parameters. Clarify whether the documented two-element option should mean a true two-element model or a total resistance split into proximal impedance and distal resistance; these are different models. Do not choose silently.

### 2. Inconsistent Windkessel endpoint state

Evidence: `openBF/src/boundary_conditions.jl:78-95`.

The nonlinear residual enforces P(As) - Pc = R1 * As * us. After solving, velocity is instead calculated from P(As) - Pout, and Q is left unchanged. These values generally do not satisfy the solved residual or Q = A*u.

Proposal: use us = (P(As) - Pc)/(R1*As) and set Q = As*us. Keep Pout in the distal capacitor discharge equation. Retain the original explicit capacitor update initially.

### 3. Taper geometry and source terms need an agreed convention

Evidence: `openBF/src/vessel.jl:162-179`.

The code sets radius_slope = (Rd - Rp)/(M - 1), then multiplies it by physical distance `(i - 1)*dx`. Thus the last radius is Rp + (Rd - Rp)*dx rather than Rd. The same slope is used as a spatial derivative. The taper threshold is 1e-4 m while its comment says 1 micrometre. The dTaudx expression uses 1.3 although beta is constructed with 1/(1 - 0.5^2) = 4/3.

Proposal: explicitly decide whether Rp and Rd refer to physical vessel faces or the first and last stored samples. Then derive radius, dA0dx, and the complete variable-wall source consistently from that convention and the stated pressure law. The geometry error is demonstrable; the exact corrected discretization and wall derivative need agreement and dedicated equilibrium tests. A direct substitution of the slope alone is insufficient.

### 4. Zero-flow inlet and characteristic compatibility

Evidence: `openBF/src/boundary_conditions.jl:38-47`.

After interpolating the outgoing characteristic, the code sets W21 = 2Q/A - W11. This gives u = Q/A using the old area, then assigns A = Q/u. A therefore remains essentially unchanged for nonzero flow and becomes 0/0 for zero flow. The new characteristic pair does not generally recover the same area through the wave-speed law.

Proposal: solve the incoming-boundary characteristic equation Q(t)/A - 4c(A) = Wminus for positive A, then update A, Q, and u consistently. This changes boundary numerics and needs explicit approval, root-selection rules, and tests for zero and reversing flow.

### 5. CFL bound misses the fastest characteristic in reverse flow

Evidence: `openBF/src/solver.jl:23`.

The code uses abs(u+c). The spectral radius of the two characteristic speeds u-c and u+c is abs(u)+c. These differ for reverse flow.

Proposal: use max(abs(u-c), abs(u+c)) in the CFL calculation. Fail explicitly for nonfinite states rather than skipping their speeds.

### 6. Viscoelastic state and operator verification

Evidence: `openBF/src/solver.jl:158-177`.

Velocity is updated before the viscoelastic solve replaces Q, leaving u inconsistent with the final Q/A. The tridiagonal construction also uses coefficient indexing whose correctness for variable Cv needs derivation: Julia's constructor accepts lower, diagonal, upper arrays, despite the local variable names used here.

Proposal: refresh u after the solve. Audit the variable-coefficient operator against the intended PDE before changing matrix entries; do not infer correctness from variable names. Preserve the existing operator until that question is resolved.

### 7. Initialization and pressure state

Evidence: `openBF/src/vessel.jl:151,198-201`; `openBF/src/output.jl:32-37`.

`initial_pressure` is read but unused. A starts at A0. The stored P array is initialized but never refreshed by the main solver; waveform output independently computes elastic pressure minus Pout.

Proposal: clarify whether initial_pressure is absolute pressure or transmural pressure. For absolute pressure, invert the wall law: A = A0 * (1 + (Pinitial-Pext)/beta)^2, requiring a positive unsquared factor. Preserve A=A0 when the option is absent. Keep stored P consistent with the state; document output pressure relative to Pout separately. Do not silently add the deprecated viscoelastic pressure postprocessing to output.

### 8. Graph validation

Evidence: `openBF/src/network.jl:126` and topology construction.

The mixed `||`/`&&` condition fails to throw when indegree exceeds two. It also does not reject a two-parent/two-child junction, which the available junction systems do not represent. Duplicate edges overwrite the lookup but remain in the vessel vector.

Proposal: validate supported junction degree combinations, unique edges, no self-loops, acyclic directed connectivity, and consistency of node identifiers before building arrays. Match graph traversal order during uncorrected reference comparisons.

### 9. Simulation time, error handling, and output

Evidence: `openBF/src/simulation.jl:137-197`.

The solver advances by dt and then records the advanced state with the old current_time. Sampling catches at most one checkpoint per step. Reaching the cycle/time limit is labelled convergence, even if the residual criterion fails. Exceptions are printed and swallowed. The preamble removes an existing results directory recursively and changes process working directory.

Proposal: record physical state time consistently, define how requested sample times are reached or interpolated, distinguish convergence from cycle-limit termination, propagate errors, and use explicit paths with safe output overwrite rules. Sampling changes affect waveform references and must be validated separately from kernel changes.

## Reference and tooling limitations

- Python 3.12 and uv are available; Julia was not found on PATH.
- Existing `.jls` files are Julia-serialized waveform references, not portable final-state arrays.
- `test/reference.jl` runs configured cycles/tolerances, contrary to AGENTS.md's description of deterministic three-cycle final-state capture.
- `test/runtests.jl` references a `test/network` fixture absent from this checkout.
- AGENTS.md's convergence warning is stale: the actual code computes true RMSE.
- The benchmark harness is `benchmark/benchmarks.jl`, not the path stated in AGENTS.md.

No Julia execution, numerical regression, or performance measurement has been performed. A Julia executable or exported reference data is needed for cross-language verification; the delivered Python runtime will not require Julia.

## Proposed Python design

Use NumPy float64 arrays, PyYAML configuration loading, and SciPy linear algebra where appropriate. Keep modules aligned with the scientific responsibilities: vessel geometry/state, network construction, MUSCL kernel, boundaries, junctions, simulation, and output. Expose `run_simulation` plus a result object with state, waveforms, termination reason, and timings. Preserve upstream Apache-2.0 attribution.

Prefer a readable array-based implementation with preallocated work buffers. Profile the canonical models before considering optional Numba acceleration. Do not enable fast-math or introduce parallelism before correctness is established.

Alternatives: a literal port gives the easiest legacy comparison but preserves known failures; a redesigned solver risks unnecessary numerical drift. Recommended: a closely mapped port with an explicit correction ledger and separately validated changes, retaining the original Julia checkout as an untouched reference.

## Validation sequence

1. Record source hashes and export Julia geometry, primitive functions, individual kernel steps, junction results, and complete waveforms in a portable format.
2. Test unchanged equations against these fixtures, reporting tolerances per quantity. Do not promise bitwise equality across Julia fast-math and Python linear algebra.
3. Validate each agreed correction independently using algebraic residuals, analytic cases, and where feasible a minimally corrected Julia reference. Numerical identity applies to the agreed corrected formulation.
4. Exercise zero/reverse inflow, Rt outlets, both resistance configurations, tapering, viscoelasticity, initialization, all junction types, and rejected topologies.
5. Run cca, ibif, adan56, and circle_of_willis; compare full time histories, convergence status, and step-level conservation. Include uta and the in-vitro model as broader integration coverage.
6. Measure runtime on identical meshes, cycle counts, convergence settings, output settings, and hardware. Publish measured results and unresolved limitations.

## Decisions recorded during implementation

The user approved proposals 1–9 and requested code comments explaining taper corrections. R1/Cc-only inputs use the recommended total-resistance split. Rp/Rd denote physical faces with cell-centered geometry; initial_pressure shares the reference of Pext. Waveforms use interpolation at fixed cycle phases. The existing variable-Cv matrix is retained with its limitation documented.

The user subsequently approved separate physical-unit junction stopping tolerances and sorting out-of-order inlet rows with a warning. Duplicate timestamps remain errors. Julia 1.13.1 was obtained in an ignored workspace validation directory, resolving the initial runtime limitation. Actual upstream MUSCL and junction functions were executed to generate portable CSV regression fixtures; the Python package does not depend on Julia.

Independent review found an additional inherited reflection-outlet inconsistency: updating u/Q without reconstructing A fails to enforce the two prescribed invariants. The user approved reconstructing A from c=(Wplus-Wminus)/8. Review also identified Windows filename collisions for labels differing only by case; the Python validation now rejects these.
