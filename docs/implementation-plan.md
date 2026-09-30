# Python solver implementation plan

Spec: [source audit](port-audit.md). User approved the corrections and requested implementation on 2026-09-28.

Status: all five implementation steps completed. See [validation report](validation.md) for measured results, independent review fixes, and remaining scientific limitations.

Use Python 3.12, NumPy, SciPy, PyYAML; no Julia runtime dependency. Execute in this session. Preserve upstream source and attribution.

1. Implement `openbfpy/vessel.py`: Float64 state, mesh, wall law, cell-centered tapered geometry, physical derivatives, initialization and outlet selection. Validate pressure inversion and resistance totals.
2. Implement `network.py`, `boundaries.py`, `junctions.py`: deterministic topology, inlet interpolation, characteristic/Windkessel boundaries, analytic junction Jacobians. Validate zero/reverse inflow, residuals, conservation and rejected graphs.
3. Implement `solver.py`: original MUSCL stages and limiter, corrected CFL, consistent tapered source, original viscoelastic matrix with refreshed velocity. Validate uniform equilibrium, limiter, timestep and matrix residuals.
4. Implement `simulation.py`, `output.py`, CLI and package metadata: physical timestamps, common cycle sampling by interpolation, convergence status, safe output. Validate end-to-end model runs and file contents.
5. Add portable Julia reference exporter, scientific regression tests, benchmark runner and documentation. Run the available checks and state limitations of cross-language verification.

Review focus: zero flow; reversing flow; nonpositive areas/invalid configuration; mixed topology including merges; output-directory collisions. Cover these with unittest tests and integration runs.

Numerical decisions: supplied R1 without R2 is total resistance split using resting characteristic impedance; Rp/Rd are physical face radii, sampled at cell centers; initial_pressure uses the same absolute/reference convention as Pext; interpolate saved states to fixed cycle-relative times without modifying CFL steps, except at cycle endpoints. Preserve original variable-Cv matrix pending separate model derivation. Every intentional departure is documented in code and the correction ledger.
