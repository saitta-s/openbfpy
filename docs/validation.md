# Validation report

Date: 2026-09-28. Implementation: openbfpy 0.1.0, with the corrections recorded
in [numerical notes](numerical-notes.md).

## Automated checks

- 27 unittest tests pass on both NumPy and optional Numba execution paths.
- Actual Julia 1.13.1 exports verify the unchanged elastic/viscoelastic MUSCL
  area and flow updates and all three junction residual/Jacobian systems.
  Relative tolerance is 1e-12, with absolute tolerances separated by units.
- Corrected boundary tests enforce the inlet characteristics, Windkessel resistor
  equation, and both reflection characteristics. Zero/reverse inflow and
  reflection coefficients -1, 0, 0.5 and 1 are exercised.
- Other checks cover analytic Jacobians, junction flow/pressure continuity,
  impedance matching, pressure initialization, tapered-source differentiation,
  viscoelastic state consistency, invalid graphs and states, Windows filename
  collisions, fixed-phase sampling, convergence versus cycle-limit termination,
  non-destructive output, and replay of saved configurations.
- The wheel builds successfully and imports independently of the source tree.
- SHA-256 checks confirm the original Julia source files are unchanged.

The taper interior rest residual, measured as max(abs(Q))/dt in the central half
of the vessel after one step, decreases with refinement:

| Cells | Residual (m^3/s^2) |
|---:|---:|
| 20 | 1.756e-7 |
| 40 | 8.128e-8 |
| 80 | 3.903e-8 |
| 160 | 1.911e-8 |

This verifies consistency in the interior, not exact preservation of tapered
equilibrium at the original constant-extrapolation ghost boundaries.

## Three-cycle integration results

All six bundled configurations were run for exactly three cardiac cycles, with
convergence stopping disabled and A, Q, u and P sampled. Every run completed
with finite waveforms and positive final cell areas. The cerebral inlet rows
were sorted with the approved warning. Runtime uses optional Numba acceleration.

| Model | Vessels | Steps | Minimum final A (m^2) | Last-cycle pressure RMSE (mmHg) | Observed time (s) |
|---|---:|---:|---:|---:|---:|
| cca | 1 | 25,645 | 2.8310e-5 | 3.2151 | 11.71 |
| ibif | 3 | 28,161 | 1.0740e-4 | 13.5169 | 36.74 |
| adan56 | 77 | 26,298 | 1.3843e-6 | 7.5196 | 806.96 |
| circle_of_willis | 33 | 101,186 | 1.7142e-6 | 11.2114 | 1038.00 |
| uta | 1 | 16,814 | 4.2788e-4 | 13.3142 | 5.77 |
| invitro | 37 | 30,977 | 8.2677e-6 | 24.6404 | 491.89 |

The model names are upstream names; the local adan56 configuration contains
77 vessels. These are finite-duration integration checks, not claims of periodic
convergence. Timings include first-use overhead and some concurrent validation
work, so they are observed durations rather than controlled performance benchmarks.
Raw results are in `validation-results.json` at the repository root.

## Optional acceleration comparison

Three timed repetitions after warmup compared identical 100-step runs. Final
A, Q, u, P and time values were checked at rtol=1e-10, atol=1e-14.

| Model | NumPy median (s) | Numba median (s) | Ratio | Maximum scaled difference |
|---|---:|---:|---:|---:|
| cca | 0.05346 | 0.02652 | 2.02x | 0 |
| ibif | 0.14489 | 0.12946 | 1.12x | 0 |
| adan56 | 6.05223 | 3.08817 | 1.96x | 4.11e-14 |

These compare the Python backends, not Python against Julia. The small
bifurcation timings have substantial noise, so no reliable speedup is claimed
for that case. Raw repetitions are in `backend-comparison.json`. No allocation
reduction, whole-simulation speedup, or superiority over Julia is claimed.

## Reproduction and limits

Environment: Windows, Python 3.12.8, NumPy 2.2.3, SciPy 1.15.2, PyYAML 6.0.3,
Numba 0.67.0; Julia fixture generation used Julia 1.13.1 and StaticArrays 1.9.22.

```sh
python -m unittest discover -s tests -v
python -m scripts.benchmark --cycles 3 --output validation-results.json
python -m scripts.compare_backends --steps 100
python -m scripts.check_taper
```

Install the `fast` extra for the backend comparison. Full-network runs can take
minutes per cycle. Full corrected waveforms are not expected to equal unmodified
Julia waveforms because of the approved changes. The variable-Cv viscoelastic
operator is translated and regression-tested but is not independently verified
against its intended continuum PDE. Clinical validity and long-time convergence
of every configuration are outside what these tests establish.
