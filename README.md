# openbfpy

A Python implementation of the openBF 2.8.0 one-dimensional blood-flow solver.
It includes the MUSCL finite-volume scheme, elastic and viscoelastic vessels,
characteristic and Windkessel boundaries, and conjunction/bifurcation/anastomosis
junctions. It requires no Julia runtime.

This port contains agreed numerical corrections. See
[numerical notes](docs/numerical-notes.md) for the exact equations, conventions,
differences from Julia, and validation limits.
The [validation report](docs/validation.md) records the tests and model runs.

## Install and run

Python 3.12 or newer:

```sh
python -m pip install -e .
python -m openbfpy openBF/models/boileau2015/cca/cca.yaml --savedir cca_results
```

Optional acceleration of the numerical array kernels:

```sh
python -m pip install -e ".[fast]"
```

The first accelerated run includes compilation; subsequent runs use cached code.
Set environment variable `OPENBFPY_DISABLE_JIT=1` to use the NumPy implementation.

```python
from openbfpy import run_simulation

result = run_simulation(
    "openBF/models/boileau2015/cca/cca.yaml",
    write_output=False,
    verbose=False,
)
print(result.termination_reason, result.convergence_error)
vessel = result.network.vessels[0]
pressure_waveform = result.waveforms[vessel.label]["P"]
```

The existing YAML format is accepted. Relative inlet paths resolve beside the
YAML file. Quantities use SI units; pressure convergence tolerance uses mmHg.
`R1` plus `Cc` without `R2` means total resistance split into a WK3 model.
`Rp`/`Rd` are vessel-face radii. Explicit `initial_pressure` uses the same pressure
reference as `Pext`; it is not transmural pressure.

Output matrices have six columns: time plus five spatial samples. `.last` stores
the final cycle; `--out-files` also appends cycles to `.out`. `--save-stats` writes
`stats.json`. The output directory must be empty: existing results are never
deleted automatically. `--no-output` disables filesystem output. Python API
exceptions propagate; a cycle limit is reported separately from convergence.

## Tests and reference checks

```sh
python -m unittest discover -s tests -v
python -m scripts.benchmark --cycles 3
```

Large network cases can take several minutes per cycle. For a shorter integration
check, add `--models cca ibif`. Measured backend comparisons are recorded in
`backend-comparison.json`; these compare NumPy and optional Numba, not Julia.

The tests include portable fixtures exported from the unmodified Julia MUSCL
and junction functions. Julia is needed only to regenerate these fixtures:

```sh
julia scripts/export_kernel_reference.jl tests/fixtures/julia
julia --project=YOUR_VALIDATION_ENV scripts/export_junction_reference.jl tests/fixtures/julia
```

The junction exporter requires StaticArrays in that validation environment.
The reference scripts load source files directly and do not modify them.

## Attribution

Adapted from INSIGNEO openBF under Apache-2.0. See [NOTICE](NOTICE),
[LICENSE](LICENSE), and the original [citation](openBF/CITATION.bib).
