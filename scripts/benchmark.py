"""Run fixed-cycle canonical cases: python -m scripts.benchmark --cycles 3."""

import argparse
import json
from pathlib import Path
import tempfile
import sys

import numpy as np
import yaml

from openbfpy import run_simulation


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "cca": "boileau2015/cca/cca.yaml",
    "ibif": "boileau2015/ibif/ibif.yaml",
    "adan56": "boileau2015/adan56/adan56.yaml",
    "circle_of_willis": "alastruey2007/circle_of_willis.yaml",
    "uta": "boileau2015/uta/uta.yaml",
    "invitro": "matthys2007/invitro_model.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--output", type=Path, default=ROOT/"benchmark-results.json")
    args = parser.parse_args()
    reports = {}
    for name in args.models:
        source = ROOT/"openBF"/"models"/MODELS[name]
        config = yaml.safe_load(source.read_text())
        config["solver"]["cycles"] = args.cycles
        config["solver"]["convergence_tolerance"] = -1
        config["write_results"] = ["A", "Q", "u", "P"]
        config["inlet_file"] = str(source.parent/config.get("inlet_file", config["project_name"]+"_inlet.dat"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.yaml"
            path.write_text(yaml.safe_dump(config))
            try:
                result = run_simulation(path, verbose=False, write_output=False)
                reports[name] = dict(seconds=result.elapsed_seconds, steps=result.steps, cycles=result.cycles,
                                     finite=all(np.all(np.isfinite(m)) for w in result.waveforms.values() for m in w.values()),
                                     minimum_area=min(float(v.A.min()) for v in result.network.vessels),
                                     pressure_rmse_mmHg=result.convergence_error if np.isfinite(result.convergence_error) else None)
            except Exception as exc:
                reports[name] = dict(error=f"{type(exc).__name__}: {exc}")
        print(name, reports[name], flush=True)
        args.output.write_text(json.dumps(reports, indent=2))
    if any("error" in report or not report["finite"] for report in reports.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
