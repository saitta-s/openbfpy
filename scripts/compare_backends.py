"""Measure optional acceleration and compare final states on identical steps."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

from .benchmark import ROOT, MODELS


def worker(backend, directory, steps):
    from openbfpy import Blood, Heart, Network
    from openbfpy._acceleration import ACCELERATED
    from openbfpy.solver import calculate_dt, solve

    if ACCELERATED != (backend == "numba"):
        raise RuntimeError(f"requested {backend}, but accelerated={ACCELERATED}")
    report = {}
    for name in ("cca", "ibif", "adan56"):
        source = ROOT/"openBF"/"models"/MODELS[name]
        config = yaml.safe_load(source.read_text())
        heart = Heart.from_file(source.parent/config.get("inlet_file", config["project_name"]+"_inlet.dat"))
        times = []
        for repetition in range(4):
            network = Network(config["network"], Blood(**config["blood"]), heart, config["solver"]["Ccfl"])
            current = 0.
            started = time.perf_counter()
            for _ in range(steps):
                dt = calculate_dt(network)
                solve(network, dt, current)
                current += dt
            if repetition:
                times.append(time.perf_counter()-started)
        np.savez(directory/f"{backend}_{name}.npz", time=current,
                 **{f"{i}_{field}": getattr(v, field) for i, v in enumerate(network.vessels) for field in ("A", "Q", "u", "P")})
        report[name] = dict(seconds=times, median_seconds=float(np.median(times)))
    (directory/f"{backend}.json").write_text(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["numpy", "numba"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT/"backend-comparison.json")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.directory, args.steps)
        return
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for backend in ("numpy", "numba"):
            env = os.environ.copy()
            env["OPENBFPY_DISABLE_JIT"] = "1" if backend == "numpy" else "0"
            subprocess.run([sys.executable, "-m", "scripts.compare_backends", "--worker", backend,
                            "--directory", str(directory), "--steps", str(args.steps)], env=env, check=True)
        baseline = json.loads((directory/"numpy.json").read_text())
        accelerated = json.loads((directory/"numba.json").read_text())
        reports = {}
        for name in baseline:
            max_difference = 0.
            with np.load(directory/f"numpy_{name}.npz") as a, np.load(directory/f"numba_{name}.npz") as b:
                for field in a.files:
                    np.testing.assert_allclose(a[field], b[field], rtol=1e-10, atol=1e-14)
                    scale = max(float(np.max(np.abs(a[field]))), 1e-30)
                    max_difference = max(max_difference, float(np.max(np.abs(a[field]-b[field])))/scale)
            reports[name] = dict(numpy=baseline[name], numba=accelerated[name],
                                 speedup=baseline[name]["median_seconds"]/accelerated[name]["median_seconds"],
                                 max_scaled_difference=max_difference, steps=args.steps)
        args.output.write_text(json.dumps(reports, indent=2))
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
