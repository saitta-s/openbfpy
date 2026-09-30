"""Simulation orchestration with fixed phase sampling and explicit termination."""

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import yaml

from .network import Heart, Network
from .output import sample, allocate_waveforms, interpolate_row, write_waveforms
from .solver import calculate_dt, solve
from .vessel import Blood, finite_float


@dataclass
class SimulationResult:
    network: Network
    waveforms: dict
    converged: bool
    cycles: int
    time: float
    convergence_error: float
    termination_reason: str
    elapsed_seconds: float
    steps: int
    output_directory: Path | None


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def convergence_error(current, previous):
    return max(float(np.sqrt(np.mean((fields["P"][:, 3]-previous[label]["P"][:, 3])**2))/133.332)
               for label, fields in current.items())


def run_simulation(yaml_config, *, verbose=True, out_files=False, save_stats=False,
                   savedir=None, write_output=True):
    """Run an existing openBF YAML model and return state and last-cycle waveforms.

    Existing nonempty output directories are rejected. Set write_output=False
    for an entirely in-memory run. Negative convergence tolerance disables early
    stopping. Exceptions propagate with their numerical context.
    """
    path = Path(yaml_config).resolve()
    config = load_config(path)
    settings = config["solver"]
    cycles, jump = settings["cycles"], settings["jump"]
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if isinstance(jump, bool) or not isinstance(jump, int) or jump < 2:
        raise ValueError("jump must be an integer >= 2")
    tolerance = finite_float(settings["convergence_tolerance"], "convergence_tolerance")
    requested = config.get("write_results", [])
    if not isinstance(requested, list) or any(f not in ("A", "Q", "u", "P") for f in requested):
        raise ValueError("write_results must be a list containing A, Q, u or P")
    fields = list(dict.fromkeys([*requested, "P"]))
    inlet_path = Path(config.get("inlet_file", f"{config['project_name']}_inlet.dat"))
    if not inlet_path.is_absolute():
        inlet_path = path.parent/inlet_path
    heart = Heart.from_file(inlet_path)
    network = Network(config["network"], Blood(**config["blood"]), heart, settings["Ccfl"])
    directory = None
    if write_output:
        directory = Path(savedir or config.get("output_directory", f"{config['project_name']}_results")).resolve()
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"output directory must be empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        # Fixed provenance filenames avoid inlet/config basename collisions.
        replay_config = config | {"inlet_file": "inlet.dat"}
        (directory/"config.yaml").write_text(yaml.safe_dump(replay_config, sort_keys=False), encoding="utf-8")
        np.savetxt(directory/"inlet.dat", heart.input_data, fmt="%.17g")
    started = time.perf_counter()
    current_time, steps = 0.0, 0
    previous = None
    error = float("inf")
    converged = False
    for cycle in range(1, cycles+1):
        start, end = (cycle-1)*heart.cardiac_period, cycle*heart.cardiac_period
        times = np.linspace(start, end, jump)
        waves = allocate_waveforms(network, fields, times)
        before = sample(network, fields)
        interpolate_row(waves, 0, before, before, 0)
        index = 1
        while current_time < end:
            dt = min(calculate_dt(network), end-current_time)
            if not np.isfinite(dt) or dt <= 0 or current_time+dt == current_time:
                raise FloatingPointError("time step cannot advance simulation")
            solve(network, dt, current_time)
            next_time = min(current_time+dt, end)
            after = sample(network, fields)
            while index < jump and times[index] <= next_time:
                fraction = (times[index]-current_time)/dt
                interpolate_row(waves, index, before, after, fraction)
                index += 1
            before = after
            current_time = next_time
            steps += 1
        if index != jump:
            raise RuntimeError("incomplete waveform sampling")
        if previous is not None:
            error = convergence_error(waves, previous)
            converged = error < tolerance
        if directory is not None:
            write_waveforms(network, waves, directory, requested, out_files)
        if verbose:
            print(f"Cycle {cycle}: pressure RMSE {error:.6g} mmHg")
        previous = waves
        if converged:
            break
    result = SimulationResult(network, waves, converged, cycle, current_time, error,
                              "converged" if converged else "cycle_limit",
                              time.perf_counter()-started, steps, directory)
    if save_stats and directory is not None:
        stats = {key: getattr(result, key) for key in ("converged", "cycles", "time", "termination_reason", "elapsed_seconds", "steps")}
        stats["convergence_error"] = error if np.isfinite(error) else None
        (directory/"stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return result
