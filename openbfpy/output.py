"""Waveform sampling and non-destructive text output."""

from pathlib import Path

import numpy as np


def sample(network, fields):
    return {v.label: {field: (getattr(v, field)[v.sample_indices].copy()
                             - (v.Pout if field == "P" else 0)) for field in fields}
            for v in network.vessels}


def allocate_waveforms(network, fields, times):
    return {v.label: {field: np.column_stack((times, np.zeros((len(times), 5))))
                      for field in fields} for v in network.vessels}


def interpolate_row(waves, index, before, after, fraction):
    for label, fields in waves.items():
        for field, matrix in fields.items():
            matrix[index, 1:] = before[label][field]+fraction*(after[label][field]-before[label][field])


def write_waveforms(network, waves, directory, fields, append=False):
    directory = Path(directory)
    for v in network.vessels:
        if not v.tosave:
            continue
        for field in fields:
            matrix = waves[v.label][field]
            np.savetxt(directory/f"{v.label}_{field}.last", matrix, fmt="%.17g")
            if append:
                with (directory/f"{v.label}_{field}.out").open("a") as stream:
                    np.savetxt(stream, matrix, fmt="%.17g")
