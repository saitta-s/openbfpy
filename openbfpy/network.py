"""Inlet data and fixed directed vascular topology."""

from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError
import warnings

import numpy as np

from .vessel import Vessel, positive


@dataclass
class Heart:
    input_data: np.ndarray

    def __post_init__(self):
        self.input_data = np.asarray(self.input_data, dtype=np.float64)
        if self.input_data.ndim != 2 or self.input_data.shape[1] != 2 or len(self.input_data) < 2:
            raise ValueError("inlet data must contain at least two rows of time and flow")
        if not np.all(np.isfinite(self.input_data)):
            raise ValueError("inlet data must be finite")
        if np.any(np.diff(self.input_data[:, 0]) < 0):
            warnings.warn("inlet rows were sorted by time", UserWarning, stacklevel=2)
            self.input_data = self.input_data[np.argsort(self.input_data[:, 0], kind="stable")]
        self.t, self.q = self.input_data.T
        if np.any(np.diff(self.t) <= 0) or self.t[0] != 0:
            raise ValueError("inlet times must be unique and start at zero")
        self.cardiac_period = positive(self.t[-1], "cardiac period")

    @classmethod
    def from_file(cls, path):
        return cls(np.loadtxt(path, ndmin=2))

    def flow(self, time):
        return float(np.interp(time % self.cardiac_period, self.t, self.q))


class Network:
    """A connected directed acyclic network of supported two/three-vessel joins."""

    def __init__(self, config, blood, heart, Ccfl=0.9):
        if not config:
            raise ValueError("network must contain vessels")
        self.blood, self.heart = blood, heart
        self.Ccfl = positive(Ccfl, "Ccfl")
        if self.Ccfl > 1:
            raise ValueError("Ccfl must be <= 1")
        self.vessels = [Vessel(c, blood) for c in config]
        self.vessels_vec = self.vessels
        self.edge_to_eid = {}
        incoming, outgoing = {}, {}
        labels = set()
        for eid, v in enumerate(self.vessels):
            # Result filenames must remain distinct on case-insensitive filesystems.
            label_key = v.label.casefold()
            if label_key in labels or (v.sn, v.tn) in self.edge_to_eid or v.sn == v.tn:
                raise ValueError("duplicate label/edge or self-loop")
            labels.add(label_key)
            self.edge_to_eid[v.sn, v.tn] = eid
            incoming.setdefault(v.tn, []).append(eid)
            outgoing.setdefault(v.sn, []).append(eid)
        nodes = sorted(set(incoming) | set(outgoing))
        for node in nodes:
            degree = (len(incoming.get(node, [])), len(outgoing.get(node, [])))
            if degree not in {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}:
                raise ValueError(f"unsupported junction at node {node}: {degree}")
        sources = [n for n in nodes if n not in incoming]
        if len(sources) != 1:
            raise ValueError("exactly one inlet is supported")
        dependencies = {n: [self.vessels[e].sn for e in incoming.get(n, [])] for n in nodes}
        try:
            self.node_order = list(TopologicalSorter(dependencies).static_order())
        except CycleError as exc:
            raise ValueError("directed cycles are unsupported") from exc
        self.incoming, self.outgoing = incoming, outgoing
        self.inlet = self.vessels[outgoing[sources[0]][0]]
        self.outlets = [self.vessels[incoming[n][0]] for n in nodes if n not in outgoing]
        for v in self.outlets:
            v.configure_outlet(blood)
        self.topo_order = [eid for node in self.node_order for eid in outgoing.get(node, [])]
