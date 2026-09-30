import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import yaml

from openbfpy import run_simulation


class SimulationTests(unittest.TestCase):
    def setup_model(self, root, flow=0, **solver):
        config = dict(project_name="test", write_results=["A", "Q", "u", "P"],
                      blood=dict(mu=.004, rho=1060.),
                      solver=dict(Ccfl=.8, cycles=3, jump=17, convergence_tolerance=1e-8) | solver,
                      network=[dict(label="v", sn=1, tn=2, L=.005, E=700000., R0=.003, Rt=0.)])
        path = root/"test.yaml"
        path.write_text(yaml.safe_dump(config))
        np.savetxt(root/"test_inlet.dat", [[0, flow], [.001, flow]])
        return path

    def test_rest_converges_and_output_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.setup_model(root)
            cwd = Path.cwd()
            result = run_simulation(path, savedir=root/"out", verbose=False, out_files=True, save_stats=True)
            self.assertEqual(Path.cwd(), cwd)
            self.assertTrue(result.converged)
            self.assertEqual(result.cycles, 2)
            self.assertEqual(result.termination_reason, "converged")
            for field in ("A", "Q", "u", "P"):
                assert_allclose(np.loadtxt(root/"out"/f"v_{field}.last"), result.waveforms["v"][field])
            self.assertEqual(np.loadtxt(root/"out"/"v_Q.out").shape, (34, 6))
            replay = run_simulation(root/"out"/"config.yaml", write_output=False, verbose=False)
            assert_allclose(replay.waveforms["v"]["P"], result.waveforms["v"]["P"])
            with self.assertRaises(FileExistsError):
                run_simulation(path, savedir=root/"out", verbose=False)

    def test_fixed_phase_sampling_and_cycle_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.setup_model(Path(folder), flow=1e-7, cycles=2, jump=100, convergence_tolerance=-1)
            result = run_simulation(path, write_output=False, verbose=False)
            self.assertFalse(result.converged)
            self.assertEqual(result.termination_reason, "cycle_limit")
            assert_allclose(result.waveforms["v"]["P"][:, 0], np.linspace(.001, .002, 100))
            self.assertTrue(np.all(np.isfinite(result.waveforms["v"]["P"])))

    def test_viscoelastic_simulation_keeps_state_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.setup_model(Path(folder), flow=1e-7, cycles=2, convergence_tolerance=-1)
            config = yaml.safe_load(path.read_text())
            config["network"][0]["visco-elastic"] = True
            path.write_text(yaml.safe_dump(config))
            result = run_simulation(path, write_output=False, verbose=False)
            v = result.network.vessels[0]
            assert_allclose(v.Q, v.A*v.u, rtol=1e-14)
            self.assertTrue(np.all(np.isfinite(v.P)))


if __name__ == "__main__":
    unittest.main()
