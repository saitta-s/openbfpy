"""Direct comparisons against exported upstream Julia kernels, when available."""

from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose

from openbfpy import Blood, Vessel
from openbfpy.solver import muscl
from openbfpy.junctions import residual_jacobian


FIXTURES = Path(__file__).parent/"fixtures"/"julia"


@unittest.skipUnless((FIXTURES/"elastic_output.csv").exists(), "Julia kernel fixtures not exported")
class JuliaKernelTests(unittest.TestCase):
    def test_actual_julia_kernel(self):
        for tag in ("elastic", "viscoelastic"):
            with self.subTest(tag=tag):
                blood = Blood(.004, 1060.)
                v = Vessel(dict(label="reference", sn=1, tn=2, L=.01, M=10, E=700000., R0=.003,
                                h0=.0003, gamma_profile=2, **{"visco-elastic": tag == "viscoelastic"}), blood)
                expected_geometry = np.loadtxt(FIXTURES/f"{tag}_geometry.csv", delimiter=",")
                assert_allclose(np.column_stack((v.A0, v.beta, v.gamma[1:-1], v.Cv)), expected_geometry, rtol=1e-14)
                v.A[:], v.Q[:] = np.loadtxt(FIXTURES/f"{tag}_input.csv", delimiter=",").T
                v.refresh()
                v.update_ghosts()
                muscl(v, 1e-5, blood)
                expected = np.loadtxt(FIXTURES/f"{tag}_output.csv", delimiter=",")
                assert_allclose(v.A, expected[:, 0], rtol=1e-12, atol=1e-18)
                assert_allclose(v.Q, expected[:, 1], rtol=1e-12, atol=1e-18)
                # Julia leaves u stale after viscoelasticity; the approved correction does not.
                if tag == "elastic":
                    assert_allclose(v.u, expected[:, 2], rtol=1e-12, atol=1e-15)
                assert_allclose(v.u, v.Q/v.A, rtol=1e-14)

    @unittest.skipUnless((FIXTURES/"conjunction_residual.csv").exists(), "Julia junction fixtures not exported")
    def test_julia_junction_equations(self):
        for tag, signs in (("conjunction", [1., -1.]), ("bifurcation", [1., -1., -1.]),
                           ("anastomosis", [1., 1., -1.])):
            n = len(signs)
            v = Vessel(dict(label="reference", sn=1, tn=2, L=.01, E=700000., R0=.003, h0=.0003), Blood(.004, 1060.))
            state = np.r_[np.linspace(.1, .3, n), np.linspace(.06, .08, n)]
            f, j = residual_jacobian(state, np.array(signs), np.linspace(40, 50, n), np.ones(n),
                                     np.full(n, v.beta[0]), np.full(n, v.A0[0]), 1060., n == 2)
            expected_f = np.loadtxt(FIXTURES/f"{tag}_residual.csv", delimiter=",")
            expected_j = np.loadtxt(FIXTURES/f"{tag}_jacobian.csv", delimiter=",")
            assert_allclose(f[:n], expected_f[:n], rtol=1e-12, atol=1e-13)
            assert_allclose(f[n], expected_f[n], rtol=1e-12, atol=1e-18)
            assert_allclose(f[n+1:], expected_f[n+1:], rtol=1e-12, atol=1e-10)
            assert_allclose(j[:n], expected_j[:n], rtol=1e-12, atol=1e-13)
            assert_allclose(j[n], expected_j[n], rtol=1e-12, atol=1e-18)
            assert_allclose(j[n+1:], expected_j[n+1:], rtol=1e-12, atol=1e-10)
