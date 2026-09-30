"""Scientific checks independent of the end-to-end model fixtures."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from openbfpy import Blood, Heart, Network, Vessel, pressure, wave_speed
from openbfpy.boundaries import inlet, windkessel, reflection
from openbfpy.junctions import join, residual_jacobian
from openbfpy.solver import calculate_dt, limit_slopes, muscl, tapered_source, viscoelastic_step


BLOOD = Blood(0.004, 1060)


def vessel_config(**kwargs):
    return dict(label="vessel", sn=1, tn=2, L=.01, R0=.003, E=700000., h0=.0003) | kwargs


class GeometryTests(unittest.TestCase):
    def test_wall_law_and_initial_pressure(self):
        v = Vessel(vessel_config(initial_pressure=13000, Pext=2000), BLOOD)
        assert_allclose(v.P, 13000, rtol=1e-14)
        derivative = v.beta/(2*np.sqrt(v.A*v.A0))
        assert_allclose(wave_speed(v.A, v.gamma[1:-1])**2, v.A/BLOOD.rho*derivative)

    def test_mesh_taper_and_derivatives(self):
        v = Vessel(vessel_config(Rp=.003, Rd=.002), BLOOD)
        self.assertEqual(v.M, 10)
        assert_allclose(v.radius[[0, -1]], [.00295, .00205])
        assert_allclose(v.dA0dx, 2*np.pi*v.radius*(-.1))
        assert_allclose(v.dbeta_dx, -v.beta*(-.1)/v.radius)

    def test_default_thickness_derivative(self):
        cfg = vessel_config(Rp=.003, Rd=.0029)
        del cfg["h0"]
        v = Vessel(cfg, BLOOD)
        x = v.x[4]
        def beta_at(x):
            r = .003+(.0029-.003)*x/.01
            return 700000/.75*(.2802*np.exp(-505.3*r)+.1324*np.exp(-11.14*r))
        expected = (beta_at(x+1e-7)-beta_at(x-1e-7))/2e-7
        assert_allclose(v.dbeta_dx[4], expected, rtol=1e-7)

    def test_outlet_dispatch_and_resistance(self):
        v = Vessel(vessel_config(Rt=.2), BLOOD)
        v.configure_outlet(BLOOD)
        self.assertFalse(v.usewk3)
        v = Vessel(vessel_config(R1=2e9, Cc=1e-9), BLOOD)
        v.configure_outlet(BLOOD)
        self.assertTrue(v.usewk3)
        self.assertAlmostEqual(v.R1+v.R2, 2e9)
        for cfg in (vessel_config(R1=1, Cc=1e-9), vessel_config(R1=1e8, R2=1e9),
                    vessel_config(Rt=0, R1=1e8, R2=1e9, Cc=1e-9)):
            with self.assertRaises(ValueError):
                Vessel(cfg, BLOOD).configure_outlet(BLOOD)


class BoundaryTests(unittest.TestCase):
    def test_reflection_reconstructs_both_characteristics(self):
        for coefficient in (-1., 0., .5, 1.):
            v = Vessel(vessel_config(Rt=coefficient), BLOOD)
            v.u[:] = .2
            v.Q[:] = .2*v.A
            wp = v.u[-1]+4*wave_speed(v.A[-1], v.gamma[-2])
            wm = v.W1M0-coefficient*(wp-v.W2M0)
            reflection(v, 1e-5)
            c = wave_speed(v.A[-1], v.gamma[-2])
            assert_allclose([v.u[-1]-4*c, v.u[-1]+4*c], [wm, wp], atol=1e-12)
            assert_allclose(v.Q[-1], v.A[-1]*v.u[-1], rtol=1e-14)

    def test_inlet_zero_forward_reverse(self):
        for q in (0., 1e-6, -1e-6):
            v = Vessel(vessel_config(), BLOOD)
            wm = -4*wave_speed(v.A[0], v.gamma[1])
            h = Heart(np.array([[0, q], [1, q]]))
            inlet(v, 0, 1e-5, h)
            self.assertGreater(v.A[0], 0)
            self.assertEqual(v.Q[0], q)
            assert_allclose(v.u[0]-4*wave_speed(v.A[0], v.gamma[1]), wm, atol=1e-12)
            assert_allclose(v.Q[0], v.A[0]*v.u[0])

    def test_inlet_interpolation_wrap(self):
        h = Heart(np.array([[0, 1], [.5, 3], [1, 1]]))
        self.assertEqual(h.flow(.25), 2)
        self.assertEqual(h.flow(1.25), 2)

    def test_sort_unsorted_inlet_and_reject_duplicates(self):
        with self.assertWarns(UserWarning):
            h = Heart(np.array([[0, 1], [.75, 4], [.5, 3], [1, 1]]))
        self.assertEqual(h.flow(.625), 3.5)
        with self.assertRaises(ValueError):
            Heart(np.array([[0, 1], [.5, 3], [.5, 4], [1, 1]]))

    def test_windkessel_resistor_and_characteristic(self):
        v = Vessel(vessel_config(R1=1e8, R2=1e9, Cc=1e-9, initial_pressure=10000,
                                 initial_flow=1e-6, Pout=200), BLOOD)
        v.configure_outlet(BLOOD)
        v.Pc = 9000
        wp = v.u[-1]+4*wave_speed(v.A[-1], v.gamma[-2])
        windkessel(v, 1e-5, BLOOD.rho)
        p = pressure(v.A[-1], v.A0[-1], v.beta[-1], v.Pext)
        assert_allclose(p-v.Pc, v.R1*v.Q[-1], rtol=1e-12)
        assert_allclose(v.u[-1]+4*wave_speed(v.A[-1], v.gamma[-2]), wp, rtol=1e-12)
        assert_allclose(v.Q[-1], v.A[-1]*v.u[-1])

    def test_impedance_matching_preserves_total_resistance(self):
        v = Vessel(vessel_config(R1=1e8, R2=1e9, Cc=1e-9, initial_pressure=10000,
                                 inlet_impedance_matching=True), BLOOD)
        v.configure_outlet(BLOOD)
        windkessel(v, 1e-6, BLOOD.rho)
        self.assertAlmostEqual(v.R1+v.R2, 1.1e9)
        self.assertGreater(v.R2, 0)

    def test_invalid_initial_pressure_and_nonfinite_state(self):
        with self.assertRaises(ValueError):
            Vessel(vessel_config(initial_pressure=-1e9), BLOOD)
        v = Vessel(vessel_config(), BLOOD)
        v.A[0] = np.nan
        with self.assertRaises(FloatingPointError):
            v.refresh()


class JunctionTests(unittest.TestCase):
    def test_jacobians_by_finite_difference(self):
        for signs in ([1., -1.], [1., -1., -1.], [1., 1., -1.]):
            signs = np.array(signs)
            n = len(signs)
            state = np.r_[np.linspace(.1, .3, n), np.linspace(.06, .08, n)]
            args = (signs, np.linspace(40, 50, n), np.ones(n), np.linspace(60000, 80000, n),
                    np.full(n, 3e-5), BLOOD.rho, n == 2)
            _, analytic = residual_jacobian(state, *args)
            numeric = np.zeros_like(analytic)
            for col in range(2*n):
                delta = np.zeros(2*n)
                delta[col] = 1e-7
                numeric[:, col] = (residual_jacobian(state+delta, *args)[0]
                                   - residual_jacobian(state-delta, *args)[0])/(2e-7)
            assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-5)

    def test_mass_and_pressure_continuity(self):
        for counts in ((1, 1), (1, 2), (2, 1)):
            parents = [Vessel(vessel_config(initial_flow=3e-6), BLOOD) for _ in range(counts[0])]
            children = [Vessel(vessel_config(initial_flow=1e-6), BLOOD) for _ in range(counts[1])]
            join(parents, children, BLOOD.rho)
            assert_allclose(sum(v.Q[-1] for v in parents), sum(v.Q[0] for v in children), atol=1e-12)
            values = []
            for v, i in [(v, -1) for v in parents]+[(v, 0) for v in children]:
                p = pressure(v.A[i], v.A0[i], v.beta[i])
                if counts == (1, 1):
                    p += .5*BLOOD.rho*v.u[i]**2
                values.append(p)
            assert_allclose(values, values[0], atol=1e-5)


class SolverTests(unittest.TestCase):
    def test_limiter_extrema_and_linear_ramp(self):
        slopes = np.zeros(5)
        limit_slopes(slopes, np.arange(5.), 2., .25)
        assert_allclose(slopes, [0, .5, .5, .5, 0])
        limit_slopes(slopes, np.array([0., 1, 0, -1, 0]), 1, .5)
        assert_allclose(slopes, [0, 0, -.5, 0, 0])

    def test_uniform_rest_is_preserved(self):
        v = Vessel(vessel_config(initial_pressure=10000), BLOOD)
        area = v.A.copy()
        muscl(v, 1e-5, BLOOD)
        assert_allclose(v.A, area, rtol=1e-15)
        assert_allclose(v.Q, 0, atol=1e-20)

    def test_reverse_flow_cfl(self):
        n = Network([vessel_config(Rt=0)], BLOOD, Heart(np.array([[0, 0], [1, 0]])))
        v = n.inlet
        v.Q[:] = -v.A*2
        v.refresh()
        expected = .9*v.dx/(2+wave_speed(v.A[0], v.gamma[1]))
        self.assertAlmostEqual(calculate_dt(n), expected)

    def test_taper_source_matches_wall_law_derivative(self):
        v = Vessel(vessel_config(Rp=.003, Rd=.002, initial_pressure=10000), BLOOD)
        a, a0, b = v.A, v.A0, v.beta
        gamma_prime = v.dbeta_dx/(3*BLOOD.rho*np.sqrt(a0))-b*v.dA0dx/(6*BLOOD.rho*a0**1.5)
        px = v.dbeta_dx*(np.sqrt(a/a0)-1)-.5*b*np.sqrt(a/a0)*v.dA0dx/a0
        assert_allclose(tapered_source(v, BLOOD.rho), gamma_prime*a**1.5-a/BLOOD.rho*px, rtol=1e-14)

    def test_taper_rest_interior_refines(self):
        from scripts.check_taper import errors
        measured = [error for _, error in errors()]
        self.assertTrue(all(fine < .6*coarse for coarse, fine in zip(measured, measured[1:])), measured)

    def test_viscoelastic_matrix_and_state(self):
        v = Vessel(vessel_config(**{"visco-elastic": True}), BLOOD)
        v.Cv *= np.linspace(.8, 1.2, v.M)
        v.Q[:] = np.linspace(1e-6, 2e-6, v.M)
        old = v.Q.copy()
        dt = 1e-5
        a = v.Cv*dt/v.dx**2
        diag = 1+2*a
        diag[0] -= a[0]
        diag[-1] -= a[-1]
        matrix = np.diag(diag)+np.diag(-a[:-1], 1)+np.diag(-a[1:], -1)
        rhs = (1-2*a)*old
        rhs[:-1] += a[1:]*old[1:]
        rhs[1:] += a[:-1]*old[:-1]
        rhs[0] += a[0]*old[0]
        rhs[-1] += a[-1]*old[-1]
        viscoelastic_step(v, dt)
        assert_allclose(matrix@v.Q, rhs, rtol=1e-13)
        assert_allclose(v.u, v.Q/v.A)


class NetworkTests(unittest.TestCase):
    def test_case_colliding_output_labels_are_rejected(self):
        h = Heart(np.array([[0, 0], [1, 0]]))
        with self.assertRaises(ValueError):
            Network([vessel_config(label="v", sn=1, tn=2),
                     vessel_config(label="V", sn=2, tn=3)], BLOOD, h)

    def test_invalid_topologies(self):
        h = Heart(np.array([[0, 0], [1, 0]]))
        edges_cases = [[(1, 1)], [(1, 2), (1, 2)], [(1, 2), (2, 3), (3, 2)],
                       [(1, 2), (2, 3), (2, 4), (2, 5)], [(1, 2), (3, 4)]]
        for edges in edges_cases:
            configs = [vessel_config(label=f"v{i}", sn=sn, tn=tn) for i, (sn, tn) in enumerate(edges)]
            with self.assertRaises(ValueError):
                Network(configs, BLOOD, h)

    def test_merge_is_supported(self):
        edges = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
        cfg = [vessel_config(label=f"v{i}", sn=sn, tn=tn) for i, (sn, tn) in enumerate(edges)]
        n = Network(cfg, BLOOD, Heart(np.array([[0, 0], [1, 0]])))
        self.assertEqual(len(n.outlets), 1)


if __name__ == "__main__":
    unittest.main()
