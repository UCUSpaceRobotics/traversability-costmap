"""Run with: python -m unittest discover -s tests -v."""
from __future__ import annotations

import csv
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
from src.labels.rhe_pipeline import (Command, Odom, _normalize_inputs, compare_preprocessed,
                          integrate_commands_zoh, nearest_hold_commands,
                          produce_samples, rollout, solve_window, planar_yaw,
                          detect_odom_anomalies, command_interval_fresh, estimate_rows,
                          continuous_label_segments)
from src.validation import format_timestamp, interpolate_state, ValidationLimits


def odom_at(t_s, x=0.0, y=0.0, heading=0.0, v=0.3, w=0.2):
    from math import sin, cos
    return Odom(int(t_s*1e9), int(t_s*1e9), 'odom','base_footprint',
                np.array([x,y,0.0]), np.array([cos(heading/2),0,0,sin(heading/2)]),
                np.array([v,w]), False)


class ExtractionTests(unittest.TestCase):
    def test_zero_order_hold_no_future_leak(self):
        t=np.array([0,100_000_000,200_000_000],dtype=np.int64)
        cmd_t=np.array([0,150_000_000],dtype=np.int64)
        v,w,age=nearest_hold_commands(t,cmd_t,np.array([.2,.5]),np.array([0.,.1]))
        np.testing.assert_array_equal(v,[.2,.2,.5])
        np.testing.assert_allclose(age,[0,.1,.05])

    def test_piecewise_command_integrals(self):
        times=np.array([0,50_000_000,150_000_000],dtype=np.int64)
        ds,da=integrate_commands_zoh(25_000_000,125_000_000,times,
                                     np.array([.2,.6,1.]),np.array([.1,.3,.5]))
        self.assertAlmostEqual(ds,.2*.025+.6*.075)
        self.assertAlmostEqual(da,.1*.025+.3*.075)

    def test_preprocessed_raw_action_measured(self):
        rows=[odom_at(10.0),odom_at(10.1,x=.03)]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'states_unlabeled.csv'
            with p.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['timestamp','position','quaternion','action'])
                for x in rows:
                    writer.writerow([f'{x.t_ns/1e9:.9f}', '[%.10g %.10g %.10g]'%tuple(x.p),
                                     '[%.10g %.10g %.10g %.10g]'%tuple(x.q),
                                     '[%.10g %.10g]'%tuple(x.measured)])
            result=compare_preprocessed(p,rows)
            self.assertEqual(result['matched_to_raw_odom'],2)
            self.assertEqual(result['mismatching_rows'],0)
            self.assertEqual(result['unmatched_timestamps'],0)

    def test_reject_unexpected_odom_frame(self):
        x=odom_at(0)
        x=Odom(x.t_ns,x.rx_ns,'odom','camera',x.p,x.q,x.measured,False)
        errors,_=_normalize_inputs([x,odom_at(.1),odom_at(.2)],
                                    [Command(0,.2,.1,'receive'),Command(100_000_000,.2,.1,'receive')],
                                    'odom','base_footprint')
        self.assertTrue(any('frame' in s for s in errors))

    def test_command_and_measured_velocity_separate(self):
        odom=[odom_at(1+i*.05, x=.02*i, v=.4) for i in range(41)]
        cmds=[Command(int((1+i*.02)*1e9),.2,.1,'receive') for i in range(101)]
        samples=produce_samples(odom,cmds,max_cmd_age=.15,max_odom_gap=.15)
        self.assertAlmostEqual(samples[0]['measured_v_mps'],.4)
        self.assertAlmostEqual(samples[0]['cmd_v_mps'],.2)
        self.assertAlmostEqual(samples[0]['drive_integral_next_m'],.01,places=8)
        self.assertEqual(sum(row['valid_interval'] for row in samples),40)

    def test_yaw_with_roll_and_pitch(self):
        q = Rotation.from_euler('xyz', [.3, .4, .8]).as_quat()[[3, 0, 1, 2]]
        self.assertAlmostEqual(float(planar_yaw(q)), .8)

    def test_freshness_inside_interval_and_at_exact_end_event(self):
        times = np.array([0, 190_000_000, 200_000_000], dtype=np.int64)
        self.assertFalse(command_interval_fresh(100_000_000, 200_000_000, times, .15))
        self.assertFalse(command_interval_fresh(100_000_000, 190_000_000, times, .15))
        self.assertTrue(command_interval_fresh(190_000_000, 200_000_000, times, .15))
        self.assertFalse(command_interval_fresh(-1, 10, times, .15))

    def test_interior_stale_command_invalidates_sample_interval(self):
        odom = [odom_at(t) for t in (1., 1.1, 1.2)]
        commands = [Command(t, .3, .2, 'receive') for t in (1_000_000_000, 1_190_000_000, 1_200_000_000)]
        rows = produce_samples(odom, commands, .15, .2)
        self.assertTrue(rows[0]['valid_interval'])
        self.assertTrue(rows[1]['valid_command'])  # fresh at both odometry endpoints
        self.assertTrue(rows[2]['valid_command'])
        self.assertFalse(rows[1]['valid_interval'])
        self.assertIn('stale_command', rows[1]['interval_reasons'])

    def test_preprocessed_image_time_states(self):
        raw = [odom_at(10., heading=.2), odom_at(10.1, x=.03, heading=.25)]
        states = [(s.t_ns, s.p, s.q, s.measured) for s in raw]
        sample, _ = interpolate_state(states, 10_025_000_000, ValidationLimits())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'states_unlabeled.csv'
            with path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'position', 'quaternion', 'action'])
                writer.writerow([format_timestamp(sample[0]), *['['+' '.join(map(str, x))+']' for x in sample[1:]]])
            result = compare_preprocessed(path, raw)
            self.assertEqual(result['mismatching_rows'], 0)
            self.assertEqual(result['unmatched_timestamps'], 0)


class AnomalyTests(unittest.TestCase):
    def make_odom(self):
        return [odom_at(1+i*.05, x=i*.025, v=.5, w=0.) for i in range(161)]

    def test_observed_velocity_spike_and_position_jump(self):
        odom = self.make_odom()
        odom[20].measured[0] = 1.425
        odom[50].p[0] = odom[49].p[0] + .11
        original = odom[50].p.copy()
        valid, intervals, reasons, interval_reasons = detect_odom_anomalies(odom)
        self.assertFalse(valid[20])
        self.assertIn('velocity_spike', reasons[20])
        self.assertFalse(intervals[19]); self.assertFalse(intervals[20])
        self.assertFalse(intervals[49])
        self.assertIn('position_discontinuity', interval_reasons[49])
        np.testing.assert_array_equal(odom[50].p, original)

    def test_heading_timestamp_and_gap(self):
        for mode in ('heading', 'duplicate', 'gap', 'missing_stamp', 'nan'):
            odom = self.make_odom()
            s = odom[30]
            if mode == 'heading':
                odom[30] = odom_at(2.5, x=.75, heading=.4, v=.5, w=0.)
            elif mode == 'duplicate':
                odom[30] = odom_at(2.45, x=.75, v=.5, w=0.)
            elif mode == 'gap':
                odom[30] = odom_at(3., x=.75, v=.5, w=0.)
            elif mode == 'missing_stamp':
                odom[30] = Odom(s.t_ns, s.rx_ns, s.frame, s.child, s.p, s.q, s.measured, True)
            else:
                s.p[0] = np.nan
            valid, intervals, _, _ = detect_odom_anomalies(odom)
            self.assertFalse(intervals[29], mode)
            if mode != 'heading':
                self.assertFalse(valid[30] if mode != 'gap' else intervals[29], mode)

    def test_command_disagreement_is_not_an_anomaly(self):
        odom = self.make_odom()
        commands = [Command(s.t_ns, .2, 0., 'receive') for s in odom]
        rows = produce_samples(odom, commands, .15, .2)
        self.assertTrue(all(r['valid_measurement'] for r in rows))
        self.assertTrue(all(r['valid_interval'] for r in rows[:-1]))

    def test_all_windows_overlapping_anomaly_are_rejected(self):
        odom = self.make_odom()
        commands = [Command(s.t_ns, .6, .2, 'receive') for s in odom]
        odom[80].measured[0] = 1.425
        rows = produce_samples(odom, commands, .15, .2)
        estimates = estimate_rows(rows, horizon_seconds=1.)
        self.assertTrue(any(r['valid'] for r in estimates[20:80]))
        self.assertTrue(any(r['valid'] for r in estimates[102:]))
        for row in estimates[80:101]:
            self.assertFalse(row['valid'], row)
        self.assertIn('velocity_spike', estimates[80]['rejection_reasons'])
        # A bad interval without a bad endpoint must also reset the whole horizon.
        clean = self.make_odom()
        rows = produce_samples(clean, commands, .15, .2)
        rows[80]['valid_interval'] = 0
        estimates = estimate_rows(rows, horizon_seconds=1.)
        for row in estimates[81:101]:
            self.assertFalse(row['valid'], row)


class RHETests(unittest.TestCase):
    def _make(self, mu=.7,nu=.45,steps=80,dt=.05):
        # Alternating command profile to make both mu and nu identifiable.
        v=np.full(steps,.45)
        w=np.concatenate((np.full(steps//2,.4),np.full(steps-steps//2,-.4)))
        ds=v*dt;dphi=w*dt
        poses=rollout(np.array([0.,0.,.2]),mu,nu,ds,dphi)
        rows=[]
        for i in range(steps+1):
            rows.append(dict(timestamp_ns=int(i*dt*1e9),
                             x_m=poses[i,0],y_m=poses[i,1],yaw_rad=poses[i,2],
                             drive_integral_next_m=float(ds[i]) if i<steps else float('nan'),
                             turn_integral_next_rad=float(dphi[i]) if i<steps else float('nan'),
                             valid_interval=1 if i<steps else 0))
        return rows

    def test_identifies_known_parameters(self):
        rows=self._make()
        result=solve_window(rows,None,.10,.15,.5,.25,.3)
        self.assertTrue(result['mu_valid'])
        self.assertTrue(result['nu_valid'])
        self.assertAlmostEqual(result['mu'],.7,places=4)
        self.assertAlmostEqual(result['nu'],.45,places=4)
        self.assertLess(result['rms_xy_m'],1e-5)

    def test_no_angular_excitation_disables_nu(self):
        rows=self._make()
        for r in rows:r['turn_integral_next_rad']=0.
        # Observe consistent straight-line motion by regenerating test data.
        states=rollout(np.array([0.,0.,.2]),.7,.9,
                       np.array([r['drive_integral_next_m'] for r in rows[:-1]]),
                       np.zeros(len(rows)-1))
        for r,p in zip(rows,states):r['x_m'],r['y_m'],r['yaw_rad']=p
        result=solve_window(rows,None,.10,.15,.5,.25,.3)
        self.assertTrue(result['mu_valid'])
        self.assertFalse(result['nu_valid'])
        self.assertTrue(np.isnan(result['nu']))

    def test_quadratic_objective_and_prior_at_window_start(self):
        rows = self._make()
        result = solve_window(rows, (.7, .45), .1, .15, .5, .25, .3,
                              np.array([rows[0]['x_m'], rows[0]['y_m'], rows[0]['yaw_rad']]))
        self.assertAlmostEqual(result['mu'], .7, places=4)
        self.assertAlmostEqual(result['nu'], .45, places=4)

    def test_estimate_rows_recovers_parameters_with_sliding_priors(self):
        rows = self._make(steps=100)
        results = estimate_rows(rows, horizon_seconds=1.)
        valid = [r for r in results if r['valid']]
        self.assertGreater(len(valid), 70)
        for result in valid:
            self.assertAlmostEqual(result['mu'], .7, places=4)
            self.assertAlmostEqual(result['nu'], .45, places=4)

    def test_low_excitation_identifiable_fits_are_accepted(self):
        rows = self._make()
        ds = np.full(len(rows)-1, .01*.05)
        dphi = np.full(len(rows)-1, .005*.05)
        poses = rollout(np.array([0., 0., .2]), .7, .45, ds, dphi)
        for i, row in enumerate(rows):
            row.update(x_m=poses[i, 0], y_m=poses[i, 1], yaw_rad=poses[i, 2])
            if i < len(ds):
                row.update(drive_integral_next_m=ds[i], turn_integral_next_rad=dphi[i])
        results = estimate_rows(rows, horizon_seconds=3.)
        for result in results:
            if result['window_samples'] == 0:
                continue
            self.assertLess(result['linear_excitation_m'], .2)
            self.assertLess(result['angular_excitation_rad'], .15)
            self.assertTrue(result['valid'], result)
            self.assertEqual(result['measurement_jacobian_rank'], 5)
            self.assertAlmostEqual(result['mu'], .7, places=3)
            self.assertAlmostEqual(result['nu'], .45, places=3)
        self.assertTrue(any(r['valid'] for r in results))

    def test_prior_cannot_make_zero_input_coefficient_identifiable(self):
        for zero_linear, zero_angular in ((False, True), (True, False), (True, True)):
            rows = self._make()
            ds = np.full(len(rows)-1, 0. if zero_linear else .45*.05)
            dphi = np.full(len(rows)-1, 0. if zero_angular else .4*.05)
            poses = rollout(np.array([0., 0., .2]), .7, .45, ds, dphi)
            for i, row in enumerate(rows):
                row.update(x_m=poses[i, 0], y_m=poses[i, 1], yaw_rad=poses[i, 2])
                if i < len(ds):
                    row.update(drive_integral_next_m=ds[i], turn_integral_next_rad=dphi[i])
            result = solve_window(rows, (.7, .45), .1, .15, .5, .25, .3, poses[0])
            self.assertFalse(result['valid'])
            self.assertEqual(result['mu_identifiable'], int(not zero_linear))
            self.assertEqual(result['nu_identifiable'], int(not zero_angular))
            if zero_linear:
                self.assertTrue(np.isnan(result['mu']))
                self.assertIn('unidentifiable_mu', result['rejection_reasons'])
            if zero_angular:
                self.assertTrue(np.isnan(result['nu']))
                self.assertIn('unidentifiable_nu', result['rejection_reasons'])


class ClipTests(unittest.TestCase):
    def test_segmentation_never_bridges_invalid_estimate(self):
        rows = [dict(timestamp_ns=i*1_000_000_000, valid=int(i != 40)) for i in range(81)]
        clips = continuous_label_segments(rows, 30.)
        self.assertEqual([(c[0]['timestamp_ns']//10**9, c[-1]['timestamp_ns']//10**9)
                          for c in clips], [(0, 39), (41, 80)])
        self.assertTrue(all(all(r['valid'] for r in c) for c in clips))

    def test_short_intervals_do_not_have_future_coverage(self):
        rows = [dict(timestamp_ns=i*1_000_000_000, valid=1) for i in range(30)]
        self.assertEqual(continuous_label_segments(rows, 30.), [])
        rows.append(dict(timestamp_ns=30_000_000_000, valid=1))
        self.assertEqual(len(continuous_label_segments(rows, 30.)), 1)

if __name__=='__main__': unittest.main()
