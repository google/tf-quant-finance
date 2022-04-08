# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for calibration.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


# TODO(b/189459394): Split swaption based and cap based tests into two files.
class HullWhiteCalibrationSwaptionTest(parameterized.TestCase,
                                       tf.test.TestCase):

  def setUp(self):
    self.mean_reversion = 0.03
    self.volatility = 0.01
    self.volatility_low = [0.002]
    self.volatility_high = [0.05]
    self.volatility_time_dep = [0.01, 0.015]

    self.expiries = np.array(
        [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 10., 10.])
    self.float_leg_start_times = np.array([
        [0.5, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],  # 6M x 2Y
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  # 6M x 5Y
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # 1Y x 2Y
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],  # 1Y x 5Y
        [2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # 2Y x 2Y
        [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],  # 2Y x 5Y
        [3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # 3Y x 2Y
        [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],  # 3Y x 5Y
        [4.0, 4.5, 5.0, 5.5, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # 4Y x 2Y
        [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],  # 4Y x 5Y
        [5.0, 5.5, 6.0, 6.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],  # 5Y x 2Y
        [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],  # 5Y x 5Y
        [10.0, 10.5, 11.0, 11.5, 12.0, 12.0, 12.0, 12.0, 12.0,
         12.0],  # 10Y x 2Y
        [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]  # 10Y x 5Y
    ])
    self.float_leg_end_times = self.float_leg_start_times + 0.5
    max_maturities = np.array(
        [2.5, 5.5, 3.0, 6.0, 4., 7., 5., 8., 6., 9., 7., 10., 12., 15.])
    for i in range(self.float_leg_end_times.shape[0]):
      self.float_leg_end_times[i] = np.clip(
          self.float_leg_end_times[i], 0.0, max_maturities[i])

    self.fixed_leg_payment_times = self.float_leg_end_times
    self.float_leg_daycount_fractions = (
        self.float_leg_end_times - self.float_leg_start_times)
    self.fixed_leg_daycount_fractions = self.float_leg_daycount_fractions
    self.fixed_leg_coupon = 0.01 * np.ones_like(self.fixed_leg_payment_times)

    super(HullWhiteCalibrationSwaptionTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_noise',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.01],
          'vol_based_calib': False,
      }, {
          'testcase_name': 'no_noise_low_vol',
          'hw_vol': [0.002],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.002],
          'vol_based_calib': False,
      }, {
          'testcase_name': 'no_noise_high_vol',
          'hw_vol': [0.05],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.05],
          'vol_based_calib': False,
      }, {
          'testcase_name': 'no_noise_bfgs',
          'hw_vol': [0.01],
          'optimizer_fn': tfp.optimizer.bfgs_minimize,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.01],
          'vol_based_calib': False,
      }, {
          'testcase_name': '5_percent_noise',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.05,
          'use_analytic_pricing': True,
          'expected_mr': [0.03080334],
          'expected_vol': [0.01036309],
          'vol_based_calib': False,
      }, {
          'testcase_name': 'mc_pricing',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': False,
          'expected_mr': [0.0325036],
          'expected_vol': [0.01037683],
          'vol_based_calib': False,
      }, {
          'testcase_name': 'no_noise_vol_based',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.01],
          'vol_based_calib': True,
      }, {
          'testcase_name': 'no_noise_low_vol_vol_based',
          'hw_vol': [0.002],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.002],
          'vol_based_calib': True,
      }, {
          'testcase_name': 'no_noise_high_vol_vol_based',
          'hw_vol': [0.05],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.05],
          'vol_based_calib': True,
      }, {
          'testcase_name': 'no_noise_bfgs_vol_based',
          'hw_vol': [0.01],
          'optimizer_fn': tfp.optimizer.bfgs_minimize,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.01],
          'vol_based_calib': True,
      }, {
          'testcase_name': '5_percent_noise_vol_based',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.05,
          'use_analytic_pricing': True,
          'expected_mr': [0.0170023],
          'expected_vol': [0.0096614],
          'vol_based_calib': True,
      }, {
          'testcase_name': 'mc_pricing_vol_based',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': False,
          'expected_mr': [0.03342537],
          'expected_vol': [0.01025818],
          'vol_based_calib': True,
      })
  def test_correctness(self, hw_vol, optimizer_fn, noise_size,
                       use_analytic_pricing, expected_mr, expected_vol,
                       vol_based_calib):
    """Tests calibration with constant parameters."""
    dtype = tf.float64

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x)
    prices = tff.models.hull_white.swaption_price(
        expiries=self.expiries,
        floating_leg_start_times=self.float_leg_start_times,
        floating_leg_end_times=self.float_leg_end_times,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        mean_reversion=self.mean_reversion,
        volatility=hw_vol,
        use_analytic_pricing=True,
        dtype=dtype)

    # Add noise
    prices = prices + tf.random.normal(
        prices.shape, stddev=noise_size * prices, seed=0, dtype=dtype)

    calibrated_result, _, _ = tff.models.hull_white.calibration_from_swaptions(
        prices=prices,
        expiries=self.expiries,
        floating_leg_start_times=self.float_leg_start_times,
        floating_leg_end_times=self.float_leg_end_times,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        mean_reversion=[0.01],
        volatility=[0.005],
        optimizer_fn=optimizer_fn,
        use_analytic_pricing=use_analytic_pricing,
        volatility_based_calibration=vol_based_calib,
        num_samples=2000,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[0, 0],
        time_step=0.1,
        maximum_iterations=50,
        dtype=dtype)
    self.assertEqual(prices.dtype, dtype)
    calib_parameters = tf.concat(
        [calibrated_result.mean_reversion.values(),
         calibrated_result.volatility.values()], axis=0)
    calib_parameters = self.evaluate(calib_parameters)
    mr = calib_parameters[:1]
    vol = calib_parameters[1:]
    self.assertAllClose(mr, expected_mr, rtol=1e-4, atol=1e-4)
    self.assertAllClose(vol, expected_vol, rtol=1e-4, atol=1e-4)


class HullWhiteCalibrationCapFloorTest(parameterized.TestCase,
                                       tf.test.TestCase):

  def setUp(self):
    self.is_valid = np.tile(
        np.arange(0, 20), reps=(10, 1)) < np.array(
            [[4], [4], [8], [8], [12], [12], [16], [16], [20], [20]])
    self.daycount_fractions = 0.25 * self.is_valid.astype(np.float64)
    self.maturities = np.cumsum(self.daycount_fractions, axis=1) * self.is_valid
    self.expiries = (np.cumsum(self.daycount_fractions, axis=1) -
                     0.25) * self.is_valid
    self.is_cap = np.array(
        [True, False, True, False, True, False, True, False, True, False])
    self.strikes = 0.01 * np.ones_like(self.is_valid)

    super(HullWhiteCalibrationCapFloorTest, self).setUp()

  def test_docstring_example(self):
    """Explicitly test the code provided in the docstring."""

    # In this example, we synthetically generate some prices. Then we use our
    # calibration to back out these prices.
    dtype = tf.float64

    daycount_fractions = np.array([
        [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    ])
    expiries = np.array([
        [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75],
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75],
    ])
    maturities = np.array([
        [0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0],
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0],
    ])
    is_cap = np.array([True, False, True, False])
    strikes = 0.01 * np.ones_like(expiries)

    # Setup - generate some observed prices using the model.
    expected_mr = [0.4]
    expected_vol = [0.01]

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x)
    prices = tff.models.hull_white.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        reference_rate_fn=zero_rate_fn,
        notional=1.0,
        mean_reversion=expected_mr,
        volatility=expected_vol,
        is_cap=tf.expand_dims(is_cap, axis=1),
        use_analytic_pricing=True,
        dtype=dtype)

    # Calibrate the model.
    calibrated_result, is_converged, _ = (
        tff.models.hull_white.calibration_from_cap_floors(
            prices=prices,
            strikes=strikes,
            expiries=expiries,
            maturities=maturities,
            daycount_fractions=daycount_fractions,
            reference_rate_fn=zero_rate_fn,
            mean_reversion=[0.3],
            volatility=[0.02],
            notional=1.0,
            is_cap=tf.expand_dims(is_cap, axis=1),
            use_analytic_pricing=True,
            optimizer_fn=None,
            num_samples=1000,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            seed=[0, 0],
            time_step=0.1,
            maximum_iterations=200,
            dtype=dtype))

    calibrated_mr = calibrated_result.mean_reversion.values()
    calibrated_vol = calibrated_result.volatility.values()

    calibrated_mr, calibrated_vol = self.evaluate(
        [calibrated_mr, calibrated_vol])
    self.assertTrue(is_converged)
    self.assertAllClose(calibrated_mr, expected_mr, atol=1e-3, rtol=1e-2)
    self.assertAllClose(calibrated_vol, expected_vol, atol=1e-3, rtol=1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_noise',
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': 0.3,
          'expected_vol': 0.01,
      },
      {
          'testcase_name': 'no_noise_low_vol',
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': 0.3,
          'expected_vol': 0.005,
          'mr_rtol': 0.1,
          'mr_atol': 0.1,
      },
      {
          'testcase_name': 'no_noise_high_vol',
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': 0.3,
          'expected_vol': 0.05
      },
      {
          'testcase_name': 'no_noise_bfgs',
          'optimizer_fn': tfp.optimizer.bfgs_minimize,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': 0.3,
          'expected_vol': 0.01
      },
      {
          'testcase_name': '1_percent_noise',
          'optimizer_fn': None,
          'noise_size': 0.01,
          'use_analytic_pricing': True,
          'expected_mr': 0.3,
          'expected_vol': 0.01,
          # Mean-reversion parameter is very sensitive to noise, so we loosen
          # the pass-fail tolerances.
          'mr_rtol': 0.1,
          'mr_atol': 0.1,
      },
      {
          'testcase_name': '5_percent_noise',
          'optimizer_fn': None,
          'noise_size': 0.05,
          'use_analytic_pricing': True,
          'expected_mr': 0.3,
          'expected_vol': 0.01,
          # Mean-reversion parameter is very sensitive to noise, so we loosen
          # the pass-fail tolerances.
          'mr_rtol': 0.1,
          'mr_atol': 0.1,
      },
      {
          'testcase_name': 'mc_pricing',
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': False,
          'expected_mr': 0.3,
          'expected_vol': 0.01,
          # Mean-reversion parameter is very sensitive to noise, so we loosen
          # the pass-fail tolerances.
          'mr_rtol': 0.1,
          'mr_atol': 0.1,
      })
  def test_correctness(self,
                       optimizer_fn,
                       noise_size,
                       use_analytic_pricing,
                       expected_mr,
                       expected_vol,
                       mr_rtol=1e-4,
                       mr_atol=1e-3,
                       vol_rtol=1e-4,
                       vol_atol=1e-3):
    """Tests calibration with constant parameters."""
    dtype = tf.float64

    # Setup - generate some observed prices using the model.
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x)
    prices = tff.models.hull_white.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        reference_rate_fn=zero_rate_fn,
        notional=1.0,
        mean_reversion=[expected_mr],
        volatility=[expected_vol],
        is_cap=tf.expand_dims(self.is_cap, axis=1),
        use_analytic_pricing=True,
        dtype=dtype)

    prices = prices + tf.random.normal(
        prices.shape, stddev=noise_size * prices, seed=0, dtype=dtype)

    # Calibrate the model.
    calibrated_model, is_converged, _ = (
        tff.models.hull_white.calibration_from_cap_floors(
            prices=prices,
            strikes=self.strikes,
            expiries=self.expiries,
            maturities=self.maturities,
            daycount_fractions=self.daycount_fractions,
            reference_rate_fn=zero_rate_fn,
            mean_reversion=[0.4],
            volatility=[0.02],
            notional=1.0,
            is_cap=tf.expand_dims(self.is_cap, axis=1),
            use_analytic_pricing=use_analytic_pricing,
            optimizer_fn=optimizer_fn,
            num_samples=1000,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            seed=[0, 0],
            time_step=0.1,
            maximum_iterations=200,
            dtype=dtype))

    calib_parameters = tf.concat(
        axis=0,
        values=[
            calibrated_model.mean_reversion.values(),
            calibrated_model.volatility.values()
        ])
    calib_parameters = self.evaluate(calib_parameters)
    mr = calib_parameters[0]
    vol = calib_parameters[1]

    # Assert model convergence to expected parameters.
    self.assertTrue(is_converged)
    self.assertAllClose(mr, expected_mr, rtol=mr_rtol, atol=mr_atol)
    self.assertAllClose(vol, expected_vol, rtol=vol_rtol, atol=vol_atol)

  def test_correctness_time_dependent_vol(self):
    """Tests calibration with time-dependent vol parameters."""
    dtype = tf.float64

    expected_mr = 0.3
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5, 2.0], values=[0.01, 0.015, 0.02], dtype=dtype)

    # Setup - generate some observed prices using the model.
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x)
    prices = tff.models.hull_white.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        reference_rate_fn=zero_rate_fn,
        notional=1.0,
        mean_reversion=[expected_mr],
        volatility=volatility,
        is_cap=tf.expand_dims(self.is_cap, axis=1),
        use_analytic_pricing=False,
        num_samples=250,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[0, 0],
        time_step=0.1,
        dtype=dtype)

    # Calibrate the model.
    calibrated_model, is_converged, _ = (
        tff.models.hull_white.calibration_from_cap_floors(
            prices=prices,
            strikes=self.strikes,
            expiries=self.expiries,
            maturities=self.maturities,
            daycount_fractions=self.daycount_fractions,
            reference_rate_fn=zero_rate_fn,
            mean_reversion=[0.4],
            volatility=volatility,
            notional=1.0,
            is_cap=tf.expand_dims(self.is_cap, axis=1),
            use_analytic_pricing=False,
            num_samples=250,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            seed=[0, 0],
            time_step=0.1,
            maximum_iterations=200,
            dtype=dtype))

    vol = calibrated_model.volatility.values()
    vol = self.evaluate(vol)

    # Assert model convergence to expected parameters.
    self.assertTrue(is_converged)

    # Assert that the calibrated y-values of the piecewise-constant function are
    # close to the true values.
    self.assertAllClose(vol, [0.01, 0.015, 0.02], atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
  tf.test.main()
