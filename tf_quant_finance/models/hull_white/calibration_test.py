# Lint as: python3
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


# @test_util.run_all_in_graph_and_eager_modes
class HullWhiteCalibrationTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.mean_reversion = [0.03]
    self.volatility = [0.01]
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

    super(HullWhiteCalibrationTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_noise',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.01],
      },
      {
          'testcase_name': 'no_noise_low_vol',
          'hw_vol': [0.002],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.002],
      },
      {
          'testcase_name': 'no_noise_high_vol',
          'hw_vol': [0.05],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.05],
      },
      {
          'testcase_name': 'no_noise_bfgs',
          'hw_vol': [0.01],
          'optimizer_fn': tfp.optimizer.bfgs_minimize,
          'noise_size': 0.0,
          'use_analytic_pricing': True,
          'expected_mr': [0.03],
          'expected_vol': [0.01],
      },
      {
          'testcase_name': '5_percent_noise',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.05,
          'use_analytic_pricing': True,
          'expected_mr': [0.03080334],
          'expected_vol': [0.01036309],
      },
      {
          'testcase_name': 'mc_pricing',
          'hw_vol': [0.01],
          'optimizer_fn': None,
          'noise_size': 0.0,
          'use_analytic_pricing': False,
          'expected_mr': [0.03683715],
          'expected_vol': [0.01037683],
      })
  def test_correctness(self, hw_vol, optimizer_fn, noise_size,
                       use_analytic_pricing, expected_mr, expected_vol):
    """Tests calibration with constant parameters."""
    dtype = tf.float64

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
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
        dim=1,
        mean_reversion=self.mean_reversion,
        volatility=hw_vol,
        use_analytic_pricing=True,
        dtype=dtype)

    # Add noise
    prices = prices + tf.random.normal(
        prices.shape, stddev=noise_size * prices, seed=0, dtype=dtype)

    calibrated_model, _, _ = tff.models.hull_white.calibration_from_swaptions(
        prices=prices[:, 0],
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
        num_samples=2000,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[0, 0],
        time_step=0.25,
        maximum_iterations=50,
        dtype=dtype)
    self.assertEqual(prices.dtype, dtype)
    calib_parameters = tf.concat(
        [calibrated_model.mean_reversion.values(),
         calibrated_model.volatility.values()], axis=0)
    calib_parameters = self.evaluate(calib_parameters)
    mr = calib_parameters[:1]
    vol = calib_parameters[1:]
    self.assertAllClose(mr, expected_mr, rtol=1e-4, atol=1e-4)
    self.assertAllClose(vol, expected_vol, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
  tf.test.main()
