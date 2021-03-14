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


@test_util.run_all_in_graph_and_eager_modes
class HJMCalibrationTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.prices = np.array([
        0.42919881, 0.98046542, 0.59045074, 1.34909391, 0.79491583, 1.81768802,
        0.93210461, 2.13625342, 1.05114573, 2.40921088, 1.12941064, 2.58857507,
        1.37029637, 3.15081683
    ])
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

    super(HJMCalibrationTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'two_factor_price',
          'optimizer_fn': None,
          'vol_based_calib': False,
          'num_hjm_factors': 2,
      }, {
          'testcase_name': 'two_factor_vol',
          'optimizer_fn': None,
          'vol_based_calib': True,
          'num_hjm_factors': 2,
      }, {
          'testcase_name': 'two_factor_bfgs',
          'optimizer_fn': tfp.optimizer.bfgs_minimize,
          'vol_based_calib': False,
          'num_hjm_factors': 2,
      }, {
          'testcase_name': 'three_factor_price',
          'optimizer_fn': None,
          'vol_based_calib': False,
          'num_hjm_factors': 3,
      }, {
          'testcase_name': 'three_factor_vol',
          'optimizer_fn': None,
          'vol_based_calib': True,
          'num_hjm_factors': 3,
      })
  def test_calibration(self, optimizer_fn, vol_based_calib, num_hjm_factors):
    """Tests calibration with constant parameters."""
    dtype = tf.float64
    mr0 = [0.01, 0.05]
    if num_hjm_factors == 3:
      mr0 = [0.01, 0.05, 0.1]

    vol0 = [0.005, 0.007]
    if num_hjm_factors == 3:
      vol0 = [0.002, 0.003, 0.008]

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    _, calib_mr, calib_vol, calib_corr, _, _ = (
        tff.models.hjm.calibration_from_swaptions(
            prices=self.prices,
            expiries=self.expiries,
            floating_leg_start_times=self.float_leg_start_times,
            floating_leg_end_times=self.float_leg_end_times,
            fixed_leg_payment_times=self.fixed_leg_payment_times,
            floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
            fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
            fixed_leg_coupon=self.fixed_leg_coupon,
            reference_rate_fn=zero_rate_fn,
            notional=100.,
            num_hjm_factors=num_hjm_factors,
            mean_reversion=mr0,
            volatility=vol0,
            optimizer_fn=optimizer_fn,
            volatility_based_calibration=vol_based_calib,
            num_samples=2000,
            random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
            seed=[0, 0],
            time_step=0.25,
            maximum_iterations=20,
            dtype=dtype))

    prices = tff.models.hjm.swaption_price(
        expiries=self.expiries,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        num_hjm_factors=num_hjm_factors,
        notional=100.,
        mean_reversion=calib_mr,
        volatility=calib_vol,
        corr_matrix=calib_corr,
        num_samples=2000,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[0, 0],
        time_step=0.25,
        dtype=dtype)

    self.assertEqual(prices.dtype, dtype)
    self.assertAllClose(prices[:, 0], self.prices, rtol=0.1, atol=0.1)


if __name__ == '__main__':
  tf.test.main()
