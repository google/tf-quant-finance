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

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class HJMCalibrationTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.prices = np.array([
        0.316698, 0.626963, 1.228723, 0.477992, 0.945398, 1.849399
    ])
    self.expiries = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    self.float_leg_start_times = np.array([
        [0.5, 1.0, 1.0, 1.0],  # 6M x 6M
        [0.5, 1.0, 1.5, 1.5],  # 6M x 1Y
        [0.5, 1.0, 1.5, 2.0],  # 6M x 2Y
        [1.0, 1.5, 1.5, 1.5],  # 1Y x 6M
        [1.0, 1.5, 2.0, 2.0],  # 1Y x 1Y
        [1.0, 1.5, 2.0, 2.5],  # 1Y x 2Y
    ])
    self.float_leg_end_times = self.float_leg_start_times + 0.5
    max_maturities = np.array(
        [1.0, 1.5, 2.5, 1.5, 2.0, 3.0])
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
          'testcase_name': 'two_factor_price_xla',
          'optimizer_fn': None,
          'vol_based_calib': False,
          'num_hjm_factors': 2,
          'time_step': None,
          'num_time_steps': 3,
          'use_xla': True,
          'use_fd': False,
          'max_iter': 4,
      })
  def test_calibration(self, optimizer_fn, vol_based_calib, num_hjm_factors,
                       time_step, num_time_steps, use_xla, use_fd, max_iter):
    """Tests calibration with constant parameters is XLA-compatible."""
    dtype = tf.float64
    mr0 = [0.01, 0.05]
    if num_hjm_factors == 1:
      mr0 = [0.01]

    vol0 = [0.005, 0.007]
    if num_hjm_factors == 1:
      vol0 = [0.002]

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    times = np.unique(np.reshape(self.expiries, [-1]))
    curve_times = np.unique(np.reshape(
        self.fixed_leg_payment_times - np.expand_dims(
            self.expiries, axis=-1), [-1]))
    random_type = None
    seed = 0
    num_samples = 200

    valuation_method = (tff.models.ValuationMethod.FINITE_DIFFERENCE
                        if use_fd else tff.models.ValuationMethod.MONTE_CARLO)
    @tf.function(experimental_compile=True)
    def _fn():
      (calib_mr, calib_vol, calib_corr), _, _ = (
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
              swaption_valuation_method=valuation_method,
              num_samples=num_samples,
              random_type=random_type,
              seed=seed,
              time_step=time_step,
              num_time_steps=num_time_steps,
              times=times,
              curve_times=curve_times,
              time_step_finite_difference=time_step,
              num_grid_points_finite_difference=41,
              maximum_iterations=max_iter,
              dtype=dtype))
      return calib_mr, calib_vol, calib_corr

    # Extracting HLO text is supported only in eager mode
    hlo_text = _fn.experimental_get_compiler_ir()(stage='hlo')

    # Check that the output has an expected format
    self.assertStartsWith(hlo_text, 'HloModule')


if __name__ == '__main__':
  tf.test.main()
