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

"""Tests for swaption.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HullWhiteSwaptionTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.mean_reversion_1d = [0.03]
    self.volatility_1d = [0.02]
    self.volatility_time_dep_1d = [0.01, 0.02]
    self.mean_reversion_2d = [0.03, 0.03]
    self.volatility_2d = [0.02, 0.02]

    self.expiries = np.array([1.0])
    self.float_leg_start_times = np.array([1.0, 1.25, 1.5, 1.75])
    self.float_leg_end_times = np.array([1.25, 1.5, 1.75, 2.0])
    self.fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    self.float_leg_daycount_fractions = 0.25 * np.ones_like(
        self.float_leg_start_times)
    self.fixed_leg_daycount_fractions = 0.25 * np.ones_like(
        self.fixed_leg_payment_times)
    self.fixed_leg_coupon = 0.011 * np.ones_like(self.fixed_leg_payment_times)

    self.expiries_1d = np.array([1.0, 1.0])
    self.float_leg_start_times_1d = np.array([[1.0, 1.25, 1.5, 1.75],
                                              [1.0, 1.25, 1.5, 1.75]])
    self.float_leg_end_times_1d = np.array([[1.25, 1.5, 1.75, 2.0],
                                            [1.25, 1.5, 1.75, 2.0]])
    self.fixed_leg_payment_times_1d = np.array([[1.25, 1.5, 1.75, 2.0],
                                                [1.25, 1.5, 1.75, 2.0]])
    self.float_leg_daycount_fractions_1d = 0.25 * np.ones_like(
        self.float_leg_start_times_1d)
    self.fixed_leg_daycount_fractions_1d = 0.25 * np.ones_like(
        self.fixed_leg_payment_times_1d)
    self.fixed_leg_coupon_1d = 0.011 * np.ones_like(
        self.fixed_leg_payment_times_1d)

    self.expiries_2d = np.array([[1.0, 1.0], [1.0, 1.0]])
    self.float_leg_start_times_2d = np.array([[[1.0, 1.25, 1.5, 1.75],
                                               [1.0, 1.25, 1.5, 1.75]],
                                              [[1.0, 1.25, 1.5, 1.75],
                                               [1.0, 1.25, 1.5, 1.75]]])
    self.float_leg_end_times_2d = np.array([[[1.25, 1.5, 1.75, 2.0],
                                             [1.25, 1.5, 1.75, 2.0]],
                                            [[1.25, 1.5, 1.75, 2.0],
                                             [1.25, 1.5, 1.75, 2.0]]])
    self.fixed_leg_payment_times_2d = np.array([[[1.25, 1.5, 1.75, 2.0],
                                                 [1.25, 1.5, 1.75, 2.0]],
                                                [[1.25, 1.5, 1.75, 2.0],
                                                 [1.25, 1.5, 1.75, 2.0]]])
    self.float_leg_daycount_fractions_2d = 0.25 * np.ones_like(
        self.float_leg_start_times_2d)
    self.fixed_leg_daycount_fractions_2d = 0.25 * np.ones_like(
        self.fixed_leg_payment_times_2d)
    self.fixed_leg_coupon_2d = 0.011 * np.ones_like(
        self.fixed_leg_payment_times_2d)

    def zero_rate_1d_fn(t):
      return 0.01 * tf.ones_like(t)
    self.zero_rate_1d_fn = zero_rate_1d_fn

    def zero_rate_2d_fn(t):
      return 0.01 * tf.ones(t.shape.as_list() + [2], dtype=t.dtype)
    self.zero_rate_2d_fn = zero_rate_2d_fn

    super(HullWhiteSwaptionTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_correctness_1d(self, use_analytic_pricing, error_tol):
    """Tests model with constant parameters in 1 dimension."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries,
        floating_leg_start_times=self.float_leg_start_times,
        floating_leg_end_times=self.float_leg_end_times,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=self.zero_rate_1d_fn,
        notional=100.,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.7163243383624043]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 5e-3,
      })
  def test_receiver_1d(self, use_analytic_pricing, error_tol):
    """Tests model with constant parameters in 1 dimension."""
    # 1y x 1y receiver swaption with quarterly payments.
    dtype = tf.float64
    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries,
        floating_leg_start_times=self.float_leg_start_times,
        floating_leg_end_times=self.float_leg_end_times,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=self.zero_rate_1d_fn,
        notional=100.,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        is_payer_swaption=False,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.813482544626056]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_time_dep_1d(self, use_analytic_pricing, error_tol):
    """Tests model with time-dependent parameters in 1 dimension."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=self.volatility_time_dep_1d,
        dtype=dtype)
    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries,
        floating_leg_start_times=self.float_leg_start_times,
        floating_leg_end_times=self.float_leg_end_times,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=self.zero_rate_1d_fn,
        notional=100.,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=volatility,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=1000000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.5593057004094042]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_1d_batch_1d(self, use_analytic_pricing, error_tol):
    """Tests 1-d batch."""
    dtype = tf.float64

    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries_1d,
        floating_leg_start_times=self.float_leg_start_times_1d,
        floating_leg_end_times=self.float_leg_end_times_1d,
        fixed_leg_payment_times=self.fixed_leg_payment_times_1d,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions_1d,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions_1d,
        fixed_leg_coupon=self.fixed_leg_coupon_1d,
        reference_rate_fn=self.zero_rate_1d_fn,
        notional=100.,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 1])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.7163243383624043],
                                [0.7163243383624043]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_1d_batch_1d_notional(self, use_analytic_pricing, error_tol):
    """Tests 1-d batch with different notionals."""
    dtype = tf.float64

    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries_1d,
        floating_leg_start_times=self.float_leg_start_times_1d,
        floating_leg_end_times=self.float_leg_end_times_1d,
        fixed_leg_payment_times=self.fixed_leg_payment_times_1d,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions_1d,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions_1d,
        fixed_leg_coupon=self.fixed_leg_coupon_1d,
        reference_rate_fn=self.zero_rate_1d_fn,
        notional=[100., 200.],
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 1])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.7163243383624043],
                                [2 * 0.7163243383624043]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_2d_batch_1d(self, use_analytic_pricing, error_tol):
    """Tests 2-d batch."""
    dtype = tf.float64

    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries_2d,
        floating_leg_start_times=self.float_leg_start_times_2d,
        floating_leg_end_times=self.float_leg_end_times_2d,
        fixed_leg_payment_times=self.fixed_leg_payment_times_2d,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions_2d,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions_2d,
        fixed_leg_coupon=self.fixed_leg_coupon_2d,
        reference_rate_fn=self.zero_rate_1d_fn,
        notional=100.,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 2, 1])
    price = self.evaluate(price)
    expected = [
        0.7163243383624043, 0.7163243383624043, 0.7163243383624043,
        0.7163243383624043
    ]
    self.assertAllClose(
        price, tf.reshape(expected, (2, 2, 1)), rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_correctness_2d(self, use_analytic_pricing, error_tol):
    """Tests model with constant parameters in 2 dimensions."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64

    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries,
        floating_leg_start_times=self.float_leg_start_times,
        floating_leg_end_times=self.float_leg_end_times,
        fixed_leg_payment_times=self.fixed_leg_payment_times,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions,
        fixed_leg_coupon=self.fixed_leg_coupon,
        reference_rate_fn=self.zero_rate_2d_fn,
        notional=100.,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 2])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.7163243383624043, 0.7163243383624043]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 5e-3,
      })
  def test_1d_batch_2d(self, use_analytic_pricing, error_tol):
    """Tests 1-d batch."""
    dtype = tf.float64

    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries_1d,
        floating_leg_start_times=self.float_leg_start_times_1d,
        floating_leg_end_times=self.float_leg_end_times_1d,
        fixed_leg_payment_times=self.fixed_leg_payment_times_1d,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions_1d,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions_1d,
        fixed_leg_coupon=self.fixed_leg_coupon_1d,
        reference_rate_fn=self.zero_rate_2d_fn,
        notional=100.,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 2])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.7163243383624043, 0.7163243383624043],
                                [0.7163243383624043, 0.7163243383624043]],
                        rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-3,
      })
  def test_2d_batch_2d(self, use_analytic_pricing, error_tol):
    """Tests 2-d batch."""
    dtype = tf.float64

    price = tff.models.hull_white.swaption_price(
        expiries=self.expiries_2d,
        floating_leg_start_times=self.float_leg_start_times_2d,
        floating_leg_end_times=self.float_leg_end_times_2d,
        fixed_leg_payment_times=self.fixed_leg_payment_times_2d,
        floating_leg_daycount_fractions=self.float_leg_daycount_fractions_2d,
        fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions_2d,
        fixed_leg_coupon=self.fixed_leg_coupon_2d,
        reference_rate_fn=self.zero_rate_2d_fn,
        notional=100.,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        seed=0,
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 2, 2])
    price = self.evaluate(price)
    expected = [
        0.7163243383624043, 0.7163243383624043, 0.7163243383624043,
        0.7163243383624043, 0.7163243383624043, 0.7163243383624043,
        0.7163243383624043, 0.7163243383624043
    ]
    self.assertAllClose(
        price, tf.reshape(expected, (2, 2, 2)), rtol=error_tol, atol=error_tol)


if __name__ == '__main__':
  tf.test.main()
