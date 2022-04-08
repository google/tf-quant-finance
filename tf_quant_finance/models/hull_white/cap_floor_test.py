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

"""Tests for cap_floor.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HullWhiteCapFloorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.mean_reversion_1d = 0.03
    self.volatility_1d = 0.02
    self.volatility_time_dep_1d = [0.01, 0.02]
    self.mean_reversion_2d = [0.03, 0.06]
    self.volatility_2d = [0.02, 0.01]

    self.expiries = np.array([0.0, 0.25, 0.5, 0.75])
    self.maturities = np.array([0.25, 0.5, 0.75, 1.0])
    self.strikes = 0.01 * np.ones_like(self.expiries)
    self.daycount_fractions = 0.25 * np.ones_like(self.expiries)
    def discount_rate_1d_fn(t):
      return 0.01 * tf.ones_like(t)
    self.discount_rate_1d_fn = discount_rate_1d_fn

    super(HullWhiteCapFloorTest, self).setUp()

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
    # 1 year cap with quarterly resets.
    dtype = tf.float64
    price = tff.models.hull_white.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        reference_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[42, 42],
        dtype=dtype)
    with self.subTest('DType'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [])
    price = self.evaluate(price)
    with self.subTest('Prices'):
      self.assertAllClose(price, 0.4072088281493774,
                          rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
      })
  def test_gradient_1d(self, use_analytic_pricing):
    """Tests that gradient calculation in 1 dimension."""
    # 1 year cap with quarterly resets.
    dtype = tf.float64
    mean_reversion_1d = tf.convert_to_tensor(self.mean_reversion_1d,
                                             dtype=dtype)
    volatility_1d = tf.convert_to_tensor(self.volatility_1d,
                                         dtype=dtype)
    # strikes of shape [1, 4] so that prices are of shape [1]
    strikes = self.strikes[None]
    with tf.GradientTape(persistent=True) as tape:
      tape.watch([mean_reversion_1d, volatility_1d])
      price = tff.models.hull_white.cap_floor_price(
          strikes=strikes,
          expiries=self.expiries,
          maturities=self.maturities,
          daycount_fractions=self.daycount_fractions,
          notional=100.0,
          mean_reversion=mean_reversion_1d,
          volatility=volatility_1d,
          reference_rate_fn=self.discount_rate_1d_fn,
          use_analytic_pricing=use_analytic_pricing,
          num_samples=10_000,
          time_step=0.1,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[42, 42],
          dtype=dtype)
      grad_mr = tape.gradient(price, mean_reversion_1d)
      grad_vol = tape.gradient(price, volatility_1d)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    with self.subTest('GradMeanReversion'):
      self.assertAllClose(grad_mr, -0.16,
                          rtol=1e-2, atol=1e-2)
    with self.subTest('GradVolatility'):
      self.assertAllClose(grad_vol, 20.32,
                          rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-2,
      })
  def test_correctness_time_dep_1d(self, use_analytic_pricing, error_tol):
    """Tests model with piecewise constant volatility in 1 dimension."""
    dtype = tf.float64
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=self.volatility_time_dep_1d,
        dtype=dtype)
    price = tff.models.hull_white.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        mean_reversion=self.mean_reversion_1d,
        volatility=volatility,
        reference_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=100_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[42, 42],
        dtype=dtype)
    with self.subTest('DType'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [])
    price = self.evaluate(price)
    with self.subTest('Prices'):
      self.assertAllClose(price, 0.2394242699989869, rtol=error_tol,
                          atol=error_tol)

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
  def test_1d_batch(self, use_analytic_pricing, error_tol):
    """Tests model with 1d batch of options."""
    dtype = tf.float64
    expiries = np.array([self.expiries, self.expiries, self.expiries])
    maturities = np.array([self.maturities, self.maturities, self.maturities])
    strikes = np.array([self.strikes,
                        0.01 + self.strikes,
                        0.02 + self.strikes])
    daycount_fractions = np.array([
        self.daycount_fractions, self.daycount_fractions,
        self.daycount_fractions
    ])
    price = tff.models.hull_white.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        notional=100.0,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        reference_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[42, 42],
        dtype=dtype)
    with self.subTest('DType'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    with self.subTest('Prices'):
      self.assertAllClose(price, [0.40720883, 0.14283513, 0.03980642],
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
  def test_2d_batch(self, use_analytic_pricing, error_tol):
    """Tests model with 2d batch of options."""
    dtype = tf.float64
    expiries = np.array([[self.expiries, self.expiries],
                         [0.1 + self.expiries, self.expiries]])
    maturities = np.array([[self.maturities, self.maturities],
                           [self.maturities, 0.1 + self.maturities]])
    strikes = np.array([[self.strikes, 0.01 + self.strikes],
                        [self.strikes, self.strikes]])
    daycount_fractions = np.array(
        [[self.daycount_fractions, self.daycount_fractions],
         [self.daycount_fractions, self.daycount_fractions]])
    price = tff.models.hull_white.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        notional=100.0,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        reference_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=100_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)
    with self.subTest('DType'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [2, 2])
    price = self.evaluate(price)
    expected = [[0.40720883, 0.14283513],
                [0.15395877, 0.83090286]]
    with self.subTest('Prices'):
      self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

if __name__ == '__main__':
  tf.test.main()
