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

"""Tests for zero_coupon_bond_option.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


# @test_util.run_all_in_graph_and_eager_modes
class HullWhiteBondOptionTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self.mean_reversion_1d = 0.03
    self.volatility_1d = 0.02
    self.volatility_time_dep_1d = [0.01, 0.02]
    def discount_rate_1d_fn(t):
      return 0.01 * tf.ones_like(t)
    self.discount_rate_1d_fn = discount_rate_1d_fn
    super(HullWhiteBondOptionTest, self).setUp()

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-4,
      })
  def test_correctness_1d(self, use_analytic_pricing, error_tol):
    """Tests model with constant parameters in 1 dimension."""
    dtype = tf.float64
    expiries = np.array(1.0)
    maturities = np.array(5.0)
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hull_white.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        discount_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        dtype=dtype)
    with self.subTest('DType'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [])
    price = self.evaluate(price)
    with self.subTest('Price'):
      self.assertAllClose(price, 0.02817777, rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-4,
      })
  def test_xla(self, use_analytic_pricing, error_tol):
    """Tests model with XLA."""
    dtype = tf.float64
    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    @tf.function(jit_compile=True)
    def xla_fn():
      return tff.models.hull_white.bond_option_price(
          strikes=strikes,
          expiries=expiries,
          maturities=maturities,
          mean_reversion=self.mean_reversion_1d,
          volatility=self.volatility_1d,
          discount_rate_fn=self.discount_rate_1d_fn,
          use_analytic_pricing=use_analytic_pricing,
          num_samples=500000,
          time_step=0.1,
          random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
          dtype=dtype)
    price_xla = xla_fn()
    price = self.evaluate(price_xla)
    self.assertAllClose(price, [0.02817777], rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-4,
      })
  def test_correctness_time_dep(self, use_analytic_pricing, error_tol):
    """Tests model with piecewise constant volatility."""
    dtype = tf.float64
    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    volatility = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=self.volatility_time_dep_1d,
        dtype=dtype)
    price = tff.models.hull_white.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        mean_reversion=self.mean_reversion_1d,
        volatility=volatility,
        discount_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        dtype=dtype)
    with self.subTest('Dtype'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    with self.subTest('Price'):
      self.assertAllClose(price, [0.02237839], rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-4,
      })
  def test_batch(self, use_analytic_pricing, error_tol):
    """Tests model with rank 1 batch of options."""
    dtype = tf.float64
    expiries = np.array([1.0, 1.0, 1.0])
    maturities = np.array([5.0, 5.0, 5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hull_white.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        discount_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        dtype=dtype)
    with self.subTest('Dtype'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    with self.subTest('Prices'):
      self.assertAllClose(price, [0.02817777, 0.02817777, 0.02817777],
                          rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'analytic',
          'use_analytic_pricing': True,
          'error_tol': 1e-8,
      }, {
          'testcase_name': 'simulation',
          'use_analytic_pricing': False,
          'error_tol': 1e-4,
      })
  def test_2d_batch(self, use_analytic_pricing, error_tol):
    """Tests model with rank 2 batch of options."""
    dtype = tf.float64
    expiries = np.array([[1.0, 1.0], [2.0, 2.0]])
    maturities = np.array([[5.0, 5.0], [4.0, 4.0]])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hull_white.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        discount_rate_fn=self.discount_rate_1d_fn,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        dtype=dtype)
    with self.subTest('Dtype'):
      self.assertEqual(price.dtype, dtype)
    with self.subTest('Shape'):
      self.assertAllEqual(price.shape, [2, 2])
    price = self.evaluate(price)
    expected = [[0.02817777, 0.02817777], [0.02042677, 0.02042677]]
    with self.subTest('Prices'):
      self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

if __name__ == '__main__':
  tf.test.main()
