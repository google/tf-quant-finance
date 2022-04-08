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

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


# @test_util.run_all_in_graph_and_eager_modes
class HJMBondOptionTest(tf.test.TestCase):

  def setUp(self):
    self.mean_reversion_1d = [0.03]
    self.volatility_1d = [0.02]
    self.volatility_time_dep_1d = [0.01, 0.02]
    self.mean_reversion_2d = [0.03, 0.06]
    self.volatility_2d = [0.02, 0.01]

    super(HJMBondOptionTest, self).setUp()

  def test_correctness_1d(self):
    """Tests model with constant parameters in 1 dimension."""
    dtype = tf.float64
    error_tol = 1e-2

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        discount_rate_fn=discount_rate_fn,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(price, [0.02817777], rtol=error_tol, atol=error_tol)

  def test_correctness_time_dep_1d(self):
    """Tests model with piecewise constant volatility in 1 dimension."""
    dtype = tf.float64
    error_tol = 1e-2

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)

    vol_piecewise_constant_fn = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=self.volatility_time_dep_1d, dtype=dtype)

    def piecewise_1d_volatility_fn(t, r_t):
      vol = vol_piecewise_constant_fn([t])
      return tf.fill(dims=[r_t.shape[0], 1], value=vol)

    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=piecewise_1d_volatility_fn,
        discount_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.05,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(price, [0.02237839], rtol=error_tol, atol=error_tol)

  def test_1d_batch(self):
    """Tests model with 1d batch of options."""
    dtype = tf.float64
    error_tol = 1e-2

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0, 1.0, 1.0])
    maturities = np.array([5.0, 5.0, 5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        discount_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [0.02817777, 0.02817777, 0.02817777],
        rtol=error_tol,
        atol=error_tol)

  def test_2d_batch(self):
    """Tests model with 2d batch of options."""
    dtype = tf.float64
    error_tol = 1e-2

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([[1.0, 1.0], [2.0, 2.0]])
    maturities = np.array([[5.0, 5.0], [4.0, 4.0]])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        discount_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 2])
    price = self.evaluate(price)
    expected = [[0.02817777, 0.02817777], [0.02042677, 0.02042677]]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor(self):
    """Tests model with constant parameters with 2 factors."""
    dtype = tf.float64
    error_tol = 1e-3

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        discount_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(price, [0.03111126], rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor_with_correlation(self):
    """Tests model with constant parameters with 2 correlated factors."""
    dtype = tf.float64
    error_tol = 1e-3

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        discount_rate_fn=discount_rate_fn,
        corr_matrix=[[1.0, 0.5], [0.5, 1.0]],
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(price, [0.036809], rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor_hull_white_consistency(self):
    """Test that under certain conditions HJM matches analytic HW results.

    For the two factor model, when both mean reversions are equivalent, then
    the HJM model matches that of a HW one-factor model with the same mean
    reversion, and effective volatility:

      eff_vol = sqrt(vol1^2 + vol2^2 + 2 rho vol1 * vol2)

    where rho is the cross correlation between the two factors.
    """
    dtype = tf.float64
    error_tol = 1e-3

    mu = 0.03
    rho = 0.5
    vol1 = 0.02
    vol2 = 0.01
    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    expiries = np.array([1.0])
    maturities = np.array([5.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)

    hjm_price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=2,
        mean_reversion=[mu, mu],
        volatility=[vol1, vol2],
        discount_rate_fn=discount_rate_fn,
        corr_matrix=[[1.0, rho], [rho, 1.0]],
        num_samples=100000,
        time_step=0.05,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    hjm_price = self.evaluate(hjm_price)

    hw_price = tff.models.hull_white.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        mean_reversion=mu,
        volatility=np.sqrt(vol1**2 + vol2**2 + 2.0 * rho * vol1 * vol2),
        discount_rate_fn=discount_rate_fn,
        use_analytic_pricing=True,
        dtype=dtype)
    hw_price = self.evaluate(hw_price)

    self.assertAllClose(hjm_price, hw_price, rtol=error_tol, atol=error_tol)

  def test_mixed_1d_batch_2_factor(self):
    """Tests mixed 1d batch with constant parameters with 2 factors."""
    dtype = tf.float64
    error_tol = 1e-2

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0, 1.0, 2.0])
    maturities = np.array([5.0, 6.0, 4.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        discount_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    expected = [0.03115176, 0.03789011, 0.02266191]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

  def test_call_put(self):
    """Tests call and put pricing."""
    dtype = tf.float64
    error_tol = 1e-2

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([1.0, 1.0, 2.0])
    maturities = np.array([5.0, 6.0, 4.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries) - 0.01
    price = tff.models.hjm.bond_option_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        is_call_options=[True, False, False],
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        discount_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        dtype=dtype,
        seed=[1, 2])
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    expected = [0.03620415, 0.03279728, 0.01784987]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)


if __name__ == '__main__':
  tf.test.main()
