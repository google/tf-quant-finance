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

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class HJMCapFloorTest(tf.test.TestCase):

  def setUp(self):
    self.mean_reversion_1d = [0.03]
    self.volatility_1d = [0.02]
    self.volatility_time_dep_1d = [0.01, 0.02]
    self.mean_reversion_2d = [0.03, 0.06]
    self.volatility_2d = [0.02, 0.01]

    self.expiries = np.array([0.0, 0.25, 0.5, 0.75])
    self.maturities = np.array([0.25, 0.5, 0.75, 1.0])
    self.strikes = 0.01 * np.ones_like(self.expiries)
    self.daycount_fractions = 0.25 * np.ones_like(self.expiries)

    super(HJMCapFloorTest, self).setUp()

  def test_correctness_1d(self):
    """Tests model with constant parameters in 1 dimension."""
    error_tol = 1e-3

    # 1 year cap with quarterly resets.
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    price = tff.models.hjm.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        reference_rate_fn=discount_rate_fn,
        num_samples=100000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[42, 42],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [])
    price = self.evaluate(price)
    self.assertAllClose(price, 0.4072088281493774,
                        rtol=error_tol, atol=error_tol)

  def test_correctness_time_dep_1d(self):
    """Tests model with piecewise constant volatility in 1 dimension."""
    error_tol = 1e-3
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    volatility_fn = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=self.volatility_time_dep_1d, dtype=dtype)

    def piecewise_1d_volatility_fn(t, r_t):
      vol = volatility_fn([t])
      return tf.fill(dims=[r_t.shape[0], 1], value=vol)

    price = tff.models.hjm.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=piecewise_1d_volatility_fn,
        reference_rate_fn=discount_rate_fn,
        num_samples=100000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[42, 42],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [])
    price = self.evaluate(price)
    self.assertAllClose(price, 0.2394242699989869, rtol=error_tol,
                        atol=error_tol)

  def test_1d_batch(self):
    """Tests model with 1d batch of options."""
    error_tol = 1e-3
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([self.expiries, self.expiries, self.expiries])
    maturities = np.array([self.maturities, self.maturities, self.maturities])
    strikes = np.array([self.strikes, self.strikes, self.strikes])
    daycount_fractions = np.array([
        self.daycount_fractions, self.daycount_fractions,
        self.daycount_fractions
    ])
    price = tff.models.hjm.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        notional=100.0,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        reference_rate_fn=discount_rate_fn,
        num_samples=100000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[42, 42],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    self.assertAllClose(price, [0.4072088281493774, 0.4072088281493774,
                                0.4072088281493774],
                        rtol=error_tol, atol=error_tol)

  def test_2d_batch(self):
    """Tests model with 2d batch of options."""
    error_tol = 1e-3
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([[self.expiries, self.expiries],
                         [self.expiries, self.expiries]])
    maturities = np.array([[self.maturities, self.maturities],
                           [self.maturities, self.maturities]])
    strikes = np.array([[self.strikes, self.strikes],
                        [self.strikes, self.strikes]])
    daycount_fractions = np.array(
        [[self.daycount_fractions, self.daycount_fractions],
         [self.daycount_fractions, self.daycount_fractions]])
    price = tff.models.hjm.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        notional=100.0,
        dim=1,
        mean_reversion=self.mean_reversion_1d,
        volatility=self.volatility_1d,
        reference_rate_fn=discount_rate_fn,
        num_samples=100000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 2])
    price = self.evaluate(price)
    expected = [[0.4072088281493774, 0.4072088281493774],
                [0.4072088281493774, 0.4072088281493774]]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor(self):
    """Tests model with constant parameters with 2 factors."""
    error_tol = 1e-3
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    price = tff.models.hjm.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        reference_rate_fn=discount_rate_fn,
        num_samples=100000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [])
    price = self.evaluate(price)
    self.assertAllClose(price, 0.45446611, rtol=error_tol, atol=error_tol)

  def test_mixed_1d_batch_2_factor(self):
    """Tests mixed 1d batch with constant parameters with 2 factors."""
    error_tol = 1e-3
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([self.expiries, self.expiries, self.expiries])
    maturities = np.array([self.maturities, self.maturities, self.maturities])
    strikes = np.array([self.strikes, self.strikes, self.strikes])
    daycount_fractions = np.array([
        self.daycount_fractions, self.daycount_fractions,
        self.daycount_fractions
    ])
    price = tff.models.hjm.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        notional=100.0,
        dim=2,
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        reference_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[4, 2],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    expected = [0.45291683, 0.45291683, 0.45291683]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor_hull_white_consistency(self):
    """Test that under certain conditions HJM matches analytic HW results.

    For the two factor model, when both mean reversions are equivalent, then
    the HJM model matches that of a HW one-factor model with the same mean
    reversion, and effective volatility:

      eff_vol = sqrt(vol1^2 + vol2^2 + 2 rho vol1 * vol2)

    where rho is the cross correlation between the two factors. In this
    specific test, we assume rho = 0.0.
    """
    error_tol = 1e-3
    dtype = tf.float64

    mu = 0.03
    vol1 = 0.02
    vol2 = 0.01
    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    hjm_price = tff.models.hjm.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        dim=2,
        mean_reversion=[mu, mu],
        volatility=[vol1, vol2],
        reference_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    hw_price = tff.models.hull_white.cap_floor_price(
        strikes=self.strikes,
        expiries=self.expiries,
        maturities=self.maturities,
        daycount_fractions=self.daycount_fractions,
        notional=100.0,
        mean_reversion=[mu],
        volatility=[np.sqrt(vol1**2 + vol2**2)],
        reference_rate_fn=discount_rate_fn,
        use_analytic_pricing=True,
        dtype=dtype)

    hjm_price = self.evaluate(hjm_price)
    hw_price = self.evaluate(hw_price)
    self.assertAllClose(hjm_price, hw_price, rtol=error_tol, atol=error_tol)

  def test_call_put(self):
    """Tests mixed 1d batch with constant parameters with 2 factors."""
    error_tol = 1e-3
    dtype = tf.float64

    discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries = np.array([self.expiries, self.expiries, self.expiries])
    maturities = np.array([self.maturities, self.maturities, self.maturities])
    strikes = np.array(
        [self.strikes - 0.005, self.strikes - 0.005, self.strikes - 0.005])
    daycount_fractions = np.array([
        self.daycount_fractions, self.daycount_fractions,
        self.daycount_fractions
    ])
    price = tff.models.hjm.cap_floor_price(
        strikes=strikes,
        expiries=expiries,
        maturities=maturities,
        daycount_fractions=daycount_fractions,
        notional=100.0,
        dim=2,
        is_cap=[[True], [False], [False]],
        mean_reversion=self.mean_reversion_2d,
        volatility=self.volatility_2d,
        reference_rate_fn=discount_rate_fn,
        num_samples=50000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[6, 7],
        dtype=dtype)
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [3])
    price = self.evaluate(price)
    expected = [0.78964927, 0.29312759, 0.29312759]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)


if __name__ == '__main__':
  tf.test.main()
