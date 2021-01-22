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
"""Tests for swaptions using HJM model."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class HJMSwaptionTest(tf.test.TestCase):

  def test_correctness_1d(self):
    """Tests model with constant parameters in 1 dimension."""
    dtype = tf.float64
    error_tol = 1e-3

    # 1y x 1y swaption with quarterly payments.
    expiries = np.array([1.0])
    fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    mean_reversion = [0.03]
    volatility = [0.02]

    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=1,
        mean_reversion=mean_reversion,
        volatility=volatility,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2])

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [[0.7163243383624043]], rtol=error_tol, atol=error_tol)

  def test_receiver_1d(self):
    """Test model with constant parameters in 1 dimension."""
    dtype = tf.float64
    error_tol = 1e-2

    # 1y x 1y swaption with quarterly payments.
    expiries = np.array([1.0])
    fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    mean_reversion = [0.03]
    volatility = [0.02]

    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=1,
        mean_reversion=mean_reversion,
        volatility=volatility,
        is_payer_swaption=False,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [[0.813482544626056]], rtol=error_tol, atol=error_tol)

  def test_time_dep_1d(self):
    """Tests model with time-dependent parameters in 1 dimension."""
    dtype = tf.float64
    error_tol = 1e-3

    # 1y x 1y swaption with quarterly payments.
    expiries = np.array([1.0])
    fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    mean_reversion = [0.03]

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    vol_piecewise_constant_fn = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[0.01, 0.02], dtype=dtype)

    def piecewise_1d_volatility_fn(t, r_t):
      vol = vol_piecewise_constant_fn([t])
      return tf.fill(dims=[r_t.shape[0], 1], value=vol)

    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=1,
        mean_reversion=mean_reversion,
        volatility=piecewise_1d_volatility_fn,
        num_samples=1000000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [[0.5593057004094042]], rtol=error_tol, atol=error_tol)

  def test_1d_batch_1d(self):
    """Tests 1-d batch."""
    dtype = tf.float64
    error_tol = 1e-3

    # 1y x 1y swaption with quarterly payments.
    expiries = np.array([1.0, 1.0])
    fixed_leg_payment_times = np.array([[1.25, 1.5, 1.75, 2.0],
                                        [1.25, 1.5, 1.75, 2.0]])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    mean_reversion = [0.03]
    volatility = [0.02]

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=1,
        mean_reversion=mean_reversion,
        volatility=volatility,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [[0.7163243383624043], [0.7163243383624043]],
        rtol=error_tol,
        atol=error_tol)

  def test_1d_batch_1d_notional(self):
    """Tests 1-d batch with different notionals."""
    dtype = tf.float64
    error_tol = 1e-3

    # 1y x 1y swaption with quarterly payments.
    expiries = np.array([1.0, 1.0])
    fixed_leg_payment_times = np.array([[1.25, 1.5, 1.75, 2.0],
                                        [1.25, 1.5, 1.75, 2.0]])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    mean_reversion = [0.03]
    volatility = [0.02]

    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=[100., 200.],
        num_hjm_factors=1,
        mean_reversion=mean_reversion,
        volatility=volatility,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [[0.7163243383624043], [2 * 0.7163243383624043]],
        rtol=error_tol,
        atol=error_tol)

  def test_2d_batch_1d(self):
    """Tests 2-d batch."""
    dtype = tf.float64
    error_tol = 1e-3

    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
    expiries_2d = np.array([[1.0, 1.0], [1.0, 1.0]])
    fixed_leg_payment_times_2d = np.array([[[1.25, 1.5, 1.75, 2.0],
                                            [1.25, 1.5, 1.75, 2.0]],
                                           [[1.25, 1.5, 1.75, 2.0],
                                            [1.25, 1.5, 1.75, 2.0]]])
    fixed_leg_daycount_fractions_2d = 0.25 * np.ones_like(
        fixed_leg_payment_times_2d)
    fixed_leg_coupon_2d = 0.011 * np.ones_like(fixed_leg_payment_times_2d)
    mean_reversion = [0.03]
    volatility = [0.02]

    price = tff.models.hjm.swaption_price(
        expiries=expiries_2d,
        fixed_leg_payment_times=fixed_leg_payment_times_2d,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions_2d,
        fixed_leg_coupon=fixed_leg_coupon_2d,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=1,
        mean_reversion=mean_reversion,
        volatility=volatility,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
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

  def test_correctness_2_factor(self):
    """Tests model with constant parameters in 2 dimensions."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    error_tol = 1e-3

    expiries = np.array([1.0])
    fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    mean_reversion = [0.03, 0.06]
    volatility = [0.02, 0.01]
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=2,
        mean_reversion=mean_reversion,
        volatility=volatility,
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1, 1])
    price = self.evaluate(price)
    self.assertAllClose(price, [[0.802226]], rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor_hull_white_consistency(self):
    """Test that under certain conditions HJM matches analytic HW results.

    For the two factor model, when both mean reversions are equivalent, then
    the HJM model matches that of a HW one-factor model with the same mean
    reversion, and effective volatility:

      eff_vol = sqrt(vol1^2 + vol2^2 + 2 rho vol1 * vol2)

    where rho is the cross correlation between the two factors. In this
    specific test, we assume rho = 0.0.
    """
    dtype = tf.float64
    error_tol = 1e-3

    expiries = np.array([1.0])
    fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    mu = 0.03
    vol1 = 0.02
    vol2 = 0.01
    eff_vol = np.sqrt(vol1**2 + vol2**2)

    hjm_price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=2,
        mean_reversion=[mu, mu],
        volatility=[vol1, vol2],
        num_samples=500000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)
    hjm_price = self.evaluate(hjm_price)

    hw_price = tff.models.hull_white.swaption_price(
        expiries=expiries,
        floating_leg_start_times=[0],  # Unused
        floating_leg_end_times=[0],  # Unused
        floating_leg_daycount_fractions=[0],  # Unused
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        dim=1,
        mean_reversion=[mu],
        volatility=[eff_vol],
        use_analytic_pricing=True,
        dtype=dtype)
    hw_price = self.evaluate(hw_price)

    self.assertNear(hjm_price, hw_price, error_tol)


if __name__ == '__main__':
  tf.test.main()
