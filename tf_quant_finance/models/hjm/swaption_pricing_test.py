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

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class HJMSwaptionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'time_step',
          'time_step': 0.1,
          'num_time_steps': None,
          'use_xla': False,
      }, {
          'testcase_name': 'num_time_steps',
          'time_step': None,
          'num_time_steps': 11,
          'use_xla': False,
      }, {
          'testcase_name': 'num_time_steps_xla',
          'time_step': None,
          'num_time_steps': 11,
          'use_xla': True,
      })
  def test_correctness_1d(self, time_step, num_time_steps, use_xla):
    """Tests model with constant parameters in 1 dimension."""
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

    if use_xla:
      curve_times = np.array(fixed_leg_payment_times - expiries)
    else:
      curve_times = None

    def _fn():
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
          num_samples=50_000,
          time_step=time_step,
          num_time_steps=num_time_steps,
          curve_times=curve_times,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2],
          dtype=dtype)
      return price

    if use_xla:
      price = self.evaluate(tf.function(_fn, jit_compile=True)())
    else:
      price = self.evaluate(_fn())
    self.assertAllClose(
        price, [0.7163243383624043], rtol=error_tol, atol=error_tol)

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
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [0.813482544626056], rtol=error_tol, atol=error_tol)

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
      del r_t
      vol = vol_piecewise_constant_fn([t])
      return vol

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
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [0.5593057004094042], rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ModelHasNoBatch',
          'model_batch_rank': 0,
          'leading_dim_match': None,
      }, {
          'testcase_name': 'ModelHas1D_Batch_LeadDimBroadcastable',
          'model_batch_rank': 1,
          'leading_dim_match': False,
      }, {
          'testcase_name': 'ModelHas1D_Batch_LeadDimMatch',
          'model_batch_rank': 1,
          'leading_dim_match': True,
      })
  def test_1d_batch_1d(self, model_batch_rank, leading_dim_match):
    """Tests 1-d batch of swaptions with 1-factor models."""
    dtype = tf.float64
    error_tol = 1e-2

    # 1y x 1y swaption with quarterly payments.
    expiries = np.array([1.0, 1.0])
    fixed_leg_payment_times = np.array([[1.25, 1.5, 1.75, 2.0],
                                        [1.25, 1.5, 1.75, 2.0]])
    mean_reversion = [0.03]
    volatility = [0.02]
    if model_batch_rank == 1:
      expiries = np.expand_dims(expiries, axis=0)
      fixed_leg_payment_times = np.expand_dims(fixed_leg_payment_times, axis=0)
      if leading_dim_match:
        expiries = np.repeat(expiries, 4, axis=0)
        fixed_leg_payment_times = np.repeat(fixed_leg_payment_times, 4, axis=0)
      def zero_rate_fn(t):  # pylint-disable=function-redefined
        ones = tf.expand_dims(tf.ones_like(t), axis=0)
        return tf.transpose(tf.transpose(ones) * 0.01 * tf.ones(
            (4), dtype=t.dtype))
      mean_reversion = [[0.03], [0.03], [0.03], [0.03]]
      volatility = [[0.02], [0.02], [0.02], [0.02]]
      output_shape = [4, 2]
    else:
      zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
      output_shape = [2]

    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)

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
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, output_shape)
    price = self.evaluate(price)
    self.assertAllClose(
        price, 0.7163243383624043 * np.ones(output_shape),
        rtol=error_tol,
        atol=error_tol)

  def test_1d_batch_1d_notional(self):
    """Tests 1-d batch with different notionals."""
    dtype = tf.float64
    error_tol = 1e-2

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
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [0.7163243383624043, 2 * 0.7163243383624043],
        rtol=error_tol,
        atol=error_tol)

  def test_2d_batch_1d(self):
    """Tests 2-d batch."""
    dtype = tf.float64
    error_tol = 1e-2

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
        num_samples=50_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2, 2])
    price = self.evaluate(price)
    expected = [[0.7163243383624043, 0.7163243383624043],
                [0.7163243383624043, 0.7163243383624043]]
    self.assertAllClose(price, expected, rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor(self):
    """Tests model with constant parameters in 2 dimensions."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    error_tol = 1e-2

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
        num_samples=25_000,
        time_step=0.1,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2],
        dtype=dtype)

    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [1])
    price = self.evaluate(price)
    self.assertAllClose(price, [0.802226], rtol=error_tol, atol=error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'monte_carlo',
          'valuation_method': tff.models.ValuationMethod.MONTE_CARLO,
          'error_tol': 5e-3,
      }, {
          'testcase_name': 'pde',
          'valuation_method': tff.models.ValuationMethod.FINITE_DIFFERENCE,
          'error_tol': 2e-3,
      })
  def test_correctness_2_factor_hull_white_consistency(
      self, valuation_method, error_tol):
    """Test that under certain conditions HJM matches analytic HW results.

    Args:
      valuation_method: The valuation method used.
      error_tol: Test error tolerance.

    For the two factor model, when both mean reversions are equivalent, then
    the HJM model matches that of a HW one-factor model with the same mean
    reversion, and effective volatility:
    eff_vol = sqrt(vol1^2 + vol2^2 + 2 rho(vol1 * vol2)
    where rho is the cross correlation between the two factors. In this
    specific test, we assume rho = 0.0.
    """
    dtype = tf.float64

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
        num_samples=25_000,
        valuation_method=valuation_method,
        time_step_finite_difference=0.05,
        num_grid_points_finite_difference=251,
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
        mean_reversion=[mu],
        volatility=[eff_vol],
        use_analytic_pricing=True,
        dtype=dtype)
    hw_price = self.evaluate(hw_price)

    self.assertNear(hjm_price, hw_price, error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'monte_carlo',
          'valuation_method': tff.models.ValuationMethod.MONTE_CARLO,
          'error_tol': 1.2e-3,
      }, {
          'testcase_name': 'pde',
          'valuation_method': tff.models.ValuationMethod.FINITE_DIFFERENCE,
          'error_tol': 4e-3,
      })
  def test_correctness_2_factor_hull_white_consistency_with_corr(
      self, valuation_method, error_tol):
    """Test that under certain conditions HJM matches analytic HW results."""
    dtype = tf.float64
    expiries = np.array([1.0])
    fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
    fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    mu = 0.03
    vol1 = 0.02
    vol2 = 0.01
    rho = -0.5
    eff_vol = np.sqrt(vol1**2 + vol2**2 + 2*rho*vol1*vol2)

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
        corr_matrix=[[1, rho], [rho, 1]],
        valuation_method=valuation_method,
        time_step_finite_difference=0.05,
        num_grid_points_finite_difference=251,
        num_samples=50_000,
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
        mean_reversion=[mu],
        volatility=[eff_vol],
        use_analytic_pricing=True,
        dtype=dtype)
    hw_price = self.evaluate(hw_price)

    self.assertNear(hjm_price, hw_price, error_tol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'time_step',
          'time_step': 0.05,
          'num_time_steps': None,
          'use_xla': False,
      }, {
          'testcase_name': 'num_time_steps',
          'time_step': None,
          'num_time_steps': 21,
          'use_xla': False,
      })
  def test_correctness_2_factor_fd(self, time_step, num_time_steps, use_xla):
    """Tests finite difference valuation for 2-factor model."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    error_tol = 1e-3

    expiries = np.array([1.002739726])
    fixed_leg_payment_times = np.array(
        [1.249315068, 1.498630137, 1.750684932, 2.002739726])
    fixed_leg_daycount_fractions = np.array(
        [0.2465753425, 0.2493150685, 0.2520547945, 0.2520547945])
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    mean_reversion = [0.03, 0.15]
    volatility = [0.01, 0.015]

    def _fn():
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
          valuation_method=tff.models.ValuationMethod.FINITE_DIFFERENCE,
          time_step_finite_difference=time_step,
          num_time_steps_finite_difference=num_time_steps,
          num_grid_points_finite_difference=251,
          time_step=0.1,
          dtype=dtype)
      return price

    if use_xla:
      price = self.evaluate(tf.function(_fn, jit_compile=True)())
    else:
      price = self.evaluate(_fn())
    quantlib_price = 0.5900860719515227
    self.assertAllClose(
        price, [quantlib_price], rtol=error_tol, atol=error_tol)

  def test_correctness_2_factor_batch_fd(self):
    """Tests finite difference valuation for a batch."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    error_tol = 1e-3

    expiries = np.array([1.002739726, 2.002739726])
    fixed_leg_payment_times = np.array(
        [[1.249315068, 1.498630137, 1.750684932, 2.002739726],
         [2.249315068, 2.498630137, 2.750684932, 3.002739726]])
    fixed_leg_daycount_fractions = np.array(
        [[0.2465753425, 0.2493150685, 0.2520547945, 0.2520547945],
         [0.2465753425, 0.2493150685, 0.2520547945, 0.2520547945]])
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    mean_reversion = [0.03, 0.15]
    volatility = [0.01, 0.015]

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
        valuation_method=tff.models.ValuationMethod.FINITE_DIFFERENCE,
        time_step_finite_difference=0.05,
        num_grid_points_finite_difference=251,
        time_step=0.1,
        dtype=dtype)

    quantlib_price = [0.5900860719515227, 0.8029012153434956]
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [quantlib_price[0], quantlib_price[1]],
        rtol=error_tol,
        atol=error_tol)

  def test_correctness_3_factor_batch_fd(self):
    """Tests finite difference valuation for a batch with 3 factor HJM."""
    # 1y x 1y swaption with quarterly payments.
    dtype = tf.float64
    error_tol = 1e-3

    expiries = np.array([1.002739726, 1.002739726])
    fixed_leg_payment_times = np.array(
        [[1.249315068, 1.498630137, 1.750684932, 2.002739726],
         [1.249315068, 1.498630137, 1.750684932, 2.002739726]])
    fixed_leg_daycount_fractions = np.array(
        [[0.2465753425, 0.2493150685, 0.2520547945, 0.2520547945],
         [0.2465753425, 0.2493150685, 0.2520547945, 0.2520547945]])
    fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
    zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)

    mean_reversion = [0.03, 0.15, 0.25]
    volatility = [0.01, 0.015, 0.009]

    price = tff.models.hjm.swaption_price(
        expiries=expiries,
        fixed_leg_payment_times=fixed_leg_payment_times,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        num_hjm_factors=3,
        mean_reversion=mean_reversion,
        volatility=volatility,
        valuation_method=tff.models.ValuationMethod.FINITE_DIFFERENCE,
        time_step_finite_difference=0.05,
        num_grid_points_finite_difference=51,
        time_step=0.1,
        dtype=dtype)

    very_approximate_benchmark = 0.56020399
    self.assertEqual(price.dtype, dtype)
    self.assertAllEqual(price.shape, [2])
    price = self.evaluate(price)
    self.assertAllClose(
        price, [very_approximate_benchmark, very_approximate_benchmark],
        rtol=error_tol,
        atol=error_tol)


if __name__ == '__main__':
  tf.test.main()
