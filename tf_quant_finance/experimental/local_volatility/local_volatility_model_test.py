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
"""Tests for the Local volatility model."""

import functools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tf_quant_finance.experimental.local_volatility import local_volatility_model
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

dupire_local_volatility_iv = local_volatility_model._dupire_local_volatility_iv
dupire_local_volatility_prices = local_volatility_model._dupire_local_volatility_prices
implied_vol = tff.black_scholes.implied_vol
LocalVolatilityModel = tff.experimental.local_volatility.LocalVolatilityModel
volatility_surface = tff.experimental.pricing_platform.framework.market_data.volatility_surface


# This function can't be moved to SetUp since that would break graph mode
# execution
def build_tensors(dim):
  """Setup basic test with a flat volatility surface."""
  year = dim * [[2021, 2022]]
  month = dim * [[1, 1]]
  day = dim * [[1, 1]]
  expiries = tff.datetime.dates_from_year_month_day(year, month, day)
  valuation_date = [(2020, 1, 1)]
  expiry_times = tff.datetime.daycount_actual_365_fixed(
      start_date=valuation_date, end_date=expiries, dtype=tf.float64)
  strikes = dim * [[[0.8, 0.9, 1.0, 1.1, 1.3], [0.8, 0.9, 1.0, 1.1, 1.5]]]
  strikes = tf.constant(strikes, dtype=tf.float64)
  iv = dim * [[[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]]]
  spot = dim * [1.0]
  spot = tf.constant(spot, dtype=tf.float64)
  return valuation_date, expiries, expiry_times, strikes, iv, spot


def build_scaled_tensors(dim, scale=1):
  """Similar to build_tensors, but uses a different set of IVs and scale."""
  year = dim * [[2021, 2022]]
  month = dim * [[1, 1]]
  day = dim * [[1, 1]]
  expiries = tff.datetime.dates_from_year_month_day(year, month, day)
  valuation_date = [(2020, 1, 1)]
  expiry_times = tff.datetime.daycount_actual_365_fixed(
      start_date=valuation_date, end_date=expiries, dtype=tf.float64)

  # 10 points
  row = np.arange(0.9, 1.3, 0.04).tolist()
  row = [scale * r for r in row]
  strikes = dim * [[row, row]]
  strikes = tf.constant(strikes, dtype=tf.float64)
  # 10 points
  iv_row = (np.arange(0.21, 0.31, 0.01)).tolist()
  iv = dim * [[iv_row, iv_row]]
  spot = dim * [1.0 * scale]
  spot = tf.constant(spot, dtype=tf.float64)
  return valuation_date, expiries, expiry_times, strikes, iv, spot


def build_volatility_surface(val_date, expiry_times, expiries, strikes, iv,
                             dtype):
  interpolator = tff.math.interpolation.interpolation_2d.Interpolation2D(
      expiry_times, strikes, iv, dtype=dtype)
  def _interpolator(t, x):
    x_transposed = tf.transpose(x)
    t = tf.broadcast_to(t, x_transposed.shape)
    return tf.transpose(interpolator.interpolate(t, x_transposed))

  return volatility_surface.VolatilitySurface(
      val_date, expiries, strikes, iv, interpolator=_interpolator, dtype=dtype)


def callable_discount_factor(t, lower=0.01, upper=0.02):
  # For time dependent rates calculate the discount rate by integrating the risk
  # free rate: exp(-int_0^t r(s) ds)
  return tf.expand_dims(
      tf.where(
          t < 0.5,
          tf.math.exp(-lower * t) * tf.ones_like(t, dtype=tf.float64),
          tf.math.exp(-lower * 0.5 - upper * (t - 0.5)) *
          tf.ones_like(t, dtype=tf.float64)), -1)


@test_util.run_all_in_graph_and_eager_modes
class LocalVolatilityTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('1d_flat', 1, [0.0], None, 20, True, False, 0, True),
      ('1d', 1, [0.0], None, 20, True, False),
      ('2d', 2, [0.0], None, 20, True, False),
      ('3d', 3, [0.0], None, 20, True, False),
      ('1d_nonzero_riskfree_rate', 1, [0.05], None, 40, True, False, 1),
      ('1d_using_vol_surface', 1, [0.0], None, 20, False, False),
      ('1d_with_callable_rate1', 1, None,
       functools.partial(callable_discount_factor,
                         upper=0.02), 40, True, False, 1),
      ('1d_with_callable_rate1_and_vol_surface', 1, None,
       functools.partial(callable_discount_factor,
                         upper=0.02), 40, False, False, 1),
      ('1d_with_callable_rate2', 1, None,
       functools.partial(callable_discount_factor,
                         upper=0.05), 40, True, False, 1),
      ('1d_with_xla', 1, [0.0], None, 20, True, True),
  )
  def test_lv_correctness(self,
                          dim,
                          risk_free_rate,
                          discount_factor_fn,
                          num_time_steps,
                          using_market_data,
                          jit_compile,
                          iv_start_index=0,
                          flat_iv=False):
    """Tests that the model reproduces implied volatility smile."""
    dtype = tf.float64
    num_samples = 10000
    if flat_iv:
      val_date, expiries, expiry_times, strikes, iv, spot = build_tensors(dim)
    else:
      tensors = build_scaled_tensors(dim)
      val_date, expiries, expiry_times, strikes, iv, spot = tensors

    # Handle the cases where we have constant rates.
    if discount_factor_fn is None:
      r = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
      discount_factor_fn = lambda t: tf.math.exp(-r * t)

    if using_market_data:
      lv = LocalVolatilityModel.from_market_data(
          dim=dim,
          valuation_date=val_date,
          expiry_dates=expiries,
          strikes=strikes,
          implied_volatilities=iv,
          spot=spot,
          discount_factor_fn=discount_factor_fn,
          dividend_yield=[0.0],
          dtype=dtype)
    else:
      vs = build_volatility_surface(
          val_date, expiry_times, expiries, strikes, iv, dtype=dtype)
      lv = LocalVolatilityModel.from_volatility_surface(
          dim=dim,
          spot=spot,
          implied_volatility_surface=vs,
          discount_factor_fn=discount_factor_fn,
          dividend_yield=[0.0],
          dtype=dtype)

    @tf.function(jit_compile=jit_compile)
    def _get_sample_paths():
      return lv.sample_paths(
          # Our test times are the same for each dim.
          times=expiry_times[0],
          num_samples=num_samples,
          initial_state=spot,
          num_time_steps=num_time_steps,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2])

    paths = self.evaluate(_get_sample_paths())
    sim_iv = self.evaluate(
        tf.function(_get_all_iv)(
            dim, expiry_times, strikes, spot, paths, iv, iv_start_index,
            discount_factor_fn, dtype))
    num_times = expiry_times[0].shape[0]
    for d in range(dim):
      for i in range(num_times):
        for j in range(iv_start_index, len(iv[0][0])):
          self.assertAllClose(sim_iv[d][i][j - iv_start_index],
                              iv[d][i][j], atol=0.05, rtol=0.005)

  @parameterized.named_parameters(
      ('iv', dupire_local_volatility_iv),
      ('prices', dupire_local_volatility_prices),
  )
  def test_dupire_local_volatility_1d(self, dupire_local_volatility):
    """Tests dupire_local_volatility correctness when dim=1."""
    dim = 1
    dtype = tf.float64
    tensors = build_scaled_tensors(dim)
    val_date, expiries, expiry_times, strikes, iv, initial_spot = tensors
    vs = build_volatility_surface(
        val_date, expiry_times, expiries, strikes, iv, dtype=dtype)
    dividend_yield = [0.]
    r = tf.convert_to_tensor([0.], dtype=dtype)
    discount_factor_fn = lambda t: tf.math.exp(-r * t)

    times = tf.convert_to_tensor([1., 2., 3.], dtype=dtype)
    spots = tf.reshape(tf.convert_to_tensor([1., 2., 3.], dtype=dtype), [3, 1])
    dupire_vols = dupire_local_volatility(times, spots, initial_spot,
                                          vs.volatility, discount_factor_fn,
                                          dividend_yield)
    true_vols = [[0.22848, 0.3, 0.3], [0.217881, 0.3, 0.3],
                 [0.217398, 0.3, 0.3]]
    for i in range(3):
      self.assertAllClose(
          dupire_vols[:, i], true_vols[i], atol=0.05, rtol=0.005)

  @parameterized.named_parameters(
      ('iv', dupire_local_volatility_iv),
      ('prices', dupire_local_volatility_prices),
  )
  def test_dupire_local_volatility_2d(self, dupire_local_volatility):
    """Tests dupire_local_volatility correctness when dim=2."""
    dim = 2
    dtype = tf.float64
    tensors = build_scaled_tensors(dim)
    val_date, expiries, expiry_times, strikes, iv, initial_spot = tensors

    vs = build_volatility_surface(
        val_date, expiry_times, expiries, strikes, iv, dtype=dtype)
    dividend_yield = [0.]
    r = tf.convert_to_tensor([0.], dtype=dtype)
    discount_factor_fn = lambda t: tf.math.exp(-r * t)

    times = tf.convert_to_tensor([1., 2.], dtype=dtype)
    times = tf.broadcast_to(times, [2, 2])
    spots = [[1., 1.5], [2., 2.5]]
    spots = tf.convert_to_tensor(spots, dtype=dtype)
    dupire_vols = dupire_local_volatility(times, spots, initial_spot,
                                          vs.volatility, discount_factor_fn,
                                          dividend_yield)
    true_vols = [[0.22848, 0.3], [0.3, 0.3]]
    for i in range(2):
      self.assertAllClose(
          dupire_vols[:, i], true_vols[i], atol=0.05, rtol=0.005)

  @parameterized.named_parameters(
      ('iv', dupire_local_volatility_iv),
      ('prices', dupire_local_volatility_prices),
  )
  def test_dupire_with_flat_surface(self, dupire_local_volatility):
    """Tests dupire_local_volatility with a flat vol surface."""
    dim = 1
    dtype = tf.float64
    val_date, expiries, expiry_times, strikes, iv, initial_spot = build_tensors(
        dim)
    vs = build_volatility_surface(
        val_date, expiry_times, expiries, strikes, iv, dtype=dtype)
    dividend_yield = [0.]
    r = tf.convert_to_tensor([0.], dtype=dtype)
    discount_factor_fn = lambda t: tf.math.exp(-r * t)

    times = tf.convert_to_tensor([1., 2., 3.], dtype=dtype)
    spots = tf.reshape(tf.convert_to_tensor([1., 2., 3.], dtype=dtype), [3, 1])
    dupire_vols = dupire_local_volatility(times, spots, initial_spot,
                                          vs.volatility, discount_factor_fn,
                                          dividend_yield)
    true_vols = [0.1] * 3
    for i in range(3):
      self.assertAllClose(dupire_vols[:, i], true_vols)

  @parameterized.named_parameters(
      ('1d', 1, [0.0], None, 20, True, False),
      ('2d', 2, [0.0], None, 20, True, False),
      ('3d', 3, [0.0], None, 20, True, False, 0),
      ('1d_nonzero_riskfree_rate', 1, [0.05], None, 20, True, False, 1),
      ('1d_using_vol_surface', 1, [0.0], None, 20, False, False),
      ('1d_with_callable_rate1', 1, None,
       functools.partial(callable_discount_factor,
                         upper=0.02), 20, True, False, 1),
      ('1d_with_callable_rate1_and_vol_surface', 1, None,
       functools.partial(callable_discount_factor,
                         upper=0.02), 20, False, False, 1),
      ('1d_with_callable_rate2', 1, None,
       functools.partial(callable_discount_factor,
                         upper=0.05), 20, True, False, 1),
      ('1d_with_xla', 1, [0.0], None, 20, True, True),
  )
  def test_interpolated_lv_correctness(self,
                                       dim,
                                       risk_free_rate,
                                       discount_factor_fn,
                                       num_time_steps,
                                       using_market_data,
                                       jit_compile,
                                       iv_start_index=0):
    """Tests that the model reproduces implied volatility smile."""
    dtype = tf.float64
    num_samples = 5000
    precompute_iv = True
    tensors = build_scaled_tensors(dim)
    val_date, expiries, expiry_times, strikes, iv, spot = tensors

    # Handle the cases where we have constant rates.
    if discount_factor_fn is None:
      r = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
      discount_factor_fn = lambda t: tf.math.exp(-r * t)

    # Grids for interpolation.
    # 10 points, matching `strikes`
    spot_grid = np.arange(0.9, 1.3, 0.04).tolist()
    times_grid = tf.range(
        start=0., limit=3., delta=3. / num_time_steps, dtype=dtype)
    times_grid = tf.sort(tf.concat([times_grid, expiry_times[0]], 0))

    if using_market_data:
      lv = LocalVolatilityModel.from_market_data(
          dim=dim,
          valuation_date=val_date,
          expiry_dates=expiries,
          strikes=strikes,
          implied_volatilities=iv,
          spot=spot,
          discount_factor_fn=discount_factor_fn,
          dividend_yield=[0.0],
          times_grid=times_grid,
          spot_grid=spot_grid,
          precompute_iv=precompute_iv,
          dtype=dtype)
    else:
      vs = build_volatility_surface(
          val_date, expiry_times, expiries, strikes, iv, dtype=dtype)
      lv = LocalVolatilityModel.from_volatility_surface(
          dim=dim,
          spot=spot,
          implied_volatility_surface=vs,
          discount_factor_fn=discount_factor_fn,
          dividend_yield=[0.0],
          times_grid=times_grid,
          spot_grid=spot_grid,
          precompute_iv=precompute_iv,
          dtype=dtype)

    @tf.function(jit_compile=jit_compile)
    def _get_sample_paths():
      return lv.sample_paths(
          # Our test times are the same for each dim.
          times=expiry_times[0],
          num_samples=num_samples,
          initial_state=spot,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[1, 2])

    paths = self.evaluate(_get_sample_paths())
    num_times = expiry_times[0].shape[0]
    sim_iv = self.evaluate(
        tf.function(_get_all_iv)(
            dim, expiry_times, strikes, spot, paths, iv, iv_start_index,
            discount_factor_fn, dtype))
    for d in range(dim):
      for i in range(num_times):
        for j in range(iv_start_index, len(iv[0][0])):
          self.assertAllClose(sim_iv[d][i][j - iv_start_index],
                              iv[d][i][j], atol=0.05, rtol=0.005)


def _get_all_iv(
    dim, expiry_times, strikes, spot, paths, iv, iv_start_index,
    discount_factor_fn, dtype):
  num_times = expiry_times[0].shape[0]
  sim_iv = dim * [num_times * [(len(iv[0][0]) - iv_start_index) * [0]]]
  for d in range(dim):
    for i in range(num_times):
      for j in range(iv_start_index, len(iv[0][0])):
        sim_iv[d][i][j - iv_start_index] = _get_implied_vol(
            expiry_times[d][i], strikes[d][i][j],
            paths[:, i, d], spot[d], discount_factor_fn, dtype)[0]
  return sim_iv


def _get_implied_vol(time, strike, paths, spot, discount_factor_fn, dtype):
  discount_factor = discount_factor_fn(time)
  num_not_nan = tf.cast(
      paths.shape[0] - tf.math.count_nonzero(tf.math.is_nan(paths)),
      paths.dtype)
  paths = tf.where(tf.math.is_nan(paths), tf.zeros_like(paths), paths)
  # Calculate reduce_mean of paths. Workaround for XLA compatibility.
  option_value = tf.math.divide_no_nan(
      tf.reduce_sum(tf.nn.relu(paths - strike)), num_not_nan)
  # ITM option value can't fall below intrinsic value.
  option_value = tf.maximum(option_value, spot - strike)

  iv = implied_vol(
      prices=discount_factor * option_value,
      strikes=strike,
      expiries=time,
      spots=spot,
      discount_factors=discount_factor,
      dtype=dtype,
      validate_args=False)
  return iv

if __name__ == '__main__':
  tf.test.main()
