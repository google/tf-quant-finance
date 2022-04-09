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
"""Tests for the Local stochastic volatility model."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

bs = tff.black_scholes
lsv = tff.experimental.local_stochastic_volatility
volatility_surface = tff.experimental.pricing_platform.framework.market_data.volatility_surface


# This function can't be moved to SetUp since that would break graph mode
# execution
def build_tensors(dim, spot, risk_free_rate):
  year = [[2021, 2022]] * dim
  month = [[1, 1]] * dim
  day = [[1, 1]] * dim
  expiries = tff.datetime.dates_from_year_month_day(year, month, day)
  valuation_date = [(2020, 1, 1)]
  expiry_times = tff.datetime.daycount_actual_365_fixed(
      start_date=valuation_date, end_date=expiries, dtype=tf.float64)
  moneyness = [[[0.1, 0.9, 1.0, 1.1, 3], [0.1, 0.9, 1.0, 1.1, 3]]] * dim
  strikes = spot * np.array(moneyness) * np.exp(
      risk_free_rate * np.array([[1.0], [2.0]]))
  iv = [[[0.135, 0.12, 0.1, 0.11, 0.13], [0.135, 0.12, 0.1, 0.11, 0.13]]] * dim
  return valuation_date, expiries, expiry_times, strikes, iv


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


@test_util.run_all_in_graph_and_eager_modes
class LocalStochasticVolatilityTest(tf.test.TestCase, parameterized.TestCase):

  def get_implied_vol(self, time, strike, paths, spot, r, dtype):
    r = tf.convert_to_tensor(r, dtype=dtype)
    discount_factor = tf.math.exp(-r * time)
    paths = tf.boolean_mask(paths, tf.math.logical_not(tf.math.is_nan(paths)))
    option_value = tf.math.reduce_mean(tf.nn.relu(paths - strike))
    iv = bs.implied_vol(
        prices=discount_factor * option_value,
        strikes=strike,
        expiries=time,
        spots=spot,
        discount_factors=discount_factor,
        dtype=dtype,
        validate_args=True)
    return iv

  @parameterized.named_parameters(
      ('1d', 1, 0.0, [0.0], [1.0], [1.0], 0.1, 0.1, 0.0, 0.2, True),
      ('1d_corr', 1, -0.5, [0.0], [1.0], [1.0], 0.1, 0.1, 0.0, 0.2, True),
      ('1d_nonzero_rate', 1, 0.0, [0.05], [1.0], [1.0
                                                 ], 0.1, 0.1, 0.0, 0.2, True),
      ('1d_low_var', 1, 0.0, [0.0], [1.0], [0.04], 0.1, 0.1, 0.0, 0.2, True),
      ('1d_high_volvol', 1, 0.0, [0.0], [1.0], [0.04
                                               ], 0.1, 0.1, 1.0, 0.5, True),
      ('1d_using_vol_surface', 1, 0.0, [0.0], [1.0], [1.0], 0.1, 0.1, 0.0, 0.2,
       False),
  )
  def test_lv_correctness(self, dim, rho, risk_free_rate, spot, variance,
                          pde_time_step, sim_time_step, mr, volvol,
                          using_market_data):
    """Tests that the model reproduces implied volatility smile."""
    dtype = tf.float64
    num_samples = 10000
    var_model = lsv.LSVVarianceModel(
        mr, variance, volvol * np.sqrt(variance), dtype=dtype)
    val_date, expiries, expiry_times, strikes, iv = build_tensors(
        dim, spot, risk_free_rate)
    if using_market_data:
      model = lsv.LocalStochasticVolatilityModel.from_market_data(
          val_date,
          expiries,
          strikes,
          iv,
          var_model,
          spot,
          variance,
          rho,
          risk_free_rate, [0.0],
          pde_time_step,
          num_grid_points=100,
          dtype=dtype)
    else:
      vs = build_volatility_surface(
          val_date, expiry_times, expiries, strikes, iv, dtype=dtype)
      model = lsv.LocalStochasticVolatilityModel.from_volatility_surface(
          vs,
          var_model,
          spot,
          variance,
          rho,
          risk_free_rate, [0.0],
          pde_time_step,
          num_grid_points=100,
          dtype=dtype)

    paths = model.sample_paths(
        [1.0, 2.0],
        num_samples=num_samples,
        initial_state=[spot[0], variance[0]],
        time_step=sim_time_step,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 2])

    for d in range(dim):
      for i in range(2):
        for j in [1, 2, 3]:
          sim_iv = self.evaluate(
              self.get_implied_vol(expiry_times[d][i], strikes[d][i][j],
                                   paths[:, i,
                                         d], spot[d], risk_free_rate, dtype))
          self.assertAllClose(sim_iv[0], iv[d][i][j], atol=0.007, rtol=0.007)


if __name__ == '__main__':
  tf.test.main()
