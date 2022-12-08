# Copyright 2022 Google LLC
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
"""Tests for asian_prices."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class AsianPriceTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods for the asian pricing module."""

  def test_option_prices(self):
    """Tests for methods for the asian pricing module.

    Results come from 'Implementing Derivatives Model' by Clewlow, Strickland
    p.118-123, via the QuantLib test suite
    """
    spots = np.array([100.0])
    dividend_rates = np.array([0.03])
    discount_rates = np.array([0.06])
    volatilities = np.array([0.2])
    strikes = np.array([100.0])
    expiries = 1.0
    sampling_times = np.linspace(0.1, 1, 10)[:, np.newaxis]

    expected_price = np.array([5.3425606635])

    computed_price = tff.black_scholes.asian_option_price(
        spots=spots,
        dividend_rates=dividend_rates,
        discount_rates=discount_rates,
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        sampling_times=sampling_times)

    self.assertAllClose(expected_price, computed_price, 1e-10)

  def test_option_prices_scalar_inputs(self):
    """Tests for methods for the asian pricing module using scalar inputs.

    Results come from 'Implementing Derivatives Model' by Clewlow, Strickland
    p.118-123, via the QuantLib test suite
    """
    spots = 100.0
    dividend_rates = 0.03
    discount_rates = 0.06
    volatilities = 0.2
    strikes = 100.0
    expiries = 1.0
    sampling_times = np.linspace(0.1, 1, 10)[:, np.newaxis]

    expected_price = np.array([5.3425606635])

    computed_price = tff.black_scholes.asian_option_price(
        spots=spots,
        dividend_rates=dividend_rates,
        discount_rates=discount_rates,
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        sampling_times=sampling_times,
        dtype=np.float64)

    self.assertAllClose(expected_price, computed_price, 1e-10)

  def test_single_sample_asian_matches_vanilla(self):
    """Tests that a single sampling asian replicates vanilla prices."""
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_price_vanillas = self.evaluate(
        tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards))
    sampling_times = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    computed_price_asians = self.evaluate(
        tff.black_scholes.asian_option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            sampling_times=sampling_times))
    self.assertArrayNear(computed_price_vanillas, computed_price_asians, 1e-10)

  def test_analytic_pricer_vs_monte_carlo(self):
    """Tests that analytic calculator and monte carlo match closely."""
    volatilities = np.array([0.1])
    spots = np.array([650.0, 650.0, 650.0, 650.0, 650.0])
    strikes = np.array([550.0, 600.0, 650.0, 680.0, 730.0])
    expiries = np.array([1.0])
    discount_rates = np.array([0.03])
    dividend_rates = np.array([0.01])

    sampling_times = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                               [1.0, 1.0, 1.0, 1.0, 1.0]])

    computed_prices = tff.black_scholes.asian_option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        sampling_times=sampling_times)

    dtype = np.float64
    num_samples = 500000
    num_timesteps = 100

    dt = 1. / num_timesteps
    sigma = tf.constant(volatilities[0], dtype=dtype)
    spot = tf.constant(spots[0], dtype=dtype)
    sampling_times = tf.constant([0.5, 1.0], dtype=dtype)
    sampling_times = tf.constant([0.5, 1.0], dtype=dtype)

    def set_up_pricer(expiries, watch_params=False):

      def price_asian_options(strikes, spot, sigma):
        # Define drift and volatility functions.
        def drift_fn(t, x):
          del t, x
          return discount_rates - dividend_rates - 0.5 * sigma**2

        def vol_fn(t, x):
          del t, x
          return tf.reshape(sigma, [1, 1])

        process = tff.models.GenericItoProcess(
            dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)
        log_spot = tf.math.log(tf.reduce_mean(spot))
        if watch_params:
          watch_params_list = [sigma]
        else:
          watch_params_list = None
        paths = process.sample_paths(
            sampling_times,
            num_samples=num_samples,
            initial_state=log_spot,
            watch_params=watch_params_list,
            random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
            seed=43,
            time_step=dt)
        prices = (
            tf.exp(-tf.expand_dims(discount_rates * expiries, axis=-1)) *
            tf.reduce_mean(
                tf.nn.relu(tf.math.exp(tf.reduce_mean(paths, [1])) - strikes),
                [0]))
        return prices

      return price_asian_options

    price_asian_options = tf.function(set_up_pricer(expiries))
    mc_prices = price_asian_options(strikes, spot, sigma)

    self.assertAllClose(computed_prices, mc_prices[0], 1e-2)


if __name__ == '__main__':
  tf.test.main()
