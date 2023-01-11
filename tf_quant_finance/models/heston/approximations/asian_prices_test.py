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

european_option_price = tff.models.heston.approximations.european_option_price
asian_option_price = tff.models.heston.approximations.asian_option_price


@test_util.run_all_in_graph_and_eager_modes
class AsianPriceTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods for the Heston asian pricing module."""

  def test_option_prices(self):
    """Tests for methods for the asian pricing module."""
    # Expected results come from tables (1)-(3) in
    # 'A Recursive Method for Discretely Monitored Geometric Asian Option
    # Prices', Bull. Korean Math. Soc. 53, 733-749 (2016)
    # by B. Kim, J. Kim, J. Kim & I. S. Wee (1y expiry, weekly sampling)
    variances = [0.09, 0.09, 0.09]
    mean_reversion = [1.15, 1.15, 1.15]
    volvol = [0.39, 0.39, 0.39]
    theta = [0.0348, 0.0348, 0.0348]
    rho = [-0.64, -0.64, -0.64]
    spots = [100.0, 100.0, 100.0]
    strikes = [90.0, 100.0, 110.0]
    discount_rates = [0.05, 0.05, 0.05]
    dividend_rates = [0.0, 0.0, 0.0]

    expiries = 1.0
    times = np.linspace(0, expiries, 52)[1:]
    expiries = [expiries, expiries, expiries]
    batch_shape = list(tf.constant(spots).shape)
    sampling_times = (np.ones(shape=list(times.shape) + batch_shape)
                      * tf.expand_dims(times, 1))

    expected_prices = tf.constant([13.6950, 7.2243, 2.9479], dtype=np.float64)

    computed_prices = asian_option_price(
        variances=variances,
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        sampling_times=sampling_times,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        dtype=tf.float64)

    self.assertAllClose(expected_prices, computed_prices, 1e-2)

  def test_single_sample_asian_matches_vanilla(self):
    """Tests that a single sampling asian replicates vanilla prices."""

    variances = tf.constant([0.09, 0.09, 0.09])
    mean_reversion = tf.constant([1.15, 1.15, 1.15])
    volvol = tf.constant([0.39, 0.39, 0.39])
    theta = tf.constant([0.0348, 0.0348, 0.0348])
    rho = tf.constant([-0.64, -0.64, -0.64])
    spots = tf.constant([100.0, 100.0, 100.0])
    strikes = tf.constant([90.0, 100.0, 110.0])
    discount_rates = tf.constant([0.05, 0.05, 0.05])
    dividend_rates = tf.constant([0.0, 0.0, 0.0])

    t_n = tf.constant([1.0], dtype=np.float32)
    expiries = tf.constant([1.0, 1.0, 1.0])
    sampling_times = (tf.ones(shape=t_n.shape + spots.shape)
                      * tf.expand_dims(t_n, 1))

    # Test dynamically shaped sampling_times
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def asian_option_price_tf(variances, spots, sampling_times):
      return asian_option_price(
          variances=variances,
          mean_reversion=mean_reversion,
          theta=theta,
          volvol=volvol,
          rho=rho,
          strikes=strikes,
          expiries=expiries,
          spots=spots,
          sampling_times=sampling_times,
          discount_rates=discount_rates,
          dividend_rates=dividend_rates)

    computed_prices_asians = asian_option_price_tf(
        variances=variances,
        spots=spots,
        sampling_times=sampling_times)

    computed_price_europeans = european_option_price(
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        variances=variances,
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho)

    self.assertAllClose(computed_prices_asians, computed_price_europeans, 1e-4)

  def test_option_prices_scalar_prices(self):
    """Tests for methods for the asian pricing module with scalar inputs."""
    # Expected results come from tables (1)-(3) in 'A Recursive Method for
    # Discretely Monitored Geometric Asian Option Prices', Bull. Korean Math.
    # Soc. 53, 733-749 (2016) by B. Kim, J. Kim, J. Kim & I. S. Wee (6m expiry,
    # daily sampling)

    variances = 0.09
    mean_reversion = 1.15
    volvol = 0.39
    theta = 0.0348
    rho = -0.64
    spots = 100.0
    strikes = [[90.0, 100.0, 100.0], [110.0, 100.0, 110.0]]

    discount_rates = 0.05
    dividend_rates = 0.0

    expiries = 0.5
    sampling_times = np.linspace(
        1/365, expiries, 182)[:, np.newaxis, np.newaxis]

    expected_prices = tf.constant(
        [[11.8924, 5.0910, 5.0910],
         [1.359, 5.0910, 1.3597]], dtype=tf.float64)

    computed_prices = asian_option_price(
        variances=variances,
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        sampling_times=sampling_times,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        dtype=tf.float64)

    self.assertAllClose(expected_prices, computed_prices, 1e-2)

  def test_seasoned_analytic_vs_monte_carlo(self):
    """Tests for methods for the asian pricing module a seasoned asian option.
    """
    # Results are compared to a Monte Carlo simulation.

    variances = 0.09
    mean_reversion = 1.15
    volvol = 0.39
    theta = 0.0348
    rho = -0.64
    spots = 100.0
    strikes = [90.0, 100.0, 110.0]
    discount_rates = 0.0

    expiries = 1.0
    sampling_times = np.linspace(0.25, expiries, 4)[:, np.newaxis]
    past_fixings = [100.0, 100.0, 100.0]

    computed_prices_asians = asian_option_price(
        variances=variances,
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        sampling_times=sampling_times,
        discount_rates=discount_rates,
        past_fixings=past_fixings)

    dtype = np.float64
    num_samples = 100000
    num_timesteps = 24

    dt = 1. / num_timesteps
    strikes_tensor = tf.constant(strikes, dtype=dtype)
    times = tf.constant(np.linspace(0.25, expiries, 4), dtype=dtype)
    log_spot = np.log(spots)

    total_fixings = len(past_fixings) + times.shape[0]
    running_accumulator = np.prod(past_fixings)

    process = tff.models.heston.HestonModel(
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        dtype=dtype)

    initial_state = tf.constant([log_spot, variances], dtype=dtype)
    paths = process.sample_paths(
        times,
        initial_state,
        time_step=dt,
        num_samples=num_samples,
        seed=123)

    prefactor = running_accumulator ** (1 / total_fixings)
    spot_prods = prefactor * tf.math.exp(tf.reduce_sum(paths[:, :, 0], axis=1)
                                         / total_fixings)
    spot_prods = tf.broadcast_to(tf.expand_dims(spot_prods, 1),
                                 spot_prods.shape + strikes_tensor.shape)
    payoffs = tf.nn.relu(spot_prods - strikes_tensor)
    mc_prices = tf.math.reduce_mean(payoffs, 0)

    self.assertAllClose(computed_prices_asians, mc_prices, 1e-2)

  def test_puts_and_calls_match_vanilla(self):
    """Tests that single sampling puts and calls both match vanilla prices."""

    variances = 0.09
    mean_reversion = 1.15
    volvol = 0.39
    theta = 0.0348
    rho = -0.64
    spots = 100.0
    strikes = [90.0, 100.0, 110.0]
    discount_rates = 0.05
    dividend_rates = 0.0

    is_call_options = [False, True, False]

    expiries = tf.constant([1.0, 1.0, 1.0])
    sampling_times = tf.constant([1.0, 1.0, 1.0])[:, tf.newaxis]

    computed_prices_asians = asian_option_price(
        variances=variances,
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        sampling_times=sampling_times,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        is_call_options=is_call_options)

    computed_price_europeans = european_option_price(
        strikes=strikes,
        expiries=expiries,
        spots=spots,
        discount_rates=discount_rates,
        dividend_rates=dividend_rates,
        variances=variances,
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        is_call_options=is_call_options)

    self.assertAllClose(computed_prices_asians, computed_price_europeans, 1e-4)

if __name__ == '__main__':
  tf.test.main()
