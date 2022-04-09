# Copyright 2019 Google LLC
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
"""Tests for Heston Model."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

HestonModel = tff.models.HestonModel
grids = tff.math.pde.grids


@test_util.run_all_in_graph_and_eager_modes
class HestonModelTest(parameterized.TestCase, tf.test.TestCase):

  def test_volatility(self):
    """Tests volatility stays close to its mean for small vol of vol."""
    theta = 0.05
    process = HestonModel(mean_reversion=1.0, theta=theta, volvol=0.00001,
                          rho=-0.0, dtype=np.float64)
    years = 1.0
    times = np.linspace(0.0, years, int(365 * years))
    num_samples = 2
    paths = process.sample_paths(
        times,
        time_step=0.01,
        num_samples=num_samples,
        initial_state=np.array([np.log(100), 0.045]),
        seed=None)
    # For small values of volvol, volatility should stay close to theta
    volatility_trace = self.evaluate(paths)[..., 1]
    max_deviation = np.max(abs(volatility_trace[:, 50:] - theta))
    self.assertAlmostEqual(max_deviation, 0.0, places=2)

  def test_state(self):
    """Tests state behaves like GBM for small vol of vol."""
    theta = 1.0
    process = HestonModel(mean_reversion=1.0, theta=theta, volvol=0.00001,
                          rho=-0.0, dtype=np.float64)
    times = [0.0, 0.5, 1.0]
    num_samples = 1000
    start_value = 100
    paths = process.sample_paths(
        times,
        time_step=0.001,
        num_samples=num_samples,
        initial_state=np.array([np.log(start_value), 1.0]),
        seed=None)
    # For small values of volvol, state should behave like Geometric
    # Brownian Motion with volatility `theta`.
    state_trace = self.evaluate(paths)[..., 0]
    # Starting point should be the same
    np.testing.assert_allclose(state_trace[:, 0], np.log(100), 1e-8)
    for i in (1, 2):
      # Mean and variance of the approximating Geometric Brownian Motions
      gbm_mean = start_value
      gbm_std = start_value * np.sqrt((np.exp(times[i]) - 1))
      np.testing.assert_allclose(np.mean(np.exp(state_trace[:, i])),
                                 gbm_mean, 1.0)
      np.testing.assert_allclose(np.std(np.exp(state_trace[:, 1])),
                                 gbm_std, 2.0)

  def test_expected_total_variance_scalar(self):
    """Tests the expected total variance calculation."""
    mean_reversion = 1.3
    theta = 0.2
    process = HestonModel(
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=0.01,
        rho=0.1,
        dtype=np.float64)
    future_time = 1.2
    initial_var = 0.3
    expected_var = self.evaluate(
        process.expected_total_variance(future_time, initial_var))
    ground_var = (initial_var - theta) * (1 - np.exp(
        -mean_reversion * future_time)) / mean_reversion + theta * future_time
    np.testing.assert_allclose(expected_var, ground_var)

  def test_expected_var_mc(self):
    """Tests the expected total var calculation matches Monte Carlo results."""

    # Define process
    initial_var = 0.1
    rho = -0.5
    volvol = 1.0
    mean_reversion = 10.0
    theta = 0.04
    dtype = tf.float64
    process = HestonModel(
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        dtype=dtype)

    # Draw paths
    num_samples = 10000
    num_times = 252
    initial_state = tf.constant([1.0, initial_var], dtype=dtype)
    future_time = 1.0
    times = tf.constant(np.linspace(0, future_time, num_times), dtype=dtype)
    time_step = 0.1
    paths = process.sample_paths(
        times,
        initial_state,
        time_step=time_step,
        num_samples=num_samples,
        seed=123)

    # Monte carlo estimate of discretized integral.
    mc_estimate_var = self.evaluate(tf.math.reduce_mean(
        tf.math.reduce_sum(tff.math.diff(times) * paths[:, :, 1], axis=1)))

    # Compare to total var from formula implementation.
    expected_var = self.evaluate(
        process.expected_total_variance(future_time, initial_var))
    np.testing.assert_allclose(expected_var, mc_estimate_var, rtol=0.01)

  def test_expected_total_variance_batch(self):
    """Tests the expected total variance calculation."""
    mean_reversion = 1.3
    theta = 0.2
    process = HestonModel(
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=0.01,
        rho=-0.0,
        dtype=np.float64)
    future_time = np.array([0.1, 1.0, 2.0])
    initial_var = np.array([0.1, 0.2, 0.3])
    expected_var = self.evaluate(
        process.expected_total_variance(future_time, initial_var))
    ground_var = (initial_var - theta) * (1 - np.exp(
        -mean_reversion * future_time)) / mean_reversion + theta * future_time
    np.testing.assert_allclose(expected_var, ground_var)

  def test_expected_var_raises_on_piecewise_params(self):
    """Tests the claimed error is raised for piecewise params."""
    dtype = tf.float64
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[1, 1.1], dtype=dtype)
    theta = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[1, 0.9], dtype=dtype)
    volvol = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.3], values=[0.1, 0.2], dtype=dtype)
    rho = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[0.4, 0.6], dtype=dtype)
    process = HestonModel(
        mean_reversion=mean_reversion,
        theta=theta,
        volvol=volvol,
        rho=rho,
        dtype=dtype)
    with self.assertRaises(ValueError):
      process.expected_total_variance(1.0, 1.0)

  @parameterized.named_parameters(
      ('SinglePrecision', np.float32),
      ('DoublePrecision', np.float64),
      ('AutoDtype', None))
  def test_piecewise_and_dtype(self, dtype):
    """Tests that piecewise constant coefficients can be handled."""
    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[1, 1.1], dtype=dtype)
    theta = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[1, 0.9], dtype=dtype)
    volvol = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.3], values=[0.1, 0.2], dtype=dtype)
    rho = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.5], values=[0.4, 0.6], dtype=dtype)
    process = HestonModel(
        mean_reversion=mean_reversion, theta=theta, volvol=volvol,
        rho=rho, dtype=dtype)
    times = [0.1, 1.0]
    num_samples = 100
    initial_state = np.array([np.log(100), 0.045], dtype=dtype)
    paths = process.sample_paths(
        times,
        time_step=0.1,
        num_samples=num_samples,
        initial_state=initial_state,
        seed=None)
    paths = self.evaluate(paths)
    state_trace, volatility_trace = paths[..., 0], paths[..., 0]
    if dtype is not None:
      with self.subTest('VolatilityDtype'):
        self.assertEqual(volatility_trace.dtype, dtype)
      with self.subTest('StateDtype'):
        self.assertEqual(state_trace.dtype, dtype)
    # Check drift and volatility calculation
    dtype = paths.dtype
    initial_state = tf.convert_to_tensor(initial_state, dtype=dtype)
    with self.subTest('Drift'):
      self.assertAllClose(
          process.drift_fn()(times[0], initial_state),
          np.array([-0.0225, 0.955]))
    with self.subTest('Volatility'):
      self.assertAllClose(
          process.volatility_fn()(times[0], initial_state),
          np.array([[0.21213203, 0.],
                    [0.00848528, 0.01944222]]))

  def test_compare_monte_carlo_to_backward_pde(self):
    dtype = tf.float64
    mean_reversion = 0.3
    theta = 0.05
    volvol = 0.02
    rho = 0.1
    maturity_time = 1.0
    initial_log_spot = 3.0
    initial_vol = 0.05
    strike = 15
    discounting = 0.5

    heston = HestonModel(
        mean_reversion=mean_reversion, theta=theta, volvol=volvol, rho=rho,
        dtype=dtype)
    initial_state = np.array([initial_log_spot, initial_vol])
    samples = heston.sample_paths(
        times=[maturity_time / 2, maturity_time],
        initial_state=initial_state,
        time_step=0.01,
        num_samples=1000,
        random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
        seed=[1, 42])
    self.assertEqual(samples.shape, [1000, 2, 2])
    log_spots = samples[:, -1, 0]
    monte_carlo_price = (
        tf.constant(np.exp(-discounting * maturity_time), dtype=dtype) *
        tf.math.reduce_mean(tf.nn.relu(tf.math.exp(log_spots) - strike)))

    s_min, s_max = 2, 4
    v_min, v_max = 0.03, 0.07
    grid_size_s, grid_size_v = 101, 101
    time_step = 0.01

    grid = grids.uniform_grid(minimums=[s_min, v_min],
                              maximums=[s_max, v_max],
                              sizes=[grid_size_s, grid_size_v],
                              dtype=dtype)

    s_mesh, _ = tf.meshgrid(grid[0], grid[1], indexing='ij')
    final_value_grid = tf.nn.relu(tf.math.exp(s_mesh) - strike)
    value_grid = heston.fd_solver_backward(
        start_time=1.0,
        end_time=0.0,
        coord_grid=grid,
        values_grid=final_value_grid,
        time_step=time_step,
        discounting=lambda *args: discounting)[0]
    pde_price = value_grid[int(grid_size_s / 2), int(grid_size_v / 2)]

    self.assertAllClose(monte_carlo_price, pde_price, atol=0.1, rtol=0.1)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SamplePathsWithNumTimeStep',
          'use_num_time_step': True,
          'use_time_grid': False,
          'supply_normal_draws': False,
      }, {
          'testcase_name': 'SamplePathsWithGrid',
          'use_num_time_step': False,
          'use_time_grid': True,
          'supply_normal_draws': False,
      }, {
          'testcase_name': 'SamplePathsWithGridAndDraws',
          'use_num_time_step': False,
          'use_time_grid': True,
          'supply_normal_draws': True,
      })
  def test_compare_monte_carlo_to_european_option(
      self, use_num_time_step, use_time_grid, supply_normal_draws):
    dtype = tf.float64
    mean_reversion_value = 0.3
    theta = 0.05
    volvol = 0.02
    rho = 0.1
    maturity_time = 1.0
    initial_log_spot = 3.0
    initial_vol = 0.05
    strike = 15
    discounting = 0.5

    mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=[0.1, 0.2],
        values=[mean_reversion_value, mean_reversion_value,
                mean_reversion_value], dtype=dtype)

    heston = HestonModel(mean_reversion=mean_reversion, theta=theta,
                         volvol=volvol, rho=rho, dtype=dtype)

    times = [maturity_time / 2, maturity_time]
    num_samples = 10000
    time_step = None
    times_grid = None
    num_time_steps = None
    normal_draws = None
    seed = [1, 42]

    if use_num_time_step:
      num_time_steps = 100
    else:
      time_step = 0.01
    if use_time_grid:
      times_grid = tf.constant(np.linspace(0.0, 1.0, 101), dtype=dtype)
    if supply_normal_draws:
      num_samples = 1
      # Use antithetic sampling
      normal_draws = tf.random.stateless_normal(
          shape=[5000, 100, 2],
          seed=seed,
          dtype=dtype)
      normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
    random_type = tff.math.random.RandomType.STATELESS_ANTITHETIC

    initial_state = np.array([initial_log_spot, initial_vol])
    samples = heston.sample_paths(
        times=times,
        initial_state=initial_state,
        time_step=time_step,
        num_samples=num_samples,
        random_type=random_type,
        seed=seed,
        num_time_steps=num_time_steps,
        normal_draws=normal_draws,
        times_grid=times_grid)
    self.assertEqual(samples.shape, [10000, 2, 2])
    log_spots = samples[:, -1, 0]
    monte_carlo_price = (
        np.exp(-discounting * maturity_time) *
        tf.math.reduce_mean(
            tf.nn.relu(tf.math.exp(log_spots) * np.exp(
                discounting * maturity_time) - strike)))

    # Calulating European option price using above parameters
    dtype = np.float64
    variances = initial_vol
    discount_rates = discounting
    expiries = maturity_time
    mean_reversion = mean_reversion_value

    spots = np.exp(initial_log_spot)
    forwards = None

    european_option_price = self.evaluate(
        tff.models.heston.approximations.european_option_price(
            mean_reversion=mean_reversion,
            theta=theta,
            volvol=volvol,
            rho=rho,
            variances=variances,
            forwards=forwards,
            spots=spots,
            expiries=expiries,
            strikes=strike,
            discount_rates=discount_rates,
            dtype=dtype))

    # Comparing monte carlo and european option price
    self.assertAllClose(
        monte_carlo_price, european_option_price, atol=0.1, rtol=0.1)

if __name__ == '__main__':
  tf.test.main()
