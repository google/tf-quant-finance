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
"""Tests for Geometric Brownian Motion."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class GeometricBrownianMotionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_univariate_constant_drift_and_volatility(self, dtype):
    """Tests univariate GBM constant drift and volatility functions."""
    drift_in = 0.05
    vol_in = 0.5
    process = tff.models.GeometricBrownianMotion(drift_in, vol_in, dtype=dtype)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1.], [2.], [3.]], dtype=dtype)
    with self.subTest("Drift"):
      drift = drift_fn(0.2, state)
      expected_drift = state * drift_in
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest("Volatility"):
      vol = volatility_fn(0.2, state)
      expected_vol = state * vol_in
      self.assertAllClose(vol, np.expand_dims(expected_vol, axis=-1),
                          atol=1e-8, rtol=1e-8)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_univariate_time_varying_drift_and_volatility(self, dtype):
    """Tests univariate GBM time varying drift and volatility functions."""
    # Generate times series for the drift and volatility.
    times = np.linspace(0.0, 10.0, 6, dtype=dtype)
    drift = np.append([0.0], np.sin(times, dtype=dtype))
    sigma = np.append([0.0], np.cos(times, dtype=dtype)) ** 2.0
    drift_in = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=times, values=drift, dtype=dtype)
    sigma_in = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=times, values=sigma, dtype=dtype)
    process = tff.models.GeometricBrownianMotion(
        drift_in, sigma_in, dtype=dtype)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1.], [2.], [3.]], dtype=dtype)
    test_times = np.array([1.0, 3.5, 7.5, 9.8, 12], dtype=dtype)
    expected_drift = drift_in(test_times)
    expected_sigma = sigma_in(test_times)

    with self.subTest("Drift"):
      drift = drift_fn(test_times, state)
      self.assertAllClose(drift, expected_drift*state, atol=1e-8, rtol=1e-8)

    with self.subTest("Volatility"):
      vol = volatility_fn(test_times, state)
      self.assertAllClose(vol, expected_sigma * tf.expand_dims(state, -1),
                          atol=1e-8, rtol=1e-8)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_univariate_integrate_parameter(self, dtype):
    """Tests univariate GBM integrate parameter."""
    # Generate times series for the volatility.
    times = np.linspace(0.0, 10.0, 6, dtype=dtype)
    a_constant = 0.3
    # PiecewiseConstFunc's first value is for the interval [-Inf, times[0]).
    time_varying = np.append([0], np.cos(times, dtype=dtype))
    time_varying_fn = tff.math.piecewise.PiecewiseConstantFunc(
        jump_locations=times, values=time_varying, dtype=dtype)
    process = tff.models.GeometricBrownianMotion(
        a_constant, a_constant, dtype=dtype)
    start_times = np.array([1.0, 3.5, 7.5, 12], dtype=dtype)
    end_times = np.array([2.0, 6.0, 8.5, 16], dtype=dtype)

    # Rounding errors in single precision reduce the precision and require a
    # lower tolerance.
    atol, rtol = (1e-7, 1e-6) if dtype == np.float32 else (1e-8, 1e-8)

    with self.subTest("Constant"):
      integral = process._integrate_parameter(a_constant, True, start_times,
                                              end_times)
      expected = a_constant * (end_times - start_times)
      self.assertAllClose(integral, expected, atol=atol, rtol=rtol)

    with self.subTest("Time dependent"):
      integral = process._integrate_parameter(time_varying_fn, False,
                                              start_times, end_times)
      expected = np.array(
          [
              # Test interval [1.0, 2.0] takes value from t = 0.0.
              1.0 * np.cos(0.0, dtype=dtype),
              # Test interval [3.5, 6.0] is covered by the piecewise constant
              # intervals {[2.0, 4.0), [4.0, 6.0]}.
              0.5 * np.cos(2.0, dtype=dtype) + 2.0 * np.cos(4.0, dtype=dtype),
              # Test interval [7.5, 8.5] is covered by the piecewise constant
              # intervals {[6.0, 8.0), [8.0, 10.0]}.
              0.5 * np.cos(6.0, dtype=dtype) + 0.5 * np.cos(8.0, dtype=dtype),
              # Test interval [12, 16] takes the value from t=10
              4.0 * np.cos(10.0, dtype=dtype)
          ],
          dtype=dtype)

      self.assertAllClose(integral, expected, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_multivariate_drift_and_volatility(self, dtype):
    """Tests multivariate GBM drift and volatility functions."""
    means = [0.05, 0.02]
    volatilities = [0.1, 0.2]
    corr_matrix = [[1, 0.1], [0.1, 1]]
    process = tff.models.MultivariateGeometricBrownianMotion(
        dim=2, means=means, volatilities=volatilities, corr_matrix=corr_matrix,
        dtype=tf.float64)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
    with self.subTest("Drift"):
      drift = drift_fn(0.2, state)
      expected_drift = np.array(means) * state
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest("Volatility"):
      vol = volatility_fn(0.2, state)
      expected_vol = np.expand_dims(
          np.array(volatilities) * state,
          axis=-1) * np.linalg.cholesky(corr_matrix)
      self.assertAllClose(vol, expected_vol, atol=1e-8, rtol=1e-8)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_multivariate_drift_and_volatility_no_corr(self, dtype):
    """Tests multivariate GBM drift and volatility functions."""
    means = [0.05, 0.02]
    volatilities = [0.1, 0.2]
    corr_matrix = [[1, 0.0], [0.0, 1]]
    process = tff.models.MultivariateGeometricBrownianMotion(
        dim=2, means=means, volatilities=volatilities,
        dtype=tf.float64)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
    with self.subTest("Drift"):
      drift = drift_fn(0.2, state)
      expected_drift = np.array(means) * state
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest("Volatility"):
      vol = volatility_fn(0.2, state)
      expected_vol = np.expand_dims(
          np.array(volatilities) * state,
          axis=-1) * np.linalg.cholesky(corr_matrix)
      self.assertAllClose(vol, expected_vol, atol=1e-8, rtol=1e-8)

  def process_sample_paths_mean_and_variance(self, mu, sigma, times,
                                             initial_state, dtype):
    """Returns the mean and variance of the sample paths for a process."""
    process = tff.models.GeometricBrownianMotion(mu, sigma, dtype=dtype)
    samples = process.sample_paths(times=times, initial_state=initial_state,
                                   random_type=tff.math.random.RandomType.SOBOL,
                                   num_samples=100000)
    log_s = tf.math.log(samples)
    mean = tf.reduce_mean(log_s, axis=0, keepdims=True)
    var = tf.reduce_mean((log_s - mean)**2, axis=0)
    return mean, var

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_univariate_sample_mean_and_variance_constant_parameters(self, dtype):
    """Tests the mean and vol of the univariate GBM sampled paths."""
    mu = 0.05
    sigma = 0.05
    times = np.array([0.1, 0.5, 1.0], dtype=dtype)
    initial_state = 2.0
    mean, var = self.process_sample_paths_mean_and_variance(mu, sigma, times,
                                                            initial_state,
                                                            dtype)
    expected_mean = ((mu - sigma**2 / 2)
                     * np.array(times) + np.log(initial_state))
    expected_var = sigma**2 * times

    with self.subTest("Drift"):
      self.assertAllClose(tf.squeeze(mean), expected_mean, atol=1e-3, rtol=1e-3)
    with self.subTest("Variance"):
      self.assertAllClose(tf.squeeze(var), expected_var, atol=1e-3, rtol=1e-3)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_univariate_sample_mean_and_variance_time_varying_parameters(self,
                                                                       dtype):
    """Tests the mean and vol of the univariate GBM sampled paths."""
    initial_state = 2.0

    with self.subTest("Drift as a step function, sigma = 0.0"):
      mu_times = np.array([0.0, 5.0, 10.0], dtype=dtype)
      mu_values = np.array([0.0, 0.0, 0.05, 0.05], dtype=dtype)
      mu = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=mu_times,
                                                    values=mu_values,
                                                    dtype=dtype)
      sigma = 0.0
      times = np.array([0.0, 1.0, 5.0, 7.0, 10.0], dtype=dtype)
      mean, var = self.process_sample_paths_mean_and_variance(
          mu, sigma, times, initial_state, dtype)
      expected_mean = np.array(
          [
              0.0,  # mu = 0 at t = 0
              0.0,  # mu = 0 for t <= 1.0
              0.0,  # mu = 0 for t < 5.0
              2.0 * 0.05,  # mu = 0.05 for 5.0 < t <= 7.0
              5.0 * 0.05   # mu = 0.05 for 5.0 < t <= 10.0
          ],
          dtype=dtype) + np.log(initial_state)
      expected_var = sigma * np.sqrt(times)  # As sigma is zero.
      self.assertAllClose(tf.squeeze(mean), expected_mean, atol=1e-3, rtol=1e-3)
      self.assertAllClose(tf.squeeze(var), expected_var, atol=1e-3, rtol=1e-3)

    with self.subTest("Drift = 0.05, sigma = step function"):
      mu = 0.05
      sigma_times = np.array([0.0, 5.0, 10.0], dtype=dtype)
      sigma_values = np.array([0.0, 0.2, 0.4, 0.6], dtype=dtype)
      sigma = tff.math.piecewise.PiecewiseConstantFunc(
          jump_locations=sigma_times,
          values=sigma_values,
          dtype=dtype)
      times = np.array([0.0, 1.0, 5.0, 7.0, 10.0], dtype=dtype)
      mean, var = self.process_sample_paths_mean_and_variance(
          mu, sigma, times, initial_state, dtype)
      expected_mean = np.array(
          [
              0.0,  # mu = 0 at t = 0
              1.0 * mu - 0.5 * 1.0 * 0.2**2,  # t = 1.0
              5.0 * mu - 0.5 * 5.0 * 0.2**2,  # t = 5.0
              7.0 * mu - 0.5 * (5.0 * 0.2**2 + 2.0 * 0.4**2),  # t = 7.0
              10.0 * mu - 0.5 * (5.0 * 0.2**2 + 5.0 * 0.4**2)  # t = 10.0
          ],
          dtype=dtype) + np.log(initial_state)
      expected_var = np.array(
          [
              0.0,  # t = 0
              1.0 * 0.2**2,  # t = 1.0
              5.0 * 0.2**2,  # t = 5.0
              5.0 * 0.2**2 + 2.0 * 0.4**2,  # t = 7.0
              5.0 * 0.2**2 + 5.0 * 0.4**2  # t = 10.0
          ],
          dtype=dtype)
      self.assertAllClose(tf.squeeze(mean), expected_mean, atol=1e-3, rtol=1e-3,
                          msg="comparing means")
      self.assertAllClose(tf.squeeze(var), expected_var, atol=1e-3, rtol=1e-3,
                          msg="comparing variances")

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_multivariate_sample_mean_and_variance(self, dtype):
    """Tests the mean and vol of the univariate GBM sampled paths."""
    means = 0.05
    volatilities = [0.1, 0.2]
    corr_matrix = [[1, 0.1], [0.1, 1]]
    process = tff.models.MultivariateGeometricBrownianMotion(
        dim=2, means=means, volatilities=volatilities, corr_matrix=corr_matrix,
        dtype=dtype)
    times = [0.1, 0.5, 1.0]
    initial_state = [1.0, 2.0]
    samples = process.sample_paths(
        times=times, initial_state=initial_state,
        random_type=tff.math.random.RandomType.SOBOL, num_samples=10000)
    log_s = tf.math.log(samples)
    mean = tf.reduce_mean(log_s, axis=0, keepdims=True)
    var = tf.reduce_mean((log_s - mean)**2, axis=0)
    expected_mean = ((process._means - process._vols**2 / 2)
                     * np.array(np.expand_dims(times, -1))
                     + np.log(initial_state))
    expected_var = process._vols**2 * np.array(np.expand_dims(times, -1))
    with self.subTest("Drift"):
      self.assertAllClose(tf.squeeze(mean), expected_mean, atol=1e-3, rtol=1e-3)
    with self.subTest("Variance"):
      self.assertAllClose(tf.squeeze(var), expected_var, atol=1e-3, rtol=1e-3)
    with self.subTest("Correlations"):
      samples = self.evaluate(samples)
      for i in range(len(times)):
        corr = np.corrcoef(samples[:, i, :], rowvar=False)
      self.assertAllClose(corr, corr_matrix, atol=1e-2, rtol=1e-2)

  def test_univariate_xla_compatible(self):
    """Tests that univariate GBM sampling is XLA-compatible."""
    process = tff.models.GeometricBrownianMotion(0.05, 0.5, dtype=tf.float64)
    @tf.function
    def sample_fn():
      return process.sample_paths(
          times=[0.1, 0.5, 1.0], initial_state=2.0, num_samples=10000)
    samples = tf.xla.experimental.compile(sample_fn)[0]
    log_s = tf.math.log(samples)
    mean = tf.reduce_mean(log_s, axis=0)
    expected_mean = ((process._mu - process._sigma**2 / 2)
                     * np.array([0.1, 0.5, 1.0]) + np.log(2.))
    self.assertAllClose(tf.squeeze(mean), expected_mean, atol=1e-2, rtol=1e-2)

  def test_multiivariate_xla_compatible(self):
    """Tests that multiivariate GBM sampling is XLA-compatible."""
    corr_matrix = [[1, 0.1], [0.1, 1]]
    process = tff.models.MultivariateGeometricBrownianMotion(
        dim=2, means=0.05, volatilities=[0.1, 0.2], corr_matrix=corr_matrix,
        dtype=tf.float64)
    times = [0.1, 0.5, 1.0]
    initial_state = [1.0, 2.0]
    @tf.function
    def sample_fn():
      return process.sample_paths(
          times=times, initial_state=initial_state, num_samples=10000)
    samples = tf.xla.experimental.compile(sample_fn)[0]
    log_s = tf.math.log(samples)
    mean = tf.reduce_mean(log_s, axis=0)
    expected_mean = ((process._means - process._vols**2 / 2)
                     * np.array(np.expand_dims(times, -1))
                     + np.log(initial_state))
    self.assertAllClose(mean, expected_mean, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
  tf.test.main()

