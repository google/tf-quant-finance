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
  def test_univariate_drift_and_volatility(self, dtype):
    """Tests univariate GBM drift and volatility functions."""
    process = tff.models.GeometricBrownianMotion(0.05, 0.5, dtype=dtype)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    state = np.array([[1.], [2.], [3.]], dtype=dtype)
    with self.subTest("Drift"):
      drift = drift_fn(0.2, state)
      expected_drift = state * 0.05
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest("Volatility"):
      vol = volatility_fn(0.2, state)
      expected_vol = state * 0.5
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
  def test_multivariate_drift_and_volatility(self, dtype):
    """Tests univariate GBM drift and volatility functions."""
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
  def test_univariate_sample_mean_and_variance(self, dtype):
    """Tests the mean and vol of the univariate GBM sampled paths."""
    process = tff.models.GeometricBrownianMotion(0.05, 0.5, dtype=dtype)
    samples = process.sample_paths(
        times=[0.1, 0.5, 1.0], initial_state=2.0,
        random_type=tff.math.random.RandomType.SOBOL, num_samples=10000)
    log_s = tf.math.log(samples)
    mean = tf.reduce_mean(log_s, axis=0, keepdims=True)
    var = tf.reduce_mean((log_s - mean)**2, axis=0)
    expected_mean = ((process._mu - process._sigma**2 / 2)
                     * np.array([0.1, 0.5, 1.0]) + np.log(2.))
    expected_var = process._sigma**2 * np.array([0.1, 0.5, 1.0])
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

