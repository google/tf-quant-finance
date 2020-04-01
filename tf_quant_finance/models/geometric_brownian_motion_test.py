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
  def test_drift_and_volatility(self, dtype):
    """Tests that drift and volatility functions."""
    process = tff.models.GeometricBrownianMotion(0.05, 0.5, dtype=dtype)
    drift_fn = process.drift_fn()
    volatility_fn = process.volatility_fn()
    with self.subTest("Drift"):
      drift = drift_fn(0.2, np.array([1., 2., 3.], dtype=dtype))
      expected_drift = np.array([1., 2., 3.]) * 0.05
      self.assertAllClose(drift, expected_drift, atol=1e-8, rtol=1e-8)
    with self.subTest("Volatility"):
      vol = volatility_fn(0.2, np.array([1., 2., 3.], dtype=dtype))
      expected_vol = np.array([[[1.]], [[2.]], [[3.]]]) * 0.5
      self.assertAllClose(vol, expected_vol, atol=1e-8, rtol=1e-8)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_sample_mean_and_variance(self, dtype):
    """Tests the mean and vol of the sampled paths."""
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

  def test_xla_compatible(self):
    """Tests that sampling is XLA-compatible."""
    process = tff.models.GeometricBrownianMotion(0.05, 0.5, dtype=np.float64)

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


if __name__ == "__main__":
  tf.test.main()

