# Copyright 2021 Google LLC
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
"""Tests for Geometric Brownian Motion Test Utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.models.geometric_brownian_motion import geometric_brownian_motion_test_utils
gmb_utils = geometric_brownian_motion_test_utils


@test_util.run_all_in_graph_and_eager_modes
class GeometricBrownianMotionTestUtilsTest(parameterized.TestCase,
                                           tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_arrays_all_close(self, dtype):
    """Tests the _arrays_all_close() helper function."""
    with self.subTest("Expected pass"):
      a = tf.ones([4, 2, 3], dtype)
      b = a * 0.99
      atol = a * 0.02
      gmb_utils.arrays_all_close(self, a, b, atol)

    with self.subTest("Shape mismatch"):
      a = tf.ones([4, 2, 3], dtype)
      b = tf.ones([4, 1, 3], dtype) * 0.99
      atol = a * 0.02
      with self.assertRaises(ValueError):
        gmb_utils.arrays_all_close(self, a, b, atol)

    with self.subTest("Values not close"):
      a = tf.ones([4, 2, 3], dtype)
      b = tf.ones([4, 2, 3], dtype) * 0.99
      c = tf.tensor_scatter_nd_update(b, [(1, 1, 1)], [0.95])
      atol = a * 0.02
      with self.assertRaises(ValueError):
        gmb_utils.arrays_all_close(self, a, c, atol)

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_generate_sample_paths_shape(self, dtype):
    mu = np.ones((4, 3, 1), dtype=dtype)
    sigma = mu
    times = np.array([1.4, 5.0], dtype=dtype)
    initial_state = np.ones_like(mu, dtype=dtype) * 100.0
    num_samples = 100
    samples = gmb_utils.generate_sample_paths(mu, sigma, times, initial_state,
                                              False, num_samples, dtype)
    samples_shape = samples.shape
    # expected shape = (batch_shape, num_samples, num_times, 1).
    self.assertEqual(samples_shape, (4, 3, num_samples, 2, 1))

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_calculate_mean_and_var_from_sample_paths_zero_sigma(self, dtype):
    atol = 1e-12
    num_samples = 100
    sample_paths = np.ones((num_samples, 2, 1), dtype=dtype)
    (mean, var, std_err_mean, std_err_var) = self.evaluate(
        gmb_utils.calculate_mean_and_variance_from_sample_paths(
            sample_paths, num_samples, dtype))

    expected_mean = np.ones((2), dtype=dtype) * np.log(1.0)
    zeros = np.zeros((2), dtype=dtype)
    self.assertArrayNear(mean, expected_mean, atol, msg="comparing means")
    self.assertArrayNear(var, zeros, atol, msg="comparing vars")
    self.assertArrayNear(std_err_mean, zeros, atol,
                         msg="comparing std error of means")
    self.assertArrayNear(std_err_var, zeros, atol,
                         msg="comparing std error of vars")

  @parameterized.named_parameters(
      {
          "testcase_name": "SinglePrecision",
          "dtype": np.float32,
      }, {
          "testcase_name": "DoublePrecision",
          "dtype": np.float64,
      })
  def test_calculate_mean_and_variance_from_sample_paths(self, dtype):
    atol = 1e-12
    num_samples = 2
    sample_paths = np.array([[[1], [1], [1]],
                             [[2], [2], [2]]], dtype=dtype)
    (mean, var, std_err_mean, std_err_var) = self.evaluate(
        gmb_utils.calculate_mean_and_variance_from_sample_paths(
            sample_paths, num_samples, dtype))

    ones = np.ones((3), dtype=dtype)
    expected_mean = 0.34657359027997264  # (ln(1) + ln(2)) / 2.
    expected_var = 0.12011325347955035  # (ln(2) - expected_mean)**2.
    # Standard error of the mean formula taken from
    # https://en.wikipedia.org/wiki/Standard_error.
    expected_se_mean = 0.24506453586713678  # sqrt(expected_var / num_samples).

    # Standard error of the sample variance (\sigma_{S^2}) is given by:
    # \sigma_{S^2} = S^2.\sqrt(2 / (n-1)).
    # Taken from 'Standard Errors of Mean, Variance, and Standard Deviation
    # Estimators' Ahn, Fessler 2003.
    # (https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf)
    expected_se_var = 0.16986579209153746  # expected_var * sqrt(2).

    self.assertArrayNear(mean, ones * expected_mean, atol,
                         msg="comparing means")
    self.assertArrayNear(var, ones * expected_var, atol,
                         msg="comparing variances")
    self.assertArrayNear(std_err_mean, ones * expected_se_mean, atol,
                         msg="comparing standard error of means")
    self.assertArrayNear(std_err_var, ones * expected_se_var, atol,
                         msg="comparing standard error of variances")

if __name__ == "__main__":
  tf.test.main()
