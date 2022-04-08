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
"""Tests for Sobol sequence generation."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

qmc = tff.math.qmc


@test_util.run_all_in_graph_and_eager_modes
class SobolTest(tf.test.TestCase):

  def test_normal_integral_mean_and_var_correctly_estimated(self):
    n = int(1000)
    # This test is almost identical to the test with the same name found in
    # nomisma_quant_finance/math/random_ops/sobol/sobol_test.py. The only
    # difference is the need to remove point (0, 0) which is sampled by default.
    dtype = tf.float64
    mu_p = tf.constant([-1., 1.], dtype=dtype)
    mu_q = tf.constant([0., 0.], dtype=dtype)
    sigma_p = tf.constant([0.5, 0.5], dtype=dtype)
    sigma_q = tf.constant([1., 1.], dtype=dtype)
    p = tfp.distributions.Normal(loc=mu_p, scale=sigma_p)
    q = tfp.distributions.Normal(loc=mu_q, scale=sigma_q)

    cdf_sample = qmc.sobol_sample(
        2, n + 1, sequence_indices=tf.range(1, n + 1), dtype=dtype)
    q_sample = q.quantile(cdf_sample)

    # Compute E_p[X].
    e_x = tf.reduce_mean(q_sample * p.prob(q_sample) / q.prob(q_sample), 0)

    # Compute E_p[X^2 - E_p[X]^2].
    e_x2 = tf.reduce_mean(
        q_sample**2 * p.prob(q_sample) / q.prob(q_sample) - e_x**2, 0)
    stddev = tf.sqrt(e_x2)

    # Keep the tolerance levels the same as in monte_carlo_test.py.
    with self.subTest('Shape'):
      self.assertEqual(p.batch_shape, e_x.shape)
    with self.subTest('Mean'):
      self.assertAllClose(
          self.evaluate(p.mean()), self.evaluate(e_x), rtol=0.01)
    with self.subTest('Variance'):
      self.assertAllClose(
          self.evaluate(p.stddev()), self.evaluate(stddev), rtol=0.02)

  def test_sobol_sample(self):

    expected = tf.constant([[0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                            [0.50000, 0.50000, 0.50000, 0.50000, 0.50000],
                            [0.25000, 0.75000, 0.75000, 0.75000, 0.25000],
                            [0.75000, 0.25000, 0.25000, 0.25000, 0.75000],
                            [0.12500, 0.62500, 0.37500, 0.12500, 0.12500],
                            [0.62500, 0.12500, 0.87500, 0.62500, 0.62500],
                            [0.37500, 0.37500, 0.62500, 0.87500, 0.37500],
                            [0.87500, 0.87500, 0.12500, 0.37500, 0.87500],
                            [0.06250, 0.93750, 0.56250, 0.31250, 0.68750],
                            [0.56250, 0.43750, 0.06250, 0.81250, 0.18750],
                            [0.31250, 0.18750, 0.31250, 0.56250, 0.93750],
                            [0.81250, 0.68750, 0.81250, 0.06250, 0.43750],
                            [0.18750, 0.31250, 0.93750, 0.43750, 0.56250],
                            [0.68750, 0.81250, 0.43750, 0.93750, 0.06250],
                            [0.43750, 0.56250, 0.18750, 0.68750, 0.81250],
                            [0.93750, 0.06250, 0.68750, 0.18750, 0.31250],
                            [0.03125, 0.53125, 0.90625, 0.96875, 0.96875],
                            [0.53125, 0.03125, 0.40625, 0.46875, 0.46875],
                            [0.28125, 0.28125, 0.15625, 0.21875, 0.71875],
                            [0.78125, 0.78125, 0.65625, 0.71875, 0.21875],
                            [0.15625, 0.15625, 0.53125, 0.84375, 0.84375],
                            [0.65625, 0.65625, 0.03125, 0.34375, 0.34375],
                            [0.40625, 0.90625, 0.28125, 0.09375, 0.59375],
                            [0.90625, 0.40625, 0.78125, 0.59375, 0.09375],
                            [0.09375, 0.46875, 0.46875, 0.65625, 0.28125],
                            [0.59375, 0.96875, 0.96875, 0.15625, 0.78125],
                            [0.34375, 0.71875, 0.71875, 0.40625, 0.03125],
                            [0.84375, 0.21875, 0.21875, 0.90625, 0.53125],
                            [0.21875, 0.84375, 0.09375, 0.53125, 0.40625]],
                           dtype=tf.float32)

    actual = qmc.sobol_sample(5, 29, validate_args=True)

    self.assertAllClose(
        self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
    self.assertEqual(actual.dtype, expected.dtype)

  def test_sobol_sample_with_sequence_indices(self):
    indices = [1, 3, 10, 15, 19, 24, 28]

    expected = tf.constant([[0.50000, 0.50000, 0.50000, 0.50000, 0.50000],
                            [0.75000, 0.25000, 0.25000, 0.25000, 0.75000],
                            [0.31250, 0.18750, 0.31250, 0.56250, 0.93750],
                            [0.93750, 0.06250, 0.68750, 0.18750, 0.31250],
                            [0.78125, 0.78125, 0.65625, 0.71875, 0.21875],
                            [0.09375, 0.46875, 0.46875, 0.65625, 0.28125],
                            [0.21875, 0.84375, 0.09375, 0.53125, 0.40625]],
                           dtype=tf.float32)

    actual = qmc.sobol_sample(
        5,
        29,
        sequence_indices=tf.constant(indices, dtype=tf.int64),
        validate_args=True)

    self.assertAllClose(
        self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
    self.assertEqual(actual.dtype, expected.dtype)

  def test_sobol_sample_with_tent_transform(self):

    expected = tf.constant([[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                            [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                            [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
                            [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
                            [0.25, 0.75, 0.75, 0.25, 0.25, 0.75],
                            [0.75, 0.25, 0.25, 0.75, 0.75, 0.25],
                            [0.75, 0.75, 0.75, 0.25, 0.75, 0.25],
                            [0.25, 0.25, 0.25, 0.75, 0.25, 0.75]],
                           dtype=tf.float32)

    actual = qmc.sobol_sample(
        6, 8, apply_tent_transform=True, validate_args=True)

    self.assertAllClose(
        self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
    self.assertEqual(actual.dtype, expected.dtype)

  def test_sobol_sample_with_dtype(self):

    for dtype in [tf.float32, tf.float64]:
      expected = tf.constant([[0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                              [0.500, 0.500, 0.500, 0.500, 0.500, 0.500],
                              [0.250, 0.750, 0.750, 0.750, 0.250, 0.250],
                              [0.750, 0.250, 0.250, 0.250, 0.750, 0.750],
                              [0.125, 0.625, 0.375, 0.125, 0.125, 0.375],
                              [0.625, 0.125, 0.875, 0.625, 0.625, 0.875],
                              [0.375, 0.375, 0.625, 0.875, 0.375, 0.125],
                              [0.875, 0.875, 0.125, 0.375, 0.875, 0.625]],
                             dtype=dtype)

      actual = qmc.sobol_sample(6, 8, validate_args=True, dtype=dtype)

      self.assertAllClose(
          self.evaluate(actual), self.evaluate(expected), rtol=1e-6)
      self.assertEqual(actual.dtype, expected.dtype)

  def test_sobol_generating_matrices(self):
    dim = 5
    num_results = 31
    num_digits = 5  # ceil(log2(num_results))

    expected = tf.constant(
        [[16, 8, 4, 2, 1], [16, 24, 20, 30, 17], [16, 24, 12, 18, 29],
         [16, 24, 4, 10, 31], [16, 8, 4, 22, 31]],
        dtype=tf.int32)

    actual = qmc.sobol_generating_matrices(
        dim, num_results, num_digits, validate_args=True)

    self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
    self.assertEqual(actual.dtype, expected.dtype)

  def test_sobol_generating_matrices_with_dtype(self):
    dim = 5
    num_results = 31
    num_digits = 5  # ceil(log2(num_results))

    for dtype in [tf.int32, tf.int64]:
      expected = tf.constant(
          [[16, 8, 4, 2, 1], [16, 24, 20, 30, 17], [16, 24, 12, 18, 29],
           [16, 24, 4, 10, 31], [16, 8, 4, 22, 31]],
          dtype=dtype)

      actual = qmc.sobol_generating_matrices(
          dim, num_results, num_digits, validate_args=True, dtype=dtype)

      self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
      self.assertEqual(actual.dtype, dtype)


if __name__ == '__main__':
  tf.test.main()
