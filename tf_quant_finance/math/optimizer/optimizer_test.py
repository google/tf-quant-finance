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
"""Tests for tff.math.optimizer."""


import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
tff_math = tff.math


@test_util.run_all_in_graph_and_eager_modes
class OptimizerTest(tf.test.TestCase):
  """Tests for optimization algorithms."""

  def test_bfgs_minimize(self):
    """Use BFGS algorithm to find all four minima of Himmelblau's function."""

    @tff_math.make_val_and_grad_fn
    def himmelblau(coord):
      x, y = coord[..., 0], coord[..., 1]
      return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2

    start = tf.constant([[1, 1],
                         [-2, 2],
                         [-1, -1],
                         [1, -2]], dtype='float64')

    results = self.evaluate(tff_math.optimizer.bfgs_minimize(
        himmelblau, initial_position=start,
        stopping_condition=tff_math.optimizer.converged_all,
        tolerance=1e-8))

    expected_minima = np.array([[3, 2],
                                [-2.805118, 3.131312],
                                [-3.779310, -3.283186],
                                [3.584428, -1.848126]])

    self.assertTrue(results.converged.all())
    self.assertEqual(results.position.shape, expected_minima.shape)
    self.assertNDArrayNear(results.position, expected_minima, 1e-5)

  def test_lbfgs_minimize(self):
    """Use L-BFGS algorithm to optimize randomly generated quadratic bowls."""

    np.random.seed(12345)
    dim = 10
    batches = 50
    minima = np.random.randn(batches, dim)
    scales = np.exp(np.random.randn(batches, dim))

    @tff_math.make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(input_tensor=scales * (x - minima) ** 2, axis=-1)

    start = tf.ones((batches, dim), dtype='float64')

    results = self.evaluate(tff_math.optimizer.lbfgs_minimize(
        quadratic, initial_position=start,
        stopping_condition=tff_math.optimizer.converged_any,
        tolerance=1e-8))

    self.assertTrue(results.converged.any())
    self.assertEqual(results.position.shape, minima.shape)
    self.assertNDArrayNear(
        results.position[results.converged], minima[results.converged], 1e-5)

  def test_nelder_mead_minimize(self):
    """Use Nelder Mead algorithm to optimize the Rosenbrock function."""

    def rosenbrock(coord):
      x, y = coord[0], coord[1]
      return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    start = tf.constant([-1.0, 1.0])
    results = self.evaluate(tff_math.optimizer.nelder_mead_minimize(
        rosenbrock,
        initial_vertex=start,
        func_tolerance=1e-12))
    self.assertTrue(results.converged)
    self.assertArrayNear(results.position, [1.0, 1.0], 1e-5)

  def test_differential_evolution(self):
    """Use differential evolution algorithm to minimize a quadratic function."""
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])
    def quadratic(x):
      return tf.reduce_sum(
          scales * tf.math.squared_difference(x, minimum), axis=-1)

    initial_population = tf.random.uniform([40, 2], seed=1243)
    results = self.evaluate(tff_math.optimizer.differential_evolution_minimize(
        quadratic,
        initial_population=initial_population,
        func_tolerance=1e-12,
        seed=2484))
    self.assertTrue(results.converged)

if __name__ == '__main__':
  tf.test.main()
