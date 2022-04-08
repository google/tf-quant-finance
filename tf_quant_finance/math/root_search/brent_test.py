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
"""Tests for math.brent."""

import math

import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math.root_search import utils

brentq = tff.math.root_search.brentq


def polynomial5(x):
  """Polynomial function with 5 roots in the [-1, 1] range."""
  return 63 * x**5 - 70 * x**3 + 15 * x + 2


def exp(x):
  """Exponential function which can operate on floats and `Tensor`s."""
  return math.exp(x) if isinstance(x, float) else tf.exp(x)


def cos(x):
  """Cosine function which can operate on floats and `Tensor`s."""
  return math.cos(x) if isinstance(x, float) else tf.cos(x)


def sin(x):
  """Sine function which can operate on floats and `Tensor`s."""
  return math.sin(x) if isinstance(x, float) else tf.sin(x)


class BrentqTest(tf.test.TestCase):

  def _testFindsAllRoots(self,
                         objective_fn,
                         left_bracket,
                         right_bracket,
                         expected_roots,
                         expected_num_iterations,
                         dtype=tf.float64,
                         absolute_root_tolerance=2e-7,
                         relative_root_tolerance=None,
                         function_tolerance=2e-7,
                         assert_check_for_num_iterations=True):
    # expected_roots are pre-calculated as follows:
    # import scipy.optimize as optimize
    # roots = optimize.brentq(objective_fn,
    #                         left_bracket[i],
    #                         right_bracket[i],
    #                         xtol=absolute_root_tolerance,
    #                         rtol=relative_root_tolerance)

    assert len(left_bracket) == len(
        right_bracket), "Brackets have different sizes"

    if relative_root_tolerance is None:
      relative_root_tolerance = utils.default_relative_root_tolerance(
          dtype)

    expected_num_iterations, result = self.evaluate([
        tf.constant(expected_num_iterations, dtype=tf.int32),
        brentq(
            objective_fn,
            tf.constant(left_bracket, dtype=dtype),
            tf.constant(right_bracket, dtype=dtype),
            absolute_root_tolerance=absolute_root_tolerance,
            relative_root_tolerance=relative_root_tolerance,
            function_tolerance=function_tolerance)
    ])

    roots, value_at_roots, num_iterations, converged = result
    zeros = [0.] * len(left_bracket)

    # The output of SciPy and Tensorflow implementation should match for
    # well-behaved functions.
    self.assertAllClose(
        roots,
        expected_roots,
        atol=2 * absolute_root_tolerance,
        rtol=2 * relative_root_tolerance)
    self.assertAllClose(value_at_roots, zeros, atol=10 * function_tolerance)
    if assert_check_for_num_iterations:
      self.assertAllEqual(num_iterations, expected_num_iterations)
    self.assertAllEqual(
        converged,
        [abs(value) <= function_tolerance for value in value_at_roots])

  @test_util.run_in_graph_and_eager_modes
  def testFindsOneRoot(self):
    self._testFindsAllRoots(
        objective_fn=lambda x: 4 * x**2 - 4,
        left_bracket=[1],  # Root
        right_bracket=[0],
        expected_roots=[1],
        expected_num_iterations=[0])

    self._testFindsAllRoots(
        objective_fn=lambda x: x**3 - 4 * x**2 + 3,
        left_bracket=[-1],
        right_bracket=[1],  # Root
        expected_roots=[1],
        expected_num_iterations=[0])

    self._testFindsAllRoots(
        objective_fn=lambda x: x**2 - 7,
        left_bracket=[2],
        right_bracket=[3],
        expected_roots=[2.6457513093775136],
        expected_num_iterations=[4])

    self._testFindsAllRoots(
        objective_fn=polynomial5,
        left_bracket=[-1],
        right_bracket=[1],
        expected_roots=[-0.14823253013216148],
        expected_num_iterations=[6])

    self._testFindsAllRoots(
        objective_fn=lambda x: exp(x) - 2 * x**2,
        left_bracket=[1],
        right_bracket=[2],
        expected_roots=[1.487962064137658],
        expected_num_iterations=[4])

    self._testFindsAllRoots(
        objective_fn=lambda x: exp(-x) - 2 * x**2,
        left_bracket=[0],
        right_bracket=[1],
        expected_roots=[0.5398352897010781],
        expected_num_iterations=[5])

    self._testFindsAllRoots(
        objective_fn=lambda x: x * (1 - cos(x)),
        left_bracket=[-1],
        right_bracket=[1],
        expected_roots=[0.0],
        expected_num_iterations=[1])

    self._testFindsAllRoots(
        objective_fn=lambda x: 1 - x + sin(x),
        left_bracket=[1],
        right_bracket=[2],
        expected_roots=[1.934563210652628],
        expected_num_iterations=[4])

    self._testFindsAllRoots(
        # Flat in the [-0.5, 0.5] range.
        objective_fn=lambda x: 0 if x == 0 else x * exp(-1 / x**2),
        left_bracket=[-10],
        right_bracket=[1],
        expected_roots=[-0.017029902449646958],
        expected_num_iterations=[22],
        # Set the tolerance to zero to match the SciPy implementation.
        function_tolerance=0)

  @test_util.run_in_graph_and_eager_modes
  def testFindsAllRoots(self):
    self._testFindsAllRoots(
        objective_fn=polynomial5,
        left_bracket=[-10, 1],
        right_bracket=[10, -1],
        expected_roots=[-0.14823252898856332, -0.14823253013216148],
        expected_num_iterations=[10, 6])

  @test_util.run_in_graph_and_eager_modes
  def testFindsAllRootsUsingFloat32(self):
    self._testFindsAllRoots(
        objective_fn=polynomial5,
        left_bracket=[-4, 1],
        right_bracket=[3, -1],
        dtype=tf.float32,
        expected_roots=[-0.14823253010472962, -0.14823253013216148],
        expected_num_iterations=[],
        assert_check_for_num_iterations=False)

  @test_util.run_in_graph_and_eager_modes
  def testFindsAllRootsUsingFloat16(self):
    left_bracket = [-2, 1]
    right_bracket = [2, -1]
    expected_num_iterations = [9, 4]

    expected_num_iterations, result = self.evaluate([
        tf.constant(expected_num_iterations, dtype=tf.int32),
        brentq(
            polynomial5, tf.constant(left_bracket, dtype=tf.float16),
            tf.constant(right_bracket, dtype=tf.float16))
    ])

    _, value_at_roots, num_iterations, _ = result

    # Simply check that the objective function is close to the root for the
    # returned estimates. Do not check the estimates themselves.
    # Using float16 may yield root estimates which differ from those returned
    # by the SciPy implementation.
    self.assertAllClose(value_at_roots, [0., 0.], atol=1e-3)
    self.assertAllEqual(num_iterations, expected_num_iterations)

  @test_util.run_in_graph_and_eager_modes
  def testFindsAnyRoots(self):
    objective_fn = lambda x: (63 * x**5 - 70 * x**3 + 15 * x + 2) / 8.

    left_bracket = [-10, 1]
    right_bracket = [10, -1]
    expected_num_iterations = [7, 6]

    expected_num_iterations, result = self.evaluate([
        tf.constant(expected_num_iterations, dtype=tf.int32),
        brentq(
            objective_fn,
            tf.constant(left_bracket, dtype=tf.float64),
            tf.constant(right_bracket, dtype=tf.float64),
            stopping_policy_fn=tf.reduce_any)
    ])

    roots, value_at_roots, num_iterations, _ = result

    expected_roots = [-0.14823253013443427, -0.14823253013443677]

    self.assertNotAllClose(roots[0], expected_roots[0])
    self.assertAllClose(roots[1], expected_roots[1])

    self.assertNotAllClose(value_at_roots[0], 0.)
    self.assertAllClose(value_at_roots[0], objective_fn(roots[0]))
    self.assertAllClose(value_at_roots[1], 0.)

    self.assertAllEqual(num_iterations, expected_num_iterations)

  @test_util.run_in_graph_and_eager_modes
  def testFindsRootForFlatFunction(self):
    # Flat in the [-0.5, 0.5] range.
    objective_fn = lambda x: 0 if x == 0 else x * exp(-1 / x**2)

    left_bracket = [-10]
    right_bracket = [1]
    expected_num_iterations = [13]

    expected_num_iterations, result = self.evaluate([
        tf.constant(expected_num_iterations, dtype=tf.int32),
        brentq(
            objective_fn,
            tf.constant(left_bracket, dtype=tf.float64),
            tf.constant(right_bracket, dtype=tf.float64))
    ])

    _, value_at_roots, num_iterations, _ = result

    # Simply check that the objective function is close to the root for the
    # returned estimate. Do not check the estimate itself.
    # Unlike Brent's original algorithm (and the SciPy implementation), this
    # implementation stops the search as soon as a good enough root estimate is
    # found. As a result, the estimate may significantly differ from the one
    # returned by SciPy for functions which are extremely flat around the root.
    self.assertAllClose(value_at_roots, [0.])
    self.assertAllEqual(num_iterations, expected_num_iterations)

  @test_util.run_in_graph_and_eager_modes
  def testWithNoIteration(self):
    left_bracket = [-10, 1]
    right_bracket = [10, -1]

    first_guess = tf.constant(left_bracket, dtype=tf.float64)
    second_guess = tf.constant(right_bracket, dtype=tf.float64)

    # Skip iteration entirely.
    # Should return a Tensor built from the best guesses in input positions.
    guess, result = self.evaluate([
        tf.constant([-10, -1], dtype=tf.float64),
        brentq(
            polynomial5, first_guess, second_guess, max_iterations=0)
    ])

    self.assertAllEqual(result.estimated_root, guess)

  @test_util.run_in_graph_and_eager_modes
  def testWithValueAtPositionssOfSameSign(self):
    f = lambda x: x**2

    # Should fail: The objective function has the same sign at both positions.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          brentq(
              f,
              tf.constant(-1, dtype=tf.float64),
              tf.constant(1, dtype=tf.float64),
              validate_args=True))

  @test_util.run_in_graph_and_eager_modes
  def testWithInvalidAbsoluteRootTolerance(self):
    f = lambda x: x**3

    # Should fail: Absolute root tolerance is negative.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          brentq(
              f,
              tf.constant(-2, dtype=tf.float64),
              tf.constant(2, dtype=tf.float64),
              absolute_root_tolerance=-2e-7,
              validate_args=True))

  @test_util.run_in_graph_and_eager_modes
  def testWithInvalidRelativeRootTolerance(self):
    f = lambda x: x**3

    # Should fail: Relative root tolerance is negative.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          brentq(
              f,
              tf.constant(-1, dtype=tf.float64),
              tf.constant(1, dtype=tf.float64),
              relative_root_tolerance=-2e-7,
              validate_args=True))

  @test_util.run_in_graph_and_eager_modes
  def testWithInvalidValueTolerance(self):
    f = lambda x: x**3

    # Should fail: Value tolerance is negative.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          brentq(
              f,
              tf.constant(-1, dtype=tf.float64),
              tf.constant(1, dtype=tf.float64),
              function_tolerance=-2e-7,
              validate_args=True))

  @test_util.run_in_graph_and_eager_modes
  def testWithInvalidMaxIterations(self):
    f = lambda x: x**3

    # Should fail: Maximum number of iterations is negative.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          brentq(
              f,
              tf.constant(-1, dtype=tf.float64),
              tf.constant(1, dtype=tf.float64),
              max_iterations=-1,
              validate_args=True))


if __name__ == "__main__":
  tf.test.main()
