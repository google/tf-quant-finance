# Lint as: python3
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
"""Tests for math.piecewise."""


import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math import piecewise


@test_util.run_all_in_graph_and_eager_modes
class Piecewise(tf.test.TestCase):
  """Tests for methods in piecewise module."""

  def test_find_interval_index_correct_dtype(self):
    """Tests find_interval_index outputs the correct type."""
    result = self.evaluate(piecewise.find_interval_index([1.0], [0.0, 1.0]))
    self.assertIsInstance(result[0], np.int32)

  def test_find_interval_index_one_interval(self):
    """Tests find_interval_index is correct with one half-open interval."""
    result = self.evaluate(piecewise.find_interval_index([1.0], [1.0]))
    self.assertAllEqual(result, [0])

    result = self.evaluate(piecewise.find_interval_index([0.0], [1.0]))
    self.assertAllEqual(result, [-1])

    result = self.evaluate(piecewise.find_interval_index([2.0], [1.0]))
    self.assertAllEqual(result, [0])

  def test_find_interval_index(self):
    """Tests find_interval_index is correct in the general case."""
    interval_lower_xs = [0.25, 0.5, 1.0, 2.0, 3.0]
    query_xs = [0.25, 3.0, 5.0, 0.0, 0.5, 0.8]
    result = piecewise.find_interval_index(query_xs, interval_lower_xs)
    self.assertAllEqual(result, [0, 4, 4, -1, 1, 1])

  def test_find_interval_index_last_interval_is_closed(self):
    """Tests find_interval_index is correct in the general case."""
    result = piecewise.find_interval_index([3.0, 4.0], [2.0, 3.0],
                                           last_interval_is_closed=True)
    self.assertAllEqual(result, [0, 1])

  def test_piecewise_constant_value_no_batch(self):
    """Tests PiecewiseConstantFunc with no batching."""
    for dtype in [np.float32, np.float64]:
      x = np.array([0., 0.1, 2., 11.])
      jump_locations = np.array([0.1, 10], dtype=dtype)
      values = tf.constant([3, 4, 5], dtype=dtype)
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      # Also verifies left-continuity
      value = piecewise_func(x)
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
      expected_value = np.array([3., 3., 4., 5.])
      self.assertAllEqual(value, expected_value)

  def test_piecewise_constant_integral_no_batch(self):
    """Tests PiecewiseConstantFunc with no batching."""
    for dtype in [np.float32, np.float64]:
      x = np.array([-4.1, 0., 1., 1.5, 2., 4.5, 5.5])
      jump_locations = np.array([1, 2, 3, 4, 5], dtype=dtype)
      values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      value = piecewise_func.integrate(x, x + 4.1)
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
      expected_value = np.array([0.41, 1.05, 1.46, 1.66, 1.86, 2.41, 2.46])
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)

  def test_piecewise_constant_value_with_batch(self):
    """Tests PiecewiseConstantFunc with batching."""
    for dtype in [np.float32, np.float64]:
      x = np.array([[[0.0, 0.1, 2.0, 11.0], [0.0, 2.0, 3.0, 9.0]],
                    [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
      jump_locations = np.array([[[0.1, 10.0], [1.5, 10.0]],
                                 [[1.0, 2.0], [5.0, 6.0]]])
      values = tf.constant([[[3, 4, 5], [3, 4, 5]],
                            [[3, 4, 5], [3, 4, 5]]], dtype=dtype)
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      # Also verifies right-continuity
      value = piecewise_func(x, left_continuous=False)
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
      expected_value = np.array([[[3.0, 4.0, 4.0, 5.0],
                                  [3.0, 4.0, 4.0, 4.0]],
                                 [[3.0, 4.0, 5.0, 5.0],
                                  [3.0, 4.0, 5.0, 5.0]]])
      self.assertAllEqual(value, expected_value)

  def test_piecewise_constant_value_with_batch_and_repetitions(self):
    """Tests PiecewiseConstantFunc with batching and repetitive values."""
    for dtype in [np.float32, np.float64]:
      x = tf.constant([[-4.1, 0.1, 1., 2., 10, 11.],
                       [1., 2., 3., 2., 5., 9.]], dtype=dtype)
      jump_locations = tf.constant([[0.1, 0.1, 1., 1., 10., 10.],
                                    [-1., 1.2, 2.2, 2.2, 2.2, 8.]], dtype=dtype)
      values = tf.constant([[3, 3, 4, 5, 5., 2, 6.],
                            [-1, -5, 2, 5, 5., 5., 1.]], dtype=dtype)
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      # Also verifies left-continuity
      value = piecewise_func(x, left_continuous=True)
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
      expected_value = np.array([[3., 3., 4., 5., 5., 6.],
                                 [-5., 2., 5., 2., 5., 1.]])
      self.assertAllEqual(value, expected_value)

  def test_piecewise_constant_integral_with_batch(self):
    """Tests PiecewiseConstantFunc with batching."""
    for dtype in [np.float32, np.float64]:
      x = np.array([[[0.0, 0.1, 2.0, 11.0], [0.0, 2.0, 3.0, 9.0]],
                    [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
      jump_locations = np.array([[[0.1, 10.0], [1.5, 10.0]],
                                 [[1.0, 2.0], [5.0, 6.0]]])
      values = tf.constant([[[3, 4, 5], [3, 4, 5]],
                            [[3, 4, 5], [3, 4, 5]]], dtype=dtype)
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      value = piecewise_func.integrate(x, x + 1.1)
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
      expected_value = np.array([[[4.3, 4.4, 4.4, 5.5],
                                  [3.3, 4.4, 4.4, 4.5]],
                                 [[3.4, 4.5, 5.5, 5.5],
                                  [3.4, 4.5, 5.5, 5.5]]])
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)

  def test_invalid_jump_batch_shape(self):
    """Tests that `jump_locations` and `values` should have the same batch."""
    for dtype in [np.float32, np.float64]:
      jump_locations = np.array([[0.1, 10], [2., 10]])
      values = tf.constant([[[3, 4, 5], [3, 4, 5]]], dtype=dtype)
      with self.assertRaises(ValueError):
        piecewise.PiecewiseConstantFunc(jump_locations, values, dtype=dtype)

  def test_invalid_value_event_shape(self):
    """Tests that `values` event shape is `jump_locations` event shape + 1."""
    for dtype in [np.float32, np.float64]:
      jump_locations = np.array([[0.1, 10], [2., 10]])
      values = tf.constant([[3, 4, 5, 6], [3, 4, 5, 7]], dtype=dtype)
      with self.assertRaises(ValueError):
        piecewise.PiecewiseConstantFunc(jump_locations, values, dtype=dtype)

  def test_matrix_event_shape_no_batch_shape(self):
    """Tests that `values` event shape is `jump_locations` event shape + 1."""
    for dtype in [np.float32, np.float64]:
      x = np.array([0., 0.1, 2., 11.])
      jump_locations = [0.1, 10]
      values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
      piecewise_func = piecewise.PiecewiseConstantFunc(
          jump_locations, values, dtype=dtype)
      value = piecewise_func(x)
      integral = piecewise_func.integrate(x, x + 1)
      expected_value = [[[1, 2], [3, 4]], [[1, 2], [3, 4]],
                        [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
      expected_integral = [[[4.6, 5.6], [6.6, 7.6]],
                           [[5, 6], [7, 8]],
                           [[5, 6], [7, 8]],
                           [[9, 10], [11, 12]]]
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)
      self.assertAllClose(integral, expected_integral, atol=1e-5, rtol=1e-5)

  def test_3d_event_shape_with_batch_shape(self):
    """Tests that `values` event shape is `jump_locations` event shape + 1."""
    for dtype in [np.float32, np.float64]:
      x = np.array([[0, 1, 2, 3], [0.5, 1.5, 2.5, 3.5]])
      jump_locations = [[0.5, 2], [0.5, 1.5]]
      values = [[[0, 1, 1.5], [2, 3, 0], [1, 0, 1]],
                [[0, 0.5, 1], [1, 3, 2], [2, 3, 1]]]
      piecewise_func = piecewise.PiecewiseConstantFunc(
          jump_locations, values, dtype=dtype)
      value = piecewise_func(x)
      integral = piecewise_func.integrate(x, x + 1)
      expected_value = [[[0, 1, 1.5],
                         [2, 3, 0],
                         [2, 3, 0],
                         [1, 0, 1]],
                        [[0, 0.5, 1],
                         [1, 3, 2],
                         [2, 3, 1],
                         [2, 3, 1]]]
      expected_integral = [[[1, 2, 0.75],
                            [2, 3, 0],
                            [1, 0, 1],
                            [1, 0, 1]],
                           [[1, 3, 2],
                            [2, 3, 1],
                            [2, 3, 1],
                            [2, 3, 1]]]
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)
      self.assertAllClose(integral, expected_integral, atol=1e-5, rtol=1e-5)

  def test_invalid_x_batch_shape(self):
    """Tests that `x` should have compatible batch with `jump_locations`."""
    for dtype in [np.float32, np.float64]:
      x = np.array([[0., 0.1, 2., 11.],
                    [0., 0.1, 2., 11.], [0., 0.1, 2., 11.]])
      jump_locations = np.array([[0.1, 10], [2., 10]])
      values = tf.constant([[3, 4, 5], [3, 4, 5]], dtype=dtype)
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      with self.assertRaises(ValueError):
        piecewise_func(x, left_continuous=False)

  def test_incompatible_x1_x2_batch_shape(self):
    """Tests that `x1` and `x2` should have the same batch shape."""
    for dtype in [np.float32, np.float64]:
      x1 = np.array([[0., 0.1, 2., 11.],
                     [0., 0.1, 2., 11.]])
      x2 = np.array([[0., 0.1, 2., 11.],
                     [0., 0.1, 2., 11.], [0., 0.1, 2., 11.]])
      x3 = x2 + 1
      jump_locations = np.array([[0.1, 10]])
      values = tf.constant([[3, 4, 5]], dtype=dtype)
      piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                       dtype=dtype)
      with self.assertRaises(ValueError):
        piecewise_func.integrate(x1, x2)
      # `x2` and `x3` have the same batch shape but different from
      # the batch shape of `jump_locations`
      with self.assertRaises(ValueError):
        piecewise_func.integrate(x2, x3)

  def test_convert_to_tensor_or_func_tensors(self):
    """Tests that tensor_or_func converts inputs into Tensors."""
    dtype = tf.float64
    inputs = [2.0, [1, 2, 3], np.arange(1, 5, 1)]
    output = []
    expected = []
    for i in inputs:
      x = (piecewise.convert_to_tensor_or_func(i, dtype))
      # Check that the returned value is a tensor and is_const flag is set.
      output.append((tf.is_tensor(x[0]), x[1]))
      expected.append((True, True))

    self.assertAllEqual(output, expected)

  def test_convert_to_tensor_or_func_PiecewiseConstantFunc(self):
    """Tests that tensor_or_func recognizes inputs of PiecewiseConstantFunc."""
    dtype = tf.float64
    times = np.arange(0, 10, 1)
    values = np.ones(11)
    pwc = piecewise.PiecewiseConstantFunc(times, values, dtype=dtype)
    output = piecewise.convert_to_tensor_or_func(pwc)
    expected = (pwc, False)
    self.assertAllEqual(output, expected)

if __name__ == '__main__':
  tf.test.main()
