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

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

piecewise = tff.math.piecewise


@test_util.run_all_in_graph_and_eager_modes
class Piecewise(parameterized.TestCase, tf.test.TestCase):
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

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64))
  def test_piecewise_constant_value_with_batch(self, dtype):
    """Tests PiecewiseConstantFunc with batching."""
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
    with self.subTest('Dtype'):
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
    expected_value = np.array([[[3.0, 4.0, 4.0, 5.0],
                                [3.0, 4.0, 4.0, 4.0]],
                               [[3.0, 4.0, 5.0, 5.0],
                                [3.0, 4.0, 5.0, 5.0]]])
    with self.subTest('Value'):
      self.assertAllEqual(value, expected_value)

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64),
      ('AutoDtype', None))
  def test_piecewise_constant_value_with_batch_and_repetitions(self, dtype):
    """Tests PiecewiseConstantFunc with batching and repetitive values."""
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
    if dtype is None:
      with self.subTest('Dtype'):
        self.assertEqual(value.dtype, jump_locations.dtype)
    else:
      with self.subTest('Dtype'):
        self.assertEqual(value.dtype, dtype)
    expected_value = np.array([[3., 3., 4., 5., 5., 6.],
                               [-5., 2., 5., 2., 5., 1.]])
    with self.subTest('DtyValue'):
      self.assertAllEqual(value, expected_value)

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64))
  def test_piecewise_constant_integral_with_batch(self, dtype):
    """Tests PiecewiseConstantFunc with batching."""
    x = np.array([[[0.0, 0.1, 2.0, 11.0], [0.0, 2.0, 3.0, 9.0]],
                  [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
    jump_locations = np.array([[[0.1, 10.0], [1.5, 10.0]],
                               [[1.0, 2.0], [5.0, 6.0]]])
    values = tf.constant([[[3, 4, 5], [3, 4, 5]],
                          [[3, 4, 5], [3, 4, 5]]], dtype=dtype)
    piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                     dtype=dtype)
    value = piecewise_func.integrate(x, x + 1.1)
    with self.subTest('Dtype'):
      self.assertEqual(value.dtype.as_numpy_dtype, dtype)
    expected_value = np.array([[[4.3, 4.4, 4.4, 5.5],
                                [3.3, 4.4, 4.4, 4.5]],
                               [[3.4, 4.5, 5.5, 5.5],
                                [3.4, 4.5, 5.5, 5.5]]])
    self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64),
      ('AutoDtype', None))
  def test_invalid_jump_batch_shape(self, dtype):
    """Tests that `jump_locations` and `values` should have the same batch."""
    jump_locations = np.array([[0.1, 10], [2., 10]])
    values = tf.constant([[[3, 4, 5], [3, 4, 5]]], dtype=dtype)
    with self.assertRaises(ValueError):
      piecewise.PiecewiseConstantFunc(jump_locations, values, dtype=dtype)

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64),
      ('AutoDtype', None))
  def test_invalid_value_event_shape(self, dtype):
    """Tests that `values` event shape is `jump_locations` event shape + 1."""
    jump_locations = np.array([[0.1, 10], [2., 10]])
    values = tf.constant([[3, 4, 5, 6], [3, 4, 5, 7]], dtype=dtype)
    with self.assertRaises(ValueError):
      piecewise.PiecewiseConstantFunc(jump_locations, values, dtype=dtype)

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64),
      ('AutoDtype', None))
  def test_matrix_event_shape_no_batch_shape(self, dtype):
    """Tests that `values` event shape is `jump_locations` event shape + 1."""
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
    if dtype is None:
      with self.subTest('Dtype'):
        # Dtype of jump_locations and of  piecewise_func should match
        self.assertEqual(piecewise_func.dtype(), tf.float32)
    else:
      with self.subTest('Dtype'):
        self.assertEqual(dtype, piecewise_func.dtype())
    with self.subTest('Values'):
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)
    with self.subTest('Integrals'):
      self.assertAllClose(integral, expected_integral, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(
      ('SinglePrecision', tf.float32),
      ('DoublePrecision', tf.float64),
      ('AutoDtype', None))
  def test_3d_event_shape_with_batch_shape(self, dtype):
    """Tests that `values` event shape is `jump_locations` event shape + 1."""
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
    with self.subTest('Values'):
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)
    with self.subTest('Integrals'):
      self.assertAllClose(integral, expected_integral, atol=1e-5, rtol=1e-5)

  def test_dynamic_shapes(self):
    """Tests for dynamically shaped inputs."""
    dtype = np.float64
    x = tf.constant([[0, 1, 2, 3], [0.5, 1.5, 2.5, 3.5]], dtype=dtype)
    jump_locations = tf.constant([[0.5, 2], [0.5, 1.5]], dtype=dtype)
    values = tf.constant([[[0, 1, 1.5], [2, 3, 0], [1, 0, 1]],
                          [[0, 0.5, 1], [1, 3, 2], [2, 3, 1]]], dtype=dtype)
    @tf.function(
        input_signature=[tf.TensorSpec([None, None], dtype=dtype),
                         tf.TensorSpec([None, None], dtype=dtype),
                         tf.TensorSpec([None, None, None], dtype=dtype)])
    def fn(x, jump_locations, values):
      piecewise_func = piecewise.PiecewiseConstantFunc(
          jump_locations, values, dtype=dtype)
      value = piecewise_func(x)
      integral = piecewise_func.integrate(x, x + 1)
      return value, integral
    value, integral = fn(x, jump_locations, values)
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
    with self.subTest('Values'):
      self.assertAllClose(value, expected_value, atol=1e-5, rtol=1e-5)
    with self.subTest('Integrals'):
      self.assertAllClose(integral, expected_integral, atol=1e-5, rtol=1e-5)

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
