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

"""Tests for nomisma_quant_finance.math.interpolation.linear.interpolate."""


from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class LinearInterpolation(tf.test.TestCase, parameterized.TestCase):
  """Tests for methods in linear_interpolation module."""

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_const_extrapolation_default_dtype(
      self, optimize_for_tpu):
    """Tests linear interpolation with const extrapolation."""
    x = [-10.0, -1.0, 1.0, 3.0, 6.0, 7.0, 8.0, 15.0, 18.0, 25.0, 30.0, 35.0]
    x_data = [-1.0, 2.0, 6.0, 8.0, 18.0, 30.0]
    y_data = [10.0, -1.0, -5.0, 7.0, 9.0, 20.0]
    result = self.evaluate(
        tff.math.interpolation.linear.interpolate(
            x, x_data, y_data,
            optimize_for_tpu=optimize_for_tpu))
    with self.subTest('Shape'):
      self.assertAllEqual(result.shape, [12])
    with self.subTest('Values'):
      self.assertAllClose(result,
                          [np.interp(x_coord, x_data, y_data) for x_coord in x],
                          1e-8)
    with self.subTest('Shape'):
      # All above real would be converted to float32.
      self.assertDTypeEqual(result, np.float32)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_const_extrapolation(
      self, optimize_for_tpu):
    """Tests linear interpolation with const extrapolation."""
    x = [-10, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 35]
    x_data = [-1, 2, 6, 8, 18, 30]
    y_data = [10, -1, -5, 7, 9, 20]
    result = self.evaluate(tff.math.interpolation.linear.interpolate(
        x, x_data, y_data,
        optimize_for_tpu=optimize_for_tpu,
        dtype=tf.float32))
    self.assertAllClose(result,
                        [np.interp(x_coord, x_data, y_data) for x_coord in x],
                        1e-8)
    self.assertIsInstance(result[0], np.float32)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_nonconst_extrapolation(
      self, optimize_for_tpu):
    """Tests linear interpolation with nonconst extrapolation."""
    x = [-10, -2, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 31, 35]
    x_data = np.array([-1, 2, 6, 8, 18, 30])
    y_data_as_list = [10, -1, -5, 7, 9, 20]
    y_data = tf.convert_to_tensor(y_data_as_list, dtype=tf.float64)
    left_slope = 2.0
    right_slope = -3.0
    result = self.evaluate(
        tff.math.interpolation.linear.interpolate(
            x,
            x_data,
            y_data,
            left_slope=left_slope,
            right_slope=right_slope,
            optimize_for_tpu=optimize_for_tpu,
            dtype=tf.float64))
    expected_left = 10.0 + left_slope * (np.array([-10.0, -2.0]) - (-1.0))
    expected_right = 20.0 + right_slope * (np.array([31.0, 35.0]) - 30.0)
    expected_middle = [
        np.interp(x_coord, x_data, y_data_as_list) for x_coord in x[2:-2]
    ]
    self.assertAllClose(
        result, np.concatenate([expected_left, expected_middle,
                                expected_right]), 1e-8)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_repeating_values(
      self, optimize_for_tpu):
    """Tests linear interpolation with repeating values in x_data."""
    x = [1.5]
    # Points (1, 1) and (2, 3) should be used to interpolate x=1.5.
    x_data = [0, 1, 1, 2, 2, 3]
    y_data = [0, 0, 1, 2, 3, 3]

    result = self.evaluate(
        tff.math.interpolation.linear.interpolate(
            x, x_data, y_data, optimize_for_tpu=optimize_for_tpu,
            dtype=tf.float32))
    self.assertAllClose(result, [1.5], 1e-8)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_unequal_lengths_xys(
      self, optimize_for_tpu):
    """Tests incompatible `x_data` and `y_data`."""
    x = [1, 2]
    x_data = [-1, 2, 6, 8, 18]
    y_data = [10, -1, -5, 7, 9, 20]
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          tff.math.interpolation.linear.interpolate(
              x, x_data, y_data, validate_args=True,
              optimize_for_tpu=optimize_for_tpu,
              dtype=tf.float64))

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_empty_xys(self, optimize_for_tpu):
    """Tests an error would be thrown if knots are empty."""
    x = [1, 2]
    x_data = []
    y_data = []
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          tff.math.interpolation.linear.interpolate(
              x, x_data, y_data,
              optimize_for_tpu=optimize_for_tpu,
              dtype=tf.float64))

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_const_extrapolation_batching(
      self, optimize_for_tpu):
    """Tests linear interpolation with const extrapolation and batching."""
    x = [[0, 1.5, 4], [4, 5.5, 8]]
    x_data = [[1, 2, 3], [5, 6, 7]]
    y_data = [[0, 2, 4], [1, 2, 3]]
    result = self.evaluate(
        tff.math.interpolation.linear.interpolate(
            x, x_data, y_data,
            optimize_for_tpu=optimize_for_tpu,
            dtype=tf.float32))
    self.assertAllClose(result, np.array([[0, 1, 4], [1, 1.5, 3]]), 1e-8)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_multiple_batching_dimensions(
      self, optimize_for_tpu):
    """Tests linear interpolation with multiple batching dimensions."""
    for dtype in (np.float32, np.float64):
      x = np.array([[[1.5], [3.5]]], dtype=dtype)
      x_data = np.array([[[1, 2], [3, 4]]], dtype=dtype)
      y_data = np.array([[[0, 1], [2, 3]]], dtype=dtype)
      result = self.evaluate(
          tff.math.interpolation.linear.interpolate(
              x, x_data, y_data,
              optimize_for_tpu=optimize_for_tpu))
      self.assertEqual(result.dtype, dtype)
      self.assertAllClose(result, np.array([[[0.5], [2.5]]]), 1e-8)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_non_const_extrapolation_batching(
      self, optimize_for_tpu):
    """Tests linear interpolation with non-const extrapolation and batching."""
    x = [[0, 1.5, 4], [4, 5.5, 8]]
    x_data = [[1, 2, 3], [5, 6, 7]]
    y_data = [[0, 2, 4], [1, 2, 3]]
    left_slope = [[1], [1]]
    right_slope = [[-1], [-1]]

    result = self.evaluate(
        tff.math.interpolation.linear.interpolate(
            x, x_data, y_data, left_slope, right_slope,
            optimize_for_tpu=optimize_for_tpu,
            dtype=tf.float32))
    self.assertAllClose(result, np.array([[-1, 1, 3], [0, 1.5, 2]]), 1e-8)

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_linear_interpolation_x_data_not_increasing(
      self, optimize_for_tpu):
    """Tests linear interpolation when x_data is not increasing."""
    x = [[0, 1.5, 4], [4, 5.5, 8]]
    x_data = [[1, 2, 3], [5, 7, 6]]
    y_data = [[0, 2, 4], [1, 2, 3]]

    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          tff.math.interpolation.linear.interpolate(
              x, x_data, y_data,
              validate_args=True,
              optimize_for_tpu=optimize_for_tpu,
              dtype=tf.float32))

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_valid_gradients(self, optimize_for_tpu):
    """Tests none of the gradients is nan."""

    # In this example, `x[0]` and `x[1]` are both less than or equal to
    # `x_data[0]`. `x[-2]` and `x[-1]` are both greater than or equal to
    # `x_data[-1]`. They are set up this way to test none of the tf.where
    # branches of the implementation have any nan. An unselected nan could still
    # propagate through gradient calculation with the end result being nan.
    x = [[-10.0, -1.0, 1.0, 3.0, 6.0, 7.0], [8.0, 15.0, 18.0, 25.0, 30.0, 35.0]]
    x_data = [[-1.0, 2.0, 6.0], [8.0, 18.0, 30.0]]

    def _value_helper_fn(y_data):
      """A helper function that returns sum of squared interplated values."""

      interpolated_values = tff.math.interpolation.linear.interpolate(
          x, x_data, y_data,
          optimize_for_tpu=optimize_for_tpu,
          dtype=tf.float64)
      return tf.reduce_sum(tf.math.square(interpolated_values))

    y_data = tf.convert_to_tensor([[10.0, -1.0, -5.0], [7.0, 9.0, 20.0]],
                                  dtype=tf.float64)
    if tf.executing_eagerly():
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(y_data)
        value = _value_helper_fn(y_data=y_data)
        gradients = tape.gradient(value, y_data)
    else:
      value = _value_helper_fn(y_data=y_data)
      gradients = tf.gradients(value, y_data)[0]

    gradients = tf.convert_to_tensor(gradients)

    self.assertFalse(self.evaluate(tf.reduce_any(tf.math.is_nan(gradients))))

  @parameterized.named_parameters(
      ('default_interpolation', False),
      ('one_hot_interpolation', True),
  )
  def test_spline_broadcast_batch(self, optimize_for_tpu):
    """Tests batch shape of spline and interpolation are broadcasted."""
    x = np.array([1.0, 1.5, 3.0])
    x_data = np.array([1.1, 2.2, 3.0, 4.0])
    y_data = x_data**2

    x_1 = tf.expand_dims(x, axis=0)
    x_data_2 = tf.expand_dims(x_data, axis=0)
    y_data_3 = tf.expand_dims(y_data, axis=0)

    result_1 = tff.math.interpolation.linear.interpolate(
        x_1, x_data, y_data, optimize_for_tpu=optimize_for_tpu)
    result_2 = tff.math.interpolation.linear.interpolate(
        x, x_data_2, y_data, optimize_for_tpu=optimize_for_tpu)
    result_3 = tff.math.interpolation.linear.interpolate(
        x, x_data, y_data_3, optimize_for_tpu=optimize_for_tpu)
    expected = np.array([[1.21, 2.53, 9.0]])
    with self.subTest('BroadcastData'):
      self.assertAllClose(result_1, expected)
    with self.subTest('BroadcastXData'):
      self.assertAllClose(result_2, expected)
    with self.subTest('BroadcastYData'):
      self.assertAllClose(result_3, expected)

  def test_linear_interpolation_dynamic_number_points(self):
    """Tests linear interpolation with multiple batching dimensions."""
    if tf.executing_eagerly():
      # No dynamic shapes in eager mode
      return
    dtype = np.float64
    x = tf.compat.v1.placeholder(dtype, [1, 2, None])
    x_data = np.array([[[1, 2], [3, 4]]])
    y_data = np.array([[[0, 1], [2, 3]]])
    op = tff.math.interpolation.linear.interpolate(x, x_data, y_data,
                                                   dtype=dtype)
    with self.cached_session() as session:
      results = session.run(
          op, feed_dict={x: [[[1.5, 2.0, 3.0], [3.5, 4.0, 2.0]]]})
    self.assertAllClose(
        results, np.array([[[0.5, 1.0, 1.0], [2.5, 3.0, 2.0]]]), 1e-8)

  def test_linear_interpolation_dynamic_shapes(self):
    """Tests linear interpolation with multiple batching dimensions."""
    dtype = np.float64
    x = [[[1.5, 2.0, 3.0], [3.5, 4.0, 2.0]]]
    x_data = [[1, 2], [3, 4]]
    y_data = [[[0, 1], [2, 3]]]

    # Define function with generic dynamic inputs
    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None], dtype=dtype),
        tf.TensorSpec([None, None], dtype=dtype),
        tf.TensorSpec([None, None, None], dtype=dtype)])
    def interpolate(x, x_data, y_data):
      return tff.math.interpolation.linear.interpolate(x, x_data, y_data)
    results = interpolate(x, x_data, y_data)
    self.assertAllClose(
        results, np.array([[[0.5, 1.0, 1.0], [2.5, 3.0, 2.0]]]), 1e-8)

if __name__ == '__main__':
  tf.test.main()
