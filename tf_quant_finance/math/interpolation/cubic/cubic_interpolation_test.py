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
"""Tests for cubic spline interpolation."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math.interpolation import cubic


@test_util.run_all_in_graph_and_eager_modes
class CubicInterpolationTest(tf.test.TestCase, parameterized.TestCase):
  optimized_for_tpu_and_boundary_condition_type_test_cases = [
      {
          "testcase_name": "default_interpolation_clamped",
          "optimize_for_tpu": False,
          "boundary_condition_type": cubic.BoundaryConditionType.CLAMPED,
      },
      {
          "testcase_name": "one_hot_interpolation_clamped",
          "optimize_for_tpu": True,
          "boundary_condition_type": cubic.BoundaryConditionType.CLAMPED,
      },
      {
          "testcase_name":
              "default_interpolation_fixed_first_derivative",
          "optimize_for_tpu":
              False,
          "boundary_condition_type":
              cubic.BoundaryConditionType.FIXED_FIRST_DERIVATIVE,
      },
      {
          "testcase_name":
              "one_hot_interpolation_fixed_first_derivative",
          "optimize_for_tpu":
              True,
          "boundary_condition_type":
              cubic.BoundaryConditionType.FIXED_FIRST_DERIVATIVE,
      },
  ]

  def build_spline(self, x_series, y_series, boundary_condition_type, d_type,
                   expected_left_boundary_derivative,
                   expected_right_boundary_derivative):
    if boundary_condition_type == cubic.BoundaryConditionType.FIXED_FIRST_DERIVATIVE:
      return tff.math.interpolation.cubic.build_spline(
          x_series,
          y_series,
          boundary_condition_type,
          expected_left_boundary_derivative,
          expected_right_boundary_derivative,
          dtype=d_type)
    else:
      return tff.math.interpolation.cubic.build_spline(
          x_series, y_series, boundary_condition_type, dtype=d_type)

  @parameterized.named_parameters(
      optimized_for_tpu_and_boundary_condition_type_test_cases)
  def test_error_calc_for_derivative(self, optimize_for_tpu,
                                     boundary_condition_type):
    """Test the deviation of the interpolated values from the actual."""
    spline_x = np.linspace(0.0, 10.0, num=11, dtype=np.float64)
    spline_y = [1.0 / (1.0 + x * x) for x in spline_x]
    x_series = np.array([spline_x])
    y_series = np.array([spline_y])
    expected_left_boundary_derivative = 12.0 * x_series[..., -1:]
    expected_right_boundary_derivative = 34.0 * x_series[..., -1:]
    spline = self.build_spline(x_series, y_series, boundary_condition_type,
                               np.float64, expected_left_boundary_derivative,
                               expected_right_boundary_derivative)

    delta = 0.001
    test_x = tf.constant(
        np.array([[0.0, 0.0 + delta, 10.0 - delta, 10.0]]), dtype=tf.float64)
    interpolated_y = tff.math.interpolation.cubic.interpolate(
        test_x, spline, optimize_for_tpu=optimize_for_tpu)
    actual_left_boundary_derivative = (interpolated_y[0][1] -
                                       interpolated_y[0][0]) / (
                                           test_x[0][1] - test_x[0][0])
    actual_right_boundary_derivative = (interpolated_y[0][3] -
                                        interpolated_y[0][2]) / (
                                            test_x[0][3] - test_x[0][2])
    if boundary_condition_type == cubic.BoundaryConditionType.CLAMPED:
      expected_left_boundary_derivative = 0.0
      expected_right_boundary_derivative = 0.0
    actual_left_boundary_derivative = self.evaluate(
        actual_left_boundary_derivative)
    actual_right_boundary_derivative = self.evaluate(
        actual_right_boundary_derivative)
    deviation_left_boundary = abs(actual_left_boundary_derivative -
                                  expected_left_boundary_derivative)
    deviation_right_boundary = abs(actual_right_boundary_derivative -
                                   expected_right_boundary_derivative)
    if expected_left_boundary_derivative != 0:
      deviation_left_boundary = (
          deviation_left_boundary / expected_left_boundary_derivative)
    if expected_right_boundary_derivative != 0:
      deviation_right_boundary = (
          deviation_right_boundary / expected_right_boundary_derivative)
    limit = 0.02
    self.assertLess(deviation_left_boundary, limit)
    self.assertLess(deviation_right_boundary, limit)

  @parameterized.named_parameters(
      optimized_for_tpu_and_boundary_condition_type_test_cases)
  def test_duplicate_x_points_for_derivative(self, optimize_for_tpu,
                                             boundary_condition_type):
    """Test the deviation of the interpolated values from the actual."""
    x_series = np.array([[1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]])
    y_series = np.array([[3.0, 1.0, 5.0, 2.0, 2.0, 2.0, 2.0]])
    expected_left_boundary_derivative = 12.0 * x_series[..., -1:]
    expected_right_boundary_derivative = 34.0 * x_series[..., -1:]
    spline = self.build_spline(x_series, y_series, boundary_condition_type,
                               np.float64, expected_left_boundary_derivative,
                               expected_right_boundary_derivative)

    delta = 0.001
    test_x = tf.constant(
        np.array([[1.0, 1.0 + delta, 4.0 - delta, 4.0]]), dtype=tf.float64)
    interpolated_y = tff.math.interpolation.cubic.interpolate(
        test_x, spline, optimize_for_tpu=optimize_for_tpu)
    actual_left_boundary_derivative = (interpolated_y[0][1] -
                                       interpolated_y[0][0]) / (
                                           test_x[0][1] - test_x[0][0])
    actual_right_boundary_derivative = (interpolated_y[0][3] -
                                        interpolated_y[0][2]) / (
                                            test_x[0][3] - test_x[0][2])
    if boundary_condition_type == cubic.BoundaryConditionType.CLAMPED:
      expected_left_boundary_derivative = 0.0
      expected_right_boundary_derivative = 0.0
    actual_left_boundary_derivative = self.evaluate(
        actual_left_boundary_derivative)
    actual_right_boundary_derivative = self.evaluate(
        actual_right_boundary_derivative)
    deviation_left_boundary = abs(actual_left_boundary_derivative -
                                  expected_left_boundary_derivative)
    deviation_right_boundary = abs(actual_right_boundary_derivative -
                                   expected_right_boundary_derivative)
    if expected_left_boundary_derivative != 0:
      deviation_left_boundary = (
          deviation_left_boundary / expected_left_boundary_derivative)
    if expected_right_boundary_derivative != 0:
      deviation_right_boundary = (
          deviation_right_boundary / expected_right_boundary_derivative)
    limit = 0.02
    self.assertLess(deviation_left_boundary, limit)
    self.assertLess(deviation_right_boundary, limit)

  @parameterized.named_parameters(
      optimized_for_tpu_and_boundary_condition_type_test_cases)
  def test_duplicate_x_points_batching(self, optimize_for_tpu,
                                       boundary_condition_type):
    """Tests a spline where there are x_points of the same value."""
    # Repeated right boundary values are allowed
    x_data = np.array([[1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    y_data = np.array([[3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    expected_left_boundary_derivative = 12.0 * x_data[..., -1:]
    expected_right_boundary_derivative = 34.0 * x_data[..., -1:]
    spline = self.build_spline(x_data, y_data, boundary_condition_type,
                               np.float64, expected_left_boundary_derivative,
                               expected_right_boundary_derivative)

    x_values = np.array([[0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0],
                         [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]])
    interpolated = tff.math.interpolation.cubic.interpolate(
        x_values, spline, optimize_for_tpu=optimize_for_tpu)
    expected = np.array(
        [[3.0, 3.0, 2.025, 1.0, 1.875, 2.6, 2.0, 2.0],
         [1.0, 1.0, 1.34134615, 2.0, 2.54326923, 3.48557692, 4.0, 5.0]])
    if boundary_condition_type == cubic.BoundaryConditionType.FIXED_FIRST_DERIVATIVE:
      expected = np.array(
          [[3.0, 3.0, 8.49166667, 1.0, 5.54166667, -18.53333333, 2.0, 2.0],
           [1.0, 1.0, 14.69294872, 2.0, -1.21474359, 5.16602564, 4.0, 5.0]])
    interpolated = self.evaluate(interpolated)
    np.testing.assert_almost_equal(expected, interpolated)

  @parameterized.named_parameters(
      ("default_interpolation", False),
      ("one_hot_interpolation", True),
  )
  def test_error_calc(self, optimize_for_tpu):
    """Test the deviation of the interpolated values from the actual."""
    sampling_points = 1000
    spline_x = np.linspace(0.0, 10.0, num=11, dtype=np.float64)
    spline_y = [1.0 / (1.0 + x * x) for x in spline_x]
    x_series = np.array([spline_x])
    y_series = np.array([spline_y])
    spline = tff.math.interpolation.cubic.build_spline(x_series, y_series)

    # There is an error if we go to 10.0
    test_range_x = np.linspace(0.0, 9.99, num=sampling_points, dtype=np.float64)
    search_args = tf.constant(np.array([test_range_x]), dtype=tf.float64)
    projected_y = tff.math.interpolation.cubic.interpolate(
        search_args, spline, optimize_for_tpu=optimize_for_tpu)
    expected_y = tf.constant([[1.0 / (1.0 + x * x) for x in test_range_x]],
                             dtype=tf.float64)
    errors = expected_y - projected_y
    deviation = self.evaluate(tfp.stats.stddev(errors[0], sample_axis=0))
    limit = 0.02
    self.assertLess(deviation, limit)

  @parameterized.named_parameters(
      ("default_interpolation", False),
      ("one_hot_interpolation", True),
  )
  def test_spline_batch(self, optimize_for_tpu):
    """Tests batching of four splines."""
    for dtype in (np.float32, np.float64):
      x_data = np.linspace(-11, 12, 24)
      x_data = np.reshape(x_data, [2, 2, 6])
      y_data = 1.0 / (1.0 + x_data * x_data)
      search_args = np.array([[[-10.5, -5.], [-4.5, 1]], [[1.5, 2.], [7.5,
                                                                      12.]]])

      spline = tff.math.interpolation.cubic.build_spline(
          x_data, y_data, dtype=dtype)
      result = tff.math.interpolation.cubic.interpolate(
          search_args, spline, optimize_for_tpu=optimize_for_tpu, dtype=dtype)

      expected = np.array([[[0.00900778, 0.02702703], [0.04705774, 1.]],
                           [[0.33135411, 0.2], [0.01756963, 0.00689655]]],
                          dtype=dtype)
      self.assertEqual(result.dtype.as_numpy_dtype, dtype)
      result = self.evaluate(result)
      np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(
      ("default_interpolation", False),
      ("one_hot_interpolation", True),
  )
  def test_spline_broadcast_batch(self, optimize_for_tpu):
    """Tests batch shape of spline and interpolation are broadcasted."""
    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_data = np.array([x_data1, x_data2])
    y_data = 1.0 / (2.0 + x_data**2)
    x_data = tf.stack(x_data, axis=0)
    dtype = np.float64
    x_value_1 = tf.constant([[[-1.2, 0.0, 0.3]]], dtype=dtype)
    x_value_2 = tf.constant([-1.2, 0.0, 0.3], dtype=dtype)
    spline = tff.math.interpolation.cubic.build_spline(x_data, y_data)

    result_1 = tff.math.interpolation.cubic.interpolate(
        x_value_1, spline, optimize_for_tpu=optimize_for_tpu, dtype=dtype)
    result_2 = tff.math.interpolation.cubic.interpolate(
        x_value_2, spline, optimize_for_tpu=optimize_for_tpu, dtype=dtype)
    expected_1 = np.array(
        [[[0.29131469, 0.5, 0.4779499], [0.5, 0.5, 0.45159077]]], dtype=dtype)
    expected_2 = np.array(
        [[0.29131469, 0.5, 0.4779499], [0.5, 0.5, 0.45159077]], dtype=dtype)
    with self.subTest("BroadcastData"):
      self.assertAllClose(result_1, expected_1)
    with self.subTest("BroadcastValues"):
      self.assertAllClose(result_2, expected_2)

  def test_invalid_spline_x_points(self):
    """Tests a spline where the x_points are not increasing."""
    x_data = tf.constant([[1.0, 2.0, 1.5, 3.0, 4.0]], dtype=tf.float64)
    y_data = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float64)

    msg = "Failed to detect invalid x_data sequence"
    with self.assertRaises(tf.errors.InvalidArgumentError, msg=msg):
      self.evaluate(
          tff.math.interpolation.cubic.build_spline(
              x_data, y_data, validate_args=True))

  @parameterized.named_parameters(
      ("default_interpolation", False),
      ("one_hot_interpolation", True),
  )
  def test_duplicate_x_points(self, optimize_for_tpu):
    """Tests a spline where there are x_points of the same value."""
    # Repeated boundary values are allowed
    x_data = np.array([[1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    y_data = np.array([[3.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    spline = tff.math.interpolation.cubic.build_spline(x_data, y_data)
    x_values = np.array([[0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0],
                         [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]])
    interpolated = tff.math.interpolation.cubic.interpolate(
        x_values, spline, optimize_for_tpu=optimize_for_tpu)
    expected = np.array([[3.0, 3.0, 1.525, 1.0, 1.925, 2.9, 2.0, 2.0],
                         [1.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]])
    interpolated = self.evaluate(interpolated)
    np.testing.assert_almost_equal(expected, interpolated)

  def test_linear_interpolation_dynamic_number_points(self):
    """Tests linear interpolation with multiple batching dimensions."""
    if tf.executing_eagerly():
      # No dynamic shapes in eager mode
      return
    dtype = np.float64
    x = tf.compat.v1.placeholder(dtype, [1, 2, None])
    x_data = np.array([[[1, 2], [3, 4]]])
    y_data = np.array([[[0, 1], [2, 3]]])
    spline = tff.math.interpolation.cubic.build_spline(
        x_data, y_data, dtype=dtype)
    op = tff.math.interpolation.cubic.interpolate(x, spline, dtype=dtype)
    with self.cached_session() as session:
      results = session.run(
          op, feed_dict={x: [[[1.5, 2.0, 3.0], [3.5, 4.0, 2.0]]]})
    self.assertAllClose(results, np.array([[[0.5, 1.0, 1.0], [2.5, 3.0, 2.0]]]),
                        1e-8)

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
      spline = tff.math.interpolation.cubic.build_spline(
          x_data, y_data, dtype=dtype)
      return tff.math.interpolation.cubic.interpolate(x, spline)
    results = interpolate(x, x_data, y_data)
    self.assertAllClose(
        results, np.array([[[0.5, 1.0, 1.0], [2.5, 3.0, 2.0]]]), 1e-8)

if __name__ == "__main__":
  tf.test.main()
