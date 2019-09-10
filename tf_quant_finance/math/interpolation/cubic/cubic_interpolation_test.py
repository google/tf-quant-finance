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

# Lint as: python2, python3
"""Tests for cubic spline interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class CubicInterpolationTest(tf.test.TestCase):

  def test_error_calc(self):
    """Test that the deviation between the interpolated values and the actual values.

       This should be less than 0.02. This value was derived by running the
       same test with scipy cubic interpolation
    """
    sampling_points = 1000
    spline_x = np.linspace(0.0, 10.0, num=11, dtype=np.float64)
    spline_y = [1.0 / (1.0 + x * x) for x in spline_x]
    x_series = np.array([spline_x])
    y_series = np.array([spline_y])
    spline = tff.math.interpolation.cubic.build_spline(x_series, y_series)

    # There is an error if we go to 10.0
    test_range_x = np.linspace(0.0, 9.99, num=sampling_points, dtype=np.float64)
    search_args = tf.constant(np.array([test_range_x]), dtype=tf.float64)
    projected_y = tff.math.interpolation.cubic.interpolate(search_args, spline)
    expected_y = tf.constant([[1.0 / (1.0 + x * x) for x in test_range_x]],
                             dtype=tf.float64)
    errors = expected_y - projected_y
    deviation = self.evaluate(tfp.stats.stddev(errors[0], sample_axis=0))
    limit = 0.02
    self.assertLess(deviation, limit)

  def test_compare_spline_64(self):
    x_data1 = np.linspace(-5.0, 5.0, num=11, dtype=np.float64)
    x_data2 = np.linspace(0.0, 10.0, num=11, dtype=np.float64)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2])
    search_args = tf.constant([[-1.2, 0.0, 0.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float64)

    spline2 = tff.math.interpolation.cubic.build_spline(x_series, y_series)
    result = tf.reshape(
        tff.math.interpolation.cubic.interpolate(search_args, spline2), [6])

    expected = tf.constant([
        0.401153371166, 1.0, 0.927547412565, 0.144129651521, 0.194406085855,
        0.037037037037
    ],
                           dtype=tf.float64)

    self.assertAllClose(expected, result)

  def test_compare_spline_32(self):
    x_data1 = np.linspace(-5.0, 5.0, num=11, dtype=np.float32)
    x_data2 = np.linspace(0.0, 10.0, num=11, dtype=np.float32)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2], dtype=np.float32)
    search_args = tf.constant([[-1.2, 0.0, 0.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float32)

    spline2 = tff.math.interpolation.cubic.build_spline(x_series, y_series)
    result = tf.reshape(
        tff.math.interpolation.cubic.interpolate(search_args, spline2), [6])

    expected = tf.constant([
        0.401153371166, 1.0, 0.927547412565, 0.144129651521, 0.194406085855,
        0.037037037037
    ],
                           dtype=tf.float32)

    self.assertAllClose(expected, result)

  def test_compare_shape_conformance_of_interpolate(self):
    """Test that the shape of the result of the interpolate method is correct.

    i.e.
    given
    x_points.shape (num_splines, spline_length)
    y_points.shape (num_splines, spline_length)
    x_test.shape (num_splines, num_test_values)

    then interpolate(x_test, spline ) -> shape(num_splines, num_test_values)
    """

    # num splines = 2
    # spline_length = 11
    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2])

    # num_test_values = 3
    search_args = tf.constant([[-1.2, 0.0, 0.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float64)

    x_data_series = tf.stack(x_series, axis=0)
    y_data_series = tf.stack(y_series, axis=0)
    x_test = tf.stack(search_args, axis=0)

    spline = tff.math.interpolation.cubic.build_spline(x_data_series,
                                                       y_data_series)
    predicted = tff.math.interpolation.cubic.interpolate(x_test, spline)

    self.assertAllEqual(tf.shape(x_test), tf.shape(predicted))

    # num_test_values = 13
    search_args11 = tf.constant(
        [[-1.2, 0.0, 0.3, 1.2, 2.1, 0.8, 0.0, 0.3, 1.2, 2.1, 0.8],
         [2.2, 1.8, 5.0, 2.2, 1.8, 5.0, 5.0, 2.2, 1.8, 5.0, 2.2]],
        dtype=tf.float64)

    x_test11 = tf.stack(search_args11, axis=0)
    predicted11 = tff.math.interpolation.cubic.interpolate(x_test11, spline)

    self.assertAllEqual(tf.shape(x_test11), tf.shape(predicted11))

  def test_invalid_interpolate_parameter_shape(self):
    """Test shape(x_points)[0] != shape(test_x)[0]."""

    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2])
    x_data_series = tf.stack(x_series, axis=0)
    y_data_series = tf.stack(y_series, axis=0)

    search_args = tf.constant([[-1.2, 0.0, 0.3]], dtype=tf.float64)
    x_test = tf.stack(search_args, axis=0)
    spline = tff.math.interpolation.cubic.build_spline(x_data_series,
                                                       y_data_series)

    msg = "Failed to catch that the test vector has less rows than x_points"
    with self.assertRaises(ValueError, msg=msg):
      tff.math.interpolation.cubic.interpolate(x_test, spline)

  def test_invalid_interpolate_parameter_value(self):
    """Test where a value to interpolate lies outside the spline points."""

    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2])
    x_data_series = tf.stack(x_series, axis=0)
    y_data_series = tf.stack(y_series, axis=0)

    # num_test_values = 3
    search_args = tf.constant([[-5.2, 0.0, 5.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float64)
    x_test = tf.stack(search_args, axis=0)
    spline = tff.math.interpolation.cubic.build_spline(x_data_series,
                                                       y_data_series)

    msg = ("Failed to catch that the test vector data lies outside of "
           "spline range")

    with self.assertRaises(tf.errors.InvalidArgumentError, msg=msg) as cm:
      self.evaluate(
          tff.math.interpolation.cubic.interpolate(
              x_test, spline, validate_args=True))
    print(cm.exception)

  def test_invalid_spline_x_points(self):
    """Test a spline where the x_points are not strictly increasing."""
    x_data = tf.constant([[1.0, 2.0, 1.5, 3.0, 4.0]], dtype=tf.float64)
    y_data = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float64)

    msg = "Failed to detect invalid x_data sequence"
    with self.assertRaises(tf.errors.InvalidArgumentError, msg=msg) as cm:
      self.evaluate(
          tff.math.interpolation.cubic.build_spline(
              x_data, y_data, validate_args=True)[2])
    print(cm.exception)

  def test_duplicate_x_points(self):
    """Test a spline where there are x_points of the same value."""
    x_data = tf.constant([[1.0, 2.0, 2.0, 3.0, 4.0]], dtype=tf.float64)
    y_data = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float64)

    msg = "Failed to detect duplicate x_data points"
    with self.assertRaises(tf.errors.InvalidArgumentError, msg=msg) as cm:
      self.evaluate(
          tff.math.interpolation.cubic.build_spline(
              x_data, y_data, validate_args=True)[2])
    print(cm.exception)

  def test_validate_args_build(self):
    """Test that validation works as intended."""
    x_data = tf.constant([[1.0, 2.0, 2.0, 3.0, 4.0]], dtype=tf.float64)
    y_data = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float64)

    # this should not fail
    self.evaluate(
        tff.math.interpolation.cubic.build_spline(
            x_data, y_data, validate_args=False)[2])

  def test_validate_args_interpolate(self):
    """Test that validation can be turned off in the interpolate call."""
    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2])
    x_data_series = tf.stack(x_series, axis=0)
    y_data_series = tf.stack(y_series, axis=0)

    # num_test_values = 3
    search_args = tf.constant([[-5.2, 0.0, 5.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float64)
    x_test = tf.stack(search_args, axis=0)
    spline = tff.math.interpolation.cubic.build_spline(x_data_series,
                                                       y_data_series)

    # this should not fail with a validation error but a separate error
    # thrown by gather_nd
    msg = "The error should be an invalid argument"
    with self.assertRaises(tf.errors.InvalidArgumentError, msg=msg) as cm:
      self.evaluate(
          tff.math.interpolation.cubic.interpolate(
              x_test, spline, validate_args=False))
      print(cm.exception)

  def test_build_and_interpolate(self):
    """Test a combined call by just calling interpolate."""
    # num splines = 2
    # spline_length = 11
    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2])

    # num_test_values = 3
    search_args = tf.constant([[-1.2, 0.0, 0.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float64)

    x_data_series = tf.stack(x_series, axis=0)
    y_data_series = tf.stack(y_series, axis=0)
    x_test = tf.stack(search_args, axis=0)

    spline = tff.math.interpolation.cubic.SplineParameters(
        x_data_series, y_data_series, None)

    predicted = tff.math.interpolation.cubic.interpolate(x_test, spline)

    self.assertAllEqual(tf.shape(x_test), tf.shape(predicted))

  def test_dtype_conversion_float32_64(self):
    """Test specifying float32 data but requiring conversion to float64."""
    x_data1 = np.linspace(-5.0, 5.0, num=11, dtype=np.float32)
    x_data2 = np.linspace(0.0, 10.0, num=11, dtype=np.float32)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2], dtype=np.float32)
    search_args = tf.constant([[-1.2, 0.0, 0.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float32)

    spline2 = tff.math.interpolation.cubic.build_spline(
        x_series, y_series, dtype=tf.float64)

    msg = "Tensor conversion float32 to float64  should fail here"
    with self.assertRaises(ValueError, msg=msg) as cm:
      self.evaluate(
          tff.math.interpolation.cubic.interpolate(
              search_args, spline2, dtype=tf.float64))
      print(cm)

  def test_dtype_conversion_float64_32(self):
    """Test specifying float64 data but requiring conversion to float32."""
    x_data1 = np.linspace(-5.0, 5.0, num=11, dtype=np.float64)
    x_data2 = np.linspace(0.0, 10.0, num=11, dtype=np.float64)
    x_series = np.array([x_data1, x_data2])
    y_data1 = [1.0 / (1.0 + x * x) for x in x_data1]
    y_data2 = [1.0 / (2.0 + x * x) for x in x_data2]
    y_series = np.array([y_data1, y_data2], dtype=np.float64)
    search_args = tf.constant([[-1.2, 0.0, 0.3], [2.2, 1.8, 5.0]],
                              dtype=tf.float64)

    spline2 = tff.math.interpolation.cubic.build_spline(
        x_series, y_series, dtype=tf.float32)
    msg = "Tensor conversion from float64 to float32 should fail here"
    with self.assertRaises(ValueError, msg=msg) as cm:
      self.evaluate(
          tff.math.interpolation.cubic.interpolate(
              search_args, spline2, dtype=tf.float32))
      print(cm)


if __name__ == "__main__":
  tf.test.main()
