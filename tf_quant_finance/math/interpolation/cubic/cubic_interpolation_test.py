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

  def test_spline_batch(self):
    """Tests batching of four splines."""
    for dtype in (np.float32, np.float64):
      x_data = np.linspace(-11, 12, 24)
      x_data = np.reshape(x_data, [2, 2, 6])
      y_data = 1.0 / (1.0 + x_data * x_data)
      search_args = np.array([[[-10.5, -5.], [-4.5, 1]],
                              [[1.5, 2.], [7.5, 12.]]])

      spline = tff.math.interpolation.cubic.build_spline(
          x_data, y_data, dtype=dtype)
      result = tff.math.interpolation.cubic.interpolate(search_args, spline,
                                                        dtype=dtype)

      expected = np.array([[[0.00900778, 0.02702703],
                            [0.04705774, 1.]],
                           [[0.33135411, 0.2],
                            [0.01756963, 0.00689655]]],
                          dtype=dtype)
      self.assertEqual(result.dtype.as_numpy_dtype, dtype)
      result = self.evaluate(result)
      np.testing.assert_almost_equal(expected, result)

  def test_invalid_interpolate_parameter_shape(self):
    """Tests batch shape of spline and interpolation should be the same."""
    x_data1 = np.linspace(-5.0, 5.0, num=11)
    x_data2 = np.linspace(0.0, 10.0, num=11)
    x_series = np.array([x_data1, x_data2])
    y_data1 = 1.0 / (1.0 + x_data1**2)
    y_data2 = 1.0 / (2.0 + x_data2**2)
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

  def test_invalid_spline_x_points(self):
    """Tests a spline where the x_points are not increasing."""
    x_data = tf.constant([[1.0, 2.0, 1.5, 3.0, 4.0]], dtype=tf.float64)
    y_data = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float64)

    msg = "Failed to detect invalid x_data sequence"
    with self.assertRaises(tf.errors.InvalidArgumentError, msg=msg):
      self.evaluate(
          tff.math.interpolation.cubic.build_spline(
              x_data, y_data, validate_args=True)[2])

  def test_duplicate_x_points(self):
    """Tests a spline where there are x_points of the same value."""
    # Repeated boundary values are allowed
    x_data = np.array([[1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    y_data = np.array([[3.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    spline = tff.math.interpolation.cubic.build_spline(x_data, y_data)
    x_values = np.array([[0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0],
                         [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]])
    interpolated = tff.math.interpolation.cubic.interpolate(x_values, spline)
    expected = np.array([[3.0, 3.0, 1.525, 1.0, 1.925, 2.9, 2.0, 2.0],
                         [1.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0]])
    interpolated = self.evaluate(interpolated)
    np.testing.assert_almost_equal(expected, interpolated)

if __name__ == "__main__":
  tf.test.main()
