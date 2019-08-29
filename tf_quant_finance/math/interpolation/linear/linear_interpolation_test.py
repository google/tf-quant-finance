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
"""Tests for interpolation.linear_interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.math.interpolation import linear
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class LinearInterpolation(tf.test.TestCase):
  """Tests for methods in linear_interpolation module."""

  def test_linear_interpolation_const_extrapolation_default_dtype(self):
    """Tests linear interpolation with const extrapolation."""
    x = [-10.0, -1.0, 1.0, 3.0, 6.0, 7.0, 8.0, 15.0, 18.0, 25.0, 30.0, 35.0]
    x_data = [-1.0, 2.0, 6.0, 8.0, 18.0, 30.0]
    y_data = [10.0, -1.0, -5.0, 7.0, 9.0, 20.0]
    result = self.evaluate(linear.interpolate(x, x_data, y_data))
    self.assertAllClose(result,
                        [np.interp(x_coord, x_data, y_data) for x_coord in x],
                        1e-8)
    # All above real would be converted to float32.
    self.assertIsInstance(result[0], np.float32)

  def test_linear_interpolation_const_extrapolation(self):
    """Tests linear interpolation with const extrapolation."""
    x = [-10, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 35]
    x_data = [-1, 2, 6, 8, 18, 30.0]
    y_data = [10, -1, -5, 7, 9, 20]
    result = self.evaluate(
        linear.interpolate(x, x_data, y_data, dtype=tf.float32))
    self.assertAllClose(result,
                        [np.interp(x_coord, x_data, y_data) for x_coord in x],
                        1e-8)
    self.assertIsInstance(result[0], np.float32)

  def test_linear_interpolation_nonconst_extrapolation(self):
    """Tests linear interpolation with nonconst extrapolation."""
    x = [-10, -2, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 31, 35]
    x_data = np.array([-1, 2, 6, 8, 18, 30.0])
    y_data_as_list = [10, -1, -5, 7, 9, 20]
    y_data = tf.convert_to_tensor(y_data_as_list, dtype=tf.float64)
    left_slope = 2.0
    right_slope = -3.0
    result = self.evaluate(
        linear.interpolate(
            x,
            x_data,
            y_data,
            left_slope=left_slope,
            right_slope=right_slope,
            dtype=tf.float64))
    expected_left = 10.0 + left_slope * (np.array([-10.0, -2.0]) - (-1.0))
    expected_right = 20.0 + right_slope * (np.array([31.0, 35.0]) - 30.0)
    expected_middle = [
        np.interp(x_coord, x_data, y_data_as_list) for x_coord in x[2:-2]
    ]
    self.assertAllClose(
        result, np.concatenate([expected_left, expected_middle,
                                expected_right]), 1e-8)

  def test_linear_interpolation_broadcast_y(self):
    """Tests compatible `x_data` and `y_data`."""
    x = [-10, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 35]
    x_data = [-1, 2, 6, 8, 18]
    y_data = 10
    result = self.evaluate(
        linear.interpolate(x, x_data, y_data, dtype=tf.float64))
    self.assertAllClose(result, np.repeat(10, len(x)), 1e-8)

  def test_linear_interpolation_unequal_lengths_xys(self):
    """Tests incompatible `x_data` and `y_data`."""
    x = [1, 2]
    x_data = [-1, 2, 6, 8, 18]
    y_data = [10, -1, -5, 7, 9, 20]
    with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
      self.evaluate(linear.interpolate(x, x_data, y_data, dtype=tf.float64))


if __name__ == '__main__':
  tf.test.main()
