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
"""Linear interpolation method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def interpolate(x,
                x_data,
                y_data,
                left_slope=0.0,
                right_slope=0.0,
                dtype=None,
                name=None):
  """Performs linear interpolation for supplied points.

  Given a set of knots whose x- and y- coordinates are in `x_data` and `y_data`,
  this function returns y-values for x-coordinates in `x` via piecewise
  linear interpolation.

  `x_data` must be strictly increasing but `y_data` don't need to be because we
  don't require the function approximated by these knots to be monotonic.

  #### Examples

  ```python
  x = [-10, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 35]
  # `x_data` must be increasing, but `y_data` don't need to be.
  x_data = [-1, 2, 6, 8, 18, 30.0]
  y_data = [10, -1, -5, 7, 9, 20]
  result = interpolate(x, x_data, y_data)

  with tf.Session() as sess:
    print(sess.run(result))
    # [ 10, 10, 2.66666667, -2, -5, 1, 7, 8.4, 9, 15.41666667, 20, 20]
  ```

  Args:
    x: x-coordinates for which we need to get interpolation. A 1-D `Tensor` of
      real dtype.
    x_data: x coordinates. A 1-D `Tensor` of real dtype. Should be sorted in
      increasing order.
    y_data: y coordinates. A 1-D `Tensor` of real dtype. Should have the
      compatible shape as `x_data`.
    left_slope: The slope to use for extrapolation with x-coordinate smaller
      than the min `x_data`. It's a 0-D `Tensor`. If not supplied, the default
      will be 0, meaning constant extrapolation, i.e. extrapolated value will be
      the leftmost `y_data`.
    right_slope: The slope to use for extrapolation with x-coordinate greater
      than the max `x_data`. It's a 0-D `Tensor`. If not supplied, the default
      will be 0, meaning constant extrapolation, i.e. extrapolated value will be
      the rightmost `y_data`.
    dtype: Optional tf.dtype for `x`, x_data`, `y_data`, `left_slope` and
      `right_slope`.  If not specified, the dtype of the inputs will be used.
    name: Python str. The name prefixed to the ops created by this function. If
      not supplied, the default name 'linear_interpolation' is used.

  Returns:
    A 1-D `Tensor` of real dtype corresponding to the x-values in `x`.
  """
  with tf.name_scope(
      name,
      default_name='linear_interpolation',
      values=[x, x_data, y_data, left_slope, right_slope]):
    x = tf.convert_to_tensor(x, dtype=dtype)
    x_data = tf.convert_to_tensor(x_data, dtype=dtype)
    y_data = tf.broadcast_to(
        tf.convert_to_tensor(y_data, dtype=dtype), shape=tf.shape(x_data))
    left_slope = tf.convert_to_tensor(left_slope, dtype=dtype)
    right_slope = tf.convert_to_tensor(right_slope, dtype=dtype)

    # TODO(b/130141692): add batching support.
    x_data_is_rank_1 = tf.assert_rank(x_data, 1)
    with tf.control_dependencies([x_data_is_rank_1]):
      # Get upper bound indices for `x`.
      upper_indices = tf.searchsorted(x_data, x, side='left', out_type=tf.int32)
      x_data_size = tf.shape(x_data)[-1]
      at_min = tf.equal(upper_indices, 0)
      at_max = tf.equal(upper_indices, x_data_size)

      # Create tensors in order to be used by `tf.where`.
      # `values_min` are extrapolated values for x-coordinates less than or
      # equal to `x_data[0]`.
      # `values_max` are extrapolated values for x-coordinates greater than
      # `x_data[-1]`.
      values_min = y_data[0] + left_slope * (
          x - tf.broadcast_to(x_data[0], shape=tf.shape(x)))
      values_max = y_data[-1] + right_slope * (
          x - tf.broadcast_to(x_data[-1], shape=tf.shape(x)))

      # `tf.where` evaluates all branches, need to cap indices to ensure it
      # won't go out of bounds.
      capped_lower_indices = tf.math.maximum(upper_indices - 1, 0)
      capped_upper_indices = tf.math.minimum(upper_indices, x_data_size - 1)
      x_data_lower = tf.gather(x_data, capped_lower_indices)
      x_data_upper = tf.gather(x_data, capped_upper_indices)
      y_data_lower = tf.gather(y_data, capped_lower_indices)
      y_data_upper = tf.gather(y_data, capped_upper_indices)

      # Nan in unselected branches could propagate through gradient calculation,
      # hence we need to clip the values to ensure no nan would occur. In this
      # case we need to ensure there is no division by zero.
      x_data_diff = x_data_upper - x_data_lower
      floor_x_diff = tf.where(at_min | at_max, x_data_diff + 1, x_data_diff)
      interpolated = y_data_lower + (x - x_data_lower) * (
          y_data_upper - y_data_lower) / floor_x_diff

      interpolated = tf.where(at_min, values_min, interpolated)
      interpolated = tf.where(at_max, values_max, interpolated)
      return interpolated
