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

"""Linear interpolation method."""

import numpy as np
import tensorflow.compat.v2 as tf


def interpolate(x,
                x_data,
                y_data,
                left_slope=None,
                right_slope=None,
                validate_args=False,
                optimize_for_tpu=False,
                dtype=None,
                name=None):
  """Performs linear interpolation for supplied points.

  Given a set of knots whose x- and y- coordinates are in `x_data` and `y_data`,
  this function returns y-values for x-coordinates in `x` via piecewise
  linear interpolation.

  `x_data` must be non decreasing, but `y_data` don't need to be because we do
  not require the function approximated by these knots to be monotonic.

  #### Examples

  ```python
  x = [-10, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 35]
  x_data = [-1, 2, 6, 8, 18, 30.0]
  y_data = [10, -1, -5, 7, 9, 20]

  result = linear_interpolation(x, x_data, y_data)
  # [ 10, 10, 2.66666667, -2, -5, 1, 7, 8.4, 9, 15.41666667, 20, 20]
  ```

  Args:
    x: x-coordinates for which we need to get interpolation. A N-D `Tensor` of
      real dtype. First N-1 dimensions represent batching dimensions.
    x_data: x coordinates. A N-D `Tensor` of real dtype. Should be sorted
      in non decreasing order. First N-1 dimensions represent batching
      dimensions.
    y_data: y coordinates. A N-D `Tensor` of real dtype. Should have the
      compatible shape as `x_data`. First N-1 dimensions represent batching
      dimensions.
    left_slope: The slope to use for extrapolation with x-coordinate smaller
      than the min `x_data`. It's a 0-D or N-D `Tensor`.
      Default value: `None`, which maps to `0.0` meaning constant extrapolation,
      i.e. extrapolated value will be the leftmost `y_data`.
    right_slope: The slope to use for extrapolation with x-coordinate greater
      than the max `x_data`. It's a 0-D or N-D `Tensor`.
      Default value: `None` which maps to `0.0` meaning constant extrapolation,
      i.e. extrapolated value will be the rightmost `y_data`.
    validate_args: Python `bool` that indicates whether the function performs
      the check if the shapes of `x_data` and `y_data` are equal and that the
      elements in `x_data` are non decreasing. If this value is set to `False`
      and the elements in `x_data` are not increasing, the result of linear
      interpolation may be wrong.
      Default value: `False`.
    optimize_for_tpu: A Python bool. If `True`, the algorithm uses one-hot
      encoding to lookup indices of `x_values` in `x_data`. This significantly
      improves performance of the algorithm on a TPU device but may slow down
      performance on the CPU.
      Default value: `False`.
    dtype: Optional tf.dtype for `x`, x_data`, `y_data`, `left_slope` and
      `right_slope`.
      Default value: `None` which means that the `dtype` inferred by TensorFlow
      is used.
    name: Python str. The name prefixed to the ops created by this function.
      Default value: `None` which maps to 'linear_interpolation'.

  Returns:
    A N-D `Tensor` of real dtype corresponding to the x-values in `x`.
  """
  name = name or 'linear_interpolation'
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    dtype = dtype or x.dtype
    x_data = tf.convert_to_tensor(x_data, dtype=dtype, name='x_data')
    y_data = tf.convert_to_tensor(y_data, dtype=dtype, name='y_data')
    batch_shape = x.shape.as_list()[:-1]
    if not batch_shape:
      x = tf.expand_dims(x, 0)
      x_data = tf.expand_dims(x_data, 0)
      y_data = tf.expand_dims(y_data, 0)

    if left_slope is None:
      left_slope = tf.constant(0.0, dtype=x.dtype, name='left_slope')
    else:
      left_slope = tf.convert_to_tensor(left_slope, dtype=dtype,
                                        name='left_slope')
    if right_slope is None:
      right_slope = tf.constant(0.0, dtype=x.dtype, name='right_slope')
    else:
      right_slope = tf.convert_to_tensor(right_slope, dtype=dtype,
                                         name='right_slope')
    control_deps = []
    if validate_args:
      # Check that `x_data` elements is non-decreasing
      diffs = x_data[..., 1:] - x_data[..., :-1]
      assertion = tf.compat.v1.debugging.assert_greater_equal(
          diffs,
          tf.zeros_like(diffs),
          message='x_data is not sorted in non-decreasing order.')
      control_deps.append(assertion)
      # Check that the shapes of `x_data` and `y_data` are equal
      control_deps.append(
          tf.compat.v1.assert_equal(tf.shape(x_data), tf.shape(y_data)))

    with tf.control_dependencies(control_deps):
      # Get upper bound indices for `x`.
      upper_indices = tf.searchsorted(x_data, x, side='left', out_type=tf.int32)
      x_data_size = x_data.shape.as_list()[-1]
      at_min = tf.equal(upper_indices, 0)
      at_max = tf.equal(upper_indices, x_data_size)
      # Create tensors in order to be used by `tf.where`.
      # `values_min` are extrapolated values for x-coordinates less than or
      # equal to `x_data[..., 0]`.
      # `values_max` are extrapolated values for x-coordinates greater than
      # `x_data[..., -1]`.

      values_min = tf.expand_dims(y_data[..., 0], -1) + left_slope * (
          x - tf.broadcast_to(
              tf.expand_dims(x_data[..., 0], -1), shape=tf.shape(x)))
      values_max = tf.expand_dims(y_data[..., -1], -1) + right_slope * (
          x - tf.broadcast_to(
              tf.expand_dims(x_data[..., -1], -1), shape=tf.shape(x)))

      # `tf.where` evaluates all branches, need to cap indices to ensure it
      # won't go out of bounds.
      capped_lower_indices = tf.math.maximum(upper_indices - 1, 0)
      capped_upper_indices = tf.math.minimum(upper_indices, x_data_size - 1)
      # Prepare indices for `tf.gather_nd` or `tf.one_hot`
      # TODO(b/156720909): Extract get_slice logic into a common utilities
      # module for cubic and linear interpolation
      if optimize_for_tpu:
        lower_encoding = tf.one_hot(capped_lower_indices, x_data_size,
                                    dtype=dtype)
        upper_encoding = tf.one_hot(capped_upper_indices, x_data_size,
                                    dtype=dtype)
      else:
        index_matrix = _prepare_indices(upper_indices)
        lower_encoding = tf.concat(
            [index_matrix, tf.expand_dims(capped_lower_indices, -1)], -1)

        upper_encoding = tf.concat(
            [index_matrix, tf.expand_dims(capped_upper_indices, -1)], -1)
      def get_slice(x, encoding):
        if optimize_for_tpu:
          return tf.math.reduce_sum(tf.expand_dims(x, axis=-2) * encoding,
                                    axis=-1)
        else:
          return tf.gather_nd(x, encoding)
      x_data_lower = get_slice(x_data, lower_encoding)
      x_data_upper = get_slice(x_data, upper_encoding)
      y_data_lower = get_slice(y_data, lower_encoding)
      y_data_upper = get_slice(y_data, upper_encoding)

      # Nan in unselected branches could propagate through gradient calculation,
      # hence we need to clip the values to ensure no nan would occur. In this
      # case we need to ensure there is no division by zero.
      x_data_diff = x_data_upper - x_data_lower
      floor_x_diff = tf.where(at_min | at_max, x_data_diff + 1, x_data_diff)
      interpolated = y_data_lower + (x - x_data_lower) * (
          y_data_upper - y_data_lower) / floor_x_diff

      interpolated = tf.where(at_min, values_min, interpolated)
      interpolated = tf.where(at_max, values_max, interpolated)
      if batch_shape:
        return interpolated
      else:
        return tf.squeeze(interpolated, 0)


def _prepare_indices(indices):
  """Prepares `tf.searchsorted` output for index argument of `tf.gather_nd`."""
  batch_shape = indices.shape.as_list()[:-1]
  num_points = indices.shape.as_list()[-1]
  batch_shape_reverse = indices.shape.as_list()[:-1]
  batch_shape_reverse.reverse()
  index_matrix = tf.constant(
      np.flip(np.transpose(np.indices(batch_shape_reverse)), -1),
      dtype=indices.dtype)
  batch_rank = len(batch_shape)
  # Broadcast index matrix to the shape of
  # `batch_shape + [num_points] + [batch_rank]`
  broadcasted_shape = batch_shape + [num_points] + [batch_rank]
  index_matrix = tf.expand_dims(index_matrix, -2) + tf.zeros(
      broadcasted_shape, dtype=indices.dtype)
  return index_matrix
