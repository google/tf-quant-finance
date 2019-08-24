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

"""Kernel for linear interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_quant_finance.math.interpolation.linear import gen_linear_interpolation


# TODO: Arg validation needed.
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
    left_slope: Scalar `Tensor` of real dtype. The slope to use for
      extrapolation with x-coordinate smaller. If not supplied, the default will
      be 0, meaning constant extrapolation, i.e. extrapolated value will be the
      leftmost `y_data`.
      Default value: 0.0
    right_slope: Scalar `Tensor` of real dtype. The slope to use for
      extrapolation with x-coordinate greater than the max `x_data`. If not
      supplied, the default will be 0, meaning constant extrapolation, i.e.
      extrapolated value will be the rightmost `y_data`.
      Default value: 0.0.
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
    # TODO: Add batching support.

    x = tf.convert_to_tensor(x, dtype=dtype)
    x_data = tf.convert_to_tensor(x_data, dtype=dtype)
    # The kernel requires 2D tensors.
    broadcasted = False
    if len(x.shape) != 2:
      broadcasted = True
      x = tf.expand_dims(x, axis=0)

    if len(x_data.shape) != 2:
      x_data = tf.expand_dims(x_data, axis=0)
    y_data = tf.broadcast_to(
        tf.convert_to_tensor(y_data, dtype=dtype), shape=tf.shape(x_data))

    left_slope = tf.broadcast_to(
        tf.convert_to_tensor(left_slope, dtype=x_data.dtype),
        tf.shape(x)[:1])
    right_slope = tf.broadcast_to(
        tf.convert_to_tensor(right_slope, dtype=x_data.dtype),
        tf.shape(x)[:1])

    y_interp = gen_linear_interpolation.linear_interpolation(
        x, x_data, y_data, left_slope, right_slope)
    if broadcasted:
      return tf.squeeze(y_interp, axis=0)
    return y_interp
