# Lint as: python3
# Copyright 2020 Google LLC
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

"""Interpolation functions in a 2-dimensional space."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.interpolation import cubic
from tf_quant_finance.math.interpolation import linear


class Interpolation2D:
  """Performs interpolation in a 2-dimensional space.

  For input `x_data` in x-direction we assume that values in y-direction are
  given by `y_data` and the corresponsing function values by `z_data`.
  For given `x_values` and `y_values` along x- and y- direction respectively,
  the interpolated function values are computed on grid `[x_values, y_values]`.
  The interpolation is first performed along y-direction for every `x_data`
  point and all `y_values` using 1-d cubic spline interpolation. Next, for
  each interpolated `y_value` point, the function values are interpolated along
  x-direction for `x_values` using 1-d cubic spline interpolation.
  Constant extrapolation is used for the linear interpolation and natural
  boundary conditions are used for the cubic spline.

  ### Example. Volatility surface interpolation

  ```python
  dtype = np.float64
  times = tf.constant([2., 2.5, 3, 4.5], dtype=dtype)
  strikes = tf.constant([16, 22, 35], dtype=dtype)

  times_data = tf.constant([1.5, 2.5, 3.5, 4.5, 5.5], dtype=dtype)
  # Corresponding squared volatility values
  sigma_square_data = tf.constant(
      [[0.15, 0.25, 0.35, 0.4, 0.45, 0.4],
       [0.2, 0.35, 0.55, 0.45, 0.4, 0.6],
       [0.3, 0.45, 0.25, 0.4, 0.5, 0.65],
       [0.25, 0.25, 0.45, 0.25, 0.5, 0.55],
       [0.35, 0.35, 0.25, 0.4, 0.55, 0.65]], dtype=dtype)
  # Interpolation is done for the total variance
  total_variance = tf.expand_dims(times_data, -1) * sigma_square_data
  # Corresponding strike values. Notice we need to broadcast to the shape of
  # `sigma_square_data`
  strike_data = tf.broadcast_to(
      tf.constant([15, 25, 35, 40, 50, 55], dtype=dtype), [5, 6])
  # Interpolate total variance on for coordinates `(times, strikes)`
  interpolator = Interpolation2D(times_data, strike_data, total_variance,
                                 dtype=dtype)
  interpolated_values = interpolator.interpolate(times, strikes)
  ```
  """

  def __init__(self,
               x_data,
               y_data,
               z_data,
               dtype=None,
               name=None):
    """Initialize the 2d-interpolation object.

    Args:
      x_data: A `Tensor` of real `dtype` and shape
        `batch_shape + [num_x_data_points]`.
        Defines the x-coordinates of the input data. `num_x_data_points` should
        be >= 2. The elements of `x_data` should be in a non-decreasing order.
      y_data: A `Tensor` of the same `dtype` as `x_data` and shape
        `batch_shape + [num_x_data_points, num_y_data_points]`. Defines the
        y-coordinates of the input data. `num_y_data_points` should be >= 2.
        The elements of `y_data` should be in a non-decreasing order along last
        dimension.
      z_data: A `Tensor` of the same shape and `dtype` as `y_data`. Defines the
        z-coordinates of the input data (i.e., the function values).
      dtype: Optional dtype for the input `Tensor`s.
        Default value: `None` which maps to the default dtype inferred by
        TensorFlow.
      name: Python `str` name prefixed to ops created by this class.
        Default value: `None` which is mapped to the default name
        `interpolation_2d`.
    """

    name = name or "interpolation_2d"
    with tf.name_scope(name):
      self._xdata = tf.convert_to_tensor(x_data, dtype=dtype, name="x_data")
      self._dtype = dtype or self._xdata.dtype
      self._ydata = tf.convert_to_tensor(
          y_data, dtype=self._dtype, name="y_data")
      self._zdata = tf.convert_to_tensor(
          z_data, dtype=self._dtype, name="z_data")
      self._name = name

      # For each `x_data` point, build a spline in y-direction
      self._spline_yz = cubic.build_spline(
          self._ydata, self._zdata, name="spline_y_direction")

  def interpolate(self,
                  x_values,
                  y_values,
                  name=None):
    """Performs 2-D interpolation on a specified set of points.

    Args:
      x_values: Real-valued `Tensor` of shape `batch_shape + [num_points]`.
        Defines the x-coordinates at which the interpolation should be
        performed. Note that `batch_shape` should be the same as in the
        underlying data.
      y_values: A `Tensor` of the same shape and `dtype` as `x_values`.
        Defines the y-coordinates at which the interpolation should be
        performed.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` which is mapped to the default name
        `interpolate`.

    Returns:
      A `Tensor` of the same shape and `dtype` as `x_values`. Represents the
      interpolated values of the function on for the coordinates
      `(x_values, y_values)`.
    """
    name = name or self._name + "_interpolate"
    with tf.name_scope(name):
      x_values = tf.convert_to_tensor(
          x_values, dtype=self._dtype, name="x_values")
      y_values = tf.convert_to_tensor(
          y_values, dtype=self._dtype, name="y_values")

      # Broadcast `y_values` to the number of `x_data` points
      y_values = tf.expand_dims(y_values, axis=-2)
      # For each `x_data` point interpolate values of the function at the
      # y_values. Shape of `xy_values` is
      # batch_shape + `[num_x_data_points, num_points]`.
      xy_values = cubic.interpolate(
          y_values, self._spline_yz, name="interpolation_in_y_direction")
      # Interpolate the value of the function along x-direction
      # Prepare xy_values for linear interpolation. Put the batch dims in front
      xy_rank = xy_values.shape.rank
      perm = [xy_rank - 1] + list(range(xy_rank - 1))
      # Shape [num_points] + batch_shape + [num_x_data_points]
      yx_values = tf.transpose(xy_values, perm=perm)
      # Get the permutation to the original shape
      perm_original = list(range(1, xy_rank)) + [0]
      # Reshape to [num_points] + batch_shape + [1]
      x_values = tf.expand_dims(tf.transpose(
          x_values, [xy_rank - 2] + list(range(xy_rank - 2))), axis=-1)
      # Interpolation takes care of braodcasting
      z_values = linear.interpolate(x_values, self._xdata, yx_values)
      return tf.squeeze(tf.transpose(z_values, perm=perm_original), axis=-2)
