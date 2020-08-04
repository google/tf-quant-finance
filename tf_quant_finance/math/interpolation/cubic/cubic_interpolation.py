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
"""Cubic Spline interpolation framework."""


import collections
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.interpolation import utils

SplineParameters = collections.namedtuple(
    "SplineParameters",
    [
        # A real `Tensor` of shape batch_shape + [num_points] containing
        # X-coordinates of the spline.
        "x_data",
        # A `Tensor` of the same shape and `dtype` as `x_data` containing
        # Y-coordinates of the spline.
        "y_data",
        # A `Tensor` of the same shape and `dtype` as `x_data` containing
        # spline interpolation coefficients
        "spline_coeffs"
    ])


def build(x_data, y_data, validate_args=False, dtype=None, name=None):
  """Builds a SplineParameters interpolation object.

  Given a `Tensor` of state points `x_data` and corresponding values `y_data`
  creates an object that contains iterpolation coefficients. The object can be
  used by the `interpolate` function to get interpolated values for a set of
  state points `x` using the cubic spline interpolation algorithm.
  It assumes that the second derivative at the first and last spline points
  are zero. The basic logic is explained in [1] (see also, e.g., [2]).

  Repeated entries in `x_data` are allowed for the boundary values of `x_data`.
  For example, `x_data` can be `[1., 1., 2, 3. 4., 4., 4.]` but not
  `[1., 2., 2., 3.]`. The repeated values play no role in interpolation and are
  useful only for interpolating multiple splines with different numbers of data
  point. It is user responsibility to verify that the corresponding
  values of `y_data` are the same for the repeated values of `x_data`.

  Typical Usage Example:

  ```python
  import tensorflow.compat.v2 as tf
  import numpy as np

  x_data = np.linspace(-5.0, 5.0,  num=11)
  y_data = 1.0/(1.0 + x_data**2)
  spline = cubic_interpolation.build(x_data, y_data)
  x_args = [3.3, 3.4, 3.9]

  y = cubic_interpolation.interpolate(x_args, spline)
  ```

  #### References:
  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
    Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf
  [2]: R. Pienaar, M Choudhry. Fitting the term structure of interest rates:
    the practical implementation of cubic spline methodology.
    Link:
    http://yieldcurve.com/mktresearch/files/PienaarChoudhry_CubicSpline2.pdf

  Args:
    x_data: A real `Tensor` of shape `[..., num_points]` containing
      X-coordinates of points to fit the splines to. The values have to
      be monotonically non-decreasing along the last dimension.
    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing
      Y-coordinates of points to fit the splines to.
    validate_args: Python `bool`. When `True`, verifies if elements of `x_data`
      are sorted in the last dimension in non-decreasing order despite possibly
      degrading runtime performance.
      Default value: False.
    dtype: Optional dtype for both `x_data` and `y_data`.
      Default value: `None` which maps to the default dtype inferred by
      TensorFlow.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which is mapped to the default name
      `cubic_spline_build`.

  Returns:
    An instance of `SplineParameters`.
  """
  # Main body of build
  with tf.compat.v1.name_scope(
      name, default_name="cubic_spline_build", values=[x_data, y_data]):
    x_data = tf.convert_to_tensor(x_data, dtype=dtype, name="x_data")
    y_data = tf.convert_to_tensor(y_data, dtype=dtype, name="y_data")
    # Sanity check inputs
    if validate_args:
      assert_sanity_check = [_validate_arguments(x_data)]
    else:
      assert_sanity_check = []
    x_data, y_data = utils.broadcast_common_batch_shape(x_data, y_data)
    with tf.compat.v1.control_dependencies(assert_sanity_check):
      spline_coeffs = _calculate_spline_coeffs(x_data, y_data)

    return SplineParameters(x_data=x_data, y_data=y_data,
                            spline_coeffs=spline_coeffs)


def interpolate(x_values,
                spline_data,
                optimize_for_tpu=False,
                dtype=None,
                name=None):
  """Interpolates spline values for the given `x_values` and the `spline_data`.

  Constant extrapolation is performed for the values outside the domain
  `spline_data.x_data`. This means that for `x > max(spline_data.x_data)`,
  `interpolate(x, spline_data) = spline_data.y_data[-1]`
  and for  `x < min(spline_data.x_data)`,
  `interpolate(x, spline_data) = spline_data.y_data[0]`.

  For the interpolation formula refer to p.548 of [1].

  #### References:
  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
    Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf

  Args:
    x_values: A real `Tensor` of shape `batch_shape + [num_points]`.
    spline_data: An instance of `SplineParameters`. `spline_data.x_data` should
      have the same batch shape as `x_values`.
    optimize_for_tpu: A Python bool. If `True`, the algorithm uses one-hot
      encoding to lookup indices of `x_values` in `spline_data.x_data`. This
      significantly improves performance of the algorithm on a TPU device but
      may slow down performance on the CPU.
      Default value: `False`.
    dtype: Optional dtype for `x_values`.
      Default value: `None` which maps to the default dtype inferred by
      TensorFlow.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which is mapped to the default name
      `cubic_spline_interpolate`.

  Returns:
      A `Tensor` of the same shape and `dtype` as `x_values`. Represents
      the interpolated values.

  Raises:
    ValueError:
      If `x_values` batch shape is different from `spline_data.x_data` batch
      shape.
  """
  name = name or "cubic_spline_interpolate"
  with tf.name_scope(name):
    x_values = tf.convert_to_tensor(x_values, dtype=dtype, name="x_values")
    dtype = x_values.dtype
    # Unpack the spline data
    x_data = spline_data.x_data
    y_data = spline_data.y_data
    spline_coeffs = spline_data.spline_coeffs
    # Try broadcast batch_shapes
    x_values, x_data = utils.broadcast_common_batch_shape(x_values, x_data)
    x_values, y_data = utils.broadcast_common_batch_shape(x_values, y_data)
    x_values, spline_coeffs = utils.broadcast_common_batch_shape(x_values,
                                                                 spline_coeffs)
    # Determine the splines to use.
    indices = tf.searchsorted(x_data, x_values, side="right") - 1
    # This selects all elements for the start of the spline interval.
    # Make sure indices lie in the permissible range
    indices_lower = tf.maximum(indices, 0)
    # This selects all elements for the end of the spline interval.
    # Make sure indices lie in the permissible range
    indices_upper = tf.minimum(indices + 1, x_data.shape.as_list()[-1] - 1)
    # Prepare indices for `tf.gather_nd` or `tf.one_hot`
    # TODO(b/156720909): Extract get_slice logic into a common utilities module
    # for cubic and linear interpolation
    if optimize_for_tpu:
      x_data_size = x_data.shape.as_list()[-1]
      lower_encoding = tf.one_hot(indices_lower, x_data_size,
                                  dtype=dtype)
      upper_encoding = tf.one_hot(indices_upper, x_data_size,
                                  dtype=dtype)
    else:
      index_matrix = utils.prepare_indices(indices)
      lower_encoding = tf.concat(
          [index_matrix, tf.expand_dims(indices_lower, -1)], -1)
      upper_encoding = tf.concat(
          [index_matrix, tf.expand_dims(indices_upper, -1)], -1)

    # Calculate dx and dy.
    # Simplified logic:
    # dx = x_data[indices + 1] - x_data[indices]
    # dy = y_data[indices + 1] - y_data[indices]
    # indices is a tensor with different values per row/spline
    # Hence use a selection matrix with gather_nd
    def get_slice(x, encoding):
      if optimize_for_tpu:
        return tf.math.reduce_sum(tf.expand_dims(x, axis=-2) * encoding,
                                  axis=-1)
      else:
        return tf.gather_nd(x, encoding)
    x0 = get_slice(x_data, lower_encoding)
    x1 = get_slice(x_data, upper_encoding)
    dx = x1 - x0

    y0 = get_slice(y_data, lower_encoding)
    y1 = get_slice(y_data, upper_encoding)
    dy = y1 - y0

    spline_coeffs0 = get_slice(spline_coeffs, lower_encoding)
    spline_coeffs1 = get_slice(spline_coeffs, upper_encoding)

    t = (x_values - x0) / dx
    t = tf.where(dx > 0, t, tf.zeros_like(t))
    df = ((t + 1.0) * spline_coeffs1 * 2.0) - ((t - 2.0) * spline_coeffs0 * 2.0)
    df1 = df * t * (t - 1) / 6.0
    result = y0 + (t * dy) + (dx * dx * df1)
    # Use constant extrapolation outside the domain
    upper_bound = tf.expand_dims(
        tf.reduce_max(x_data, -1), -1) + tf.zeros_like(result)
    lower_bound = tf.expand_dims(
        tf.reduce_min(x_data, -1), -1) + tf.zeros_like(result)
    result = tf.where(tf.logical_and(x_values <= upper_bound,
                                     x_values >= lower_bound),
                      result, tf.where(x_values > upper_bound, y0, y1))
    return result


def _calculate_spline_coeffs(x_data, y_data):
  """Calculates the coefficients for the spline interpolation.

  These are the values of the second derivative of the spline at `x_data`.
  See p.548 of [1].

  Below is an outline of the function when number of observations if equal to 7.
  The coefficients are obtained by building and solving a tridiagonal linear
  system of equations with symmetric matrix

   w2,  dx2,   0,   0,   0
   dx2,  w3, dx3,   0,   0
   0,  dx3,   w4, dx4,   0
   0,    0,  dx4,  w5, dx5
   0,    0,    0, dx5,  w6

   where:
   wn = 2 * (x_data[n-2] + x_data[n-1])
   dxn = x_data[n-1] - x_data[n-2]

   and the right hand side of the equation is:
   [[3*( (d2-d1)/X1 - (d1-d0)/x0],
    [3*( (d3-d2)/X2 - (d2-d1)/x1],
    ...
   ]

   with di = y_data[..., i]

   Solve for `spline_coeffs`, so that  matrix * spline_coeffs = rhs

   the solution is the `spline_coeffs` parameter of the spline equation:

   y_pred = a(spline_coeffs) * t^3 + b(spline_coeffs) * t^2
            + c(spline_coeffs) * t + d(spline_coeffs)
   with t being the proportion of the difference between the x value of
   the spline used and the nx_value of the next spline:

   t = (x_values - x_data[:,n]) / (x_data[:,n+1]-x_data[:,n])

   and `a`, `b`, `c`, and `d` are functions of `spline_coeffs` and `x_data` and
   are provided in the `interpolate` function.

  #### References:
  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
    Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf

  Args:
    x_data: A real `Tensor` of shape `[..., num_points]` containing
      X-coordinates of points to fit the splines to. The values have to
      be monotonically non-decreasing along the last dimension.
    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing
      Y-coordinates of points to fit the splines to.

  Returns:
     A `Tensor` of the same shape and `dtype` as `x_data`. Represents the
     spline coefficients for the cubic spline interpolation.
  """

  # `dx` is the distances between the x points. It is 1 element shorter than
  # `x_data`
  dx = x_data[..., 1:] - x_data[..., :-1]

  # `diag_values` are the diagonal values 2 * (x_data[i+1] - x_data[i-1])
  # its length 2 shorter

  diag_values = 2.0 * (x_data[..., 2:] - x_data[..., :-2])
  superdiag = dx[..., 1:]
  subdiag = dx[..., :-1]

  corr_term = tf.logical_or(tf.equal(superdiag, 0), tf.equal(subdiag, 0))
  diag_values_corr = tf.where(corr_term,
                              tf.ones_like(diag_values), diag_values)
  superdiag_corr = tf.where(tf.equal(subdiag, 0),
                            tf.zeros_like(superdiag), superdiag)
  subdiag_corr = tf.where(tf.equal(superdiag, 0),
                          tf.zeros_like(subdiag), subdiag)
  diagonals = tf.stack([superdiag_corr, diag_values_corr, subdiag_corr],
                       axis=-2)

  # determine the rhs of the equation
  dd = (y_data[..., 1:] - y_data[..., :-1]) / dx
  dd = tf.where(tf.equal(dx, 0), tf.zeros_like(dd), dd)
  # rhs is a column vector:
  # [[-3((y1-y0)/dx0 - (y2-y1)/dx0], ...]
  rhs = -3 * (dd[..., :-1] - dd[..., 1:])
  rhs = tf.where(corr_term, tf.zeros_like(rhs), rhs)
  # Partial pivoting is unnecessary since the matrix is diagonally dominant.
  spline_coeffs = tf.linalg.tridiagonal_solve(diagonals, rhs,
                                              partial_pivoting=False)
  # Reshape `spline_coeffs`
  zero = tf.zeros_like(dx[..., :1], dtype=x_data.dtype)
  spline_coeffs = tf.concat([zero, spline_coeffs, zero], axis=-1)
  return spline_coeffs


def _validate_arguments(x_data):
  """Checks that input arguments are in the non-decreasing order."""
  diffs = x_data[..., 1:] - x_data[..., :-1]
  return tf.compat.v1.debugging.assert_greater_equal(
      diffs,
      tf.zeros_like(diffs),
      message="x_data is not sorted in non-decreasing order.")
