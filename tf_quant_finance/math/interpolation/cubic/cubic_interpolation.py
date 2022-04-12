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

import enum
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils


__all__ = [
    'BoundaryConditionType',
    'SplineParameters',
    'build',
    'interpolate',
]


@enum.unique
class BoundaryConditionType(enum.Enum):
  """Specifies which boundary condition type to use for the cubic interpolation.

  * `NATURAL`: the cubic interpolation set second derivative equal to zero
  at boundaries.
  * `CLAMPED`: the cubic interpolation set first derivative equal to zero
  at boundaries.
  * `FIXED_FIRST_DERIVATIVE`: the cubic interpolation set first derivative to
  certain value at boundaries.
  """
  NATURAL = 1
  CLAMPED = 2
  FIXED_FIRST_DERIVATIVE = 3


@tff_utils.dataclass
class SplineParameters:
  """Cubic spline parameters.

  Attributes:
    x_data: A real `Tensor` of shape batch_shape + [num_points] containing
      X-coordinates of the spline.
    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing
      Y-coordinates of the spline.
    spline_coeffs: A `Tensor` of the same shape and `dtype` as `x_data`
      containing spline interpolation coefficients
  """
  x_data: types.RealTensor
  y_data: types.RealTensor
  spline_coeffs: types.RealTensor


def build(x_data: types.RealTensor,
          y_data: types.RealTensor,
          boundary_condition_type: BoundaryConditionType = None,
          left_boundary_value: types.RealTensor = None,
          right_boundary_value: types.RealTensor = None,
          validate_args: bool = False,
          dtype: tf.DType = None,
          name=None) -> SplineParameters:
  """Builds a SplineParameters interpolation object.

  Given a `Tensor` of state points `x_data` and corresponding values `y_data`
  creates an object that contains interpolation coefficients. The object can be
  used by the `interpolate` function to get interpolated values for a set of
  state points `x` using the cubic spline interpolation algorithm.
  It assumes that the second derivative at the first and last spline points
  are zero. The basic logic is explained in [1] (see also, e.g., [2]).

  Repeated entries in `x_data` are only allowed for the *right* boundary values
  of `x_data`.
  For example, `x_data` can be `[1., 2, 3. 4., 4., 4.]` but not
  `[1., 1., 2., 3.]`. The repeated values play no role in interpolation and are
  useful only for interpolating multiple splines with different numbers of data
  point. It is user responsibility to verify that the corresponding
  values of `y_data` are the same for the repeated values of `x_data`.

  Typical Usage Example:

  ```python
  import tensorflow as tf
  import tf_quant_finance as tff
  import numpy as np

  x_data = tf.linspace(-5.0, 5.0,  num=11)
  y_data = 1.0/(1.0 + x_data**2)
  spline = tff.math.interpolation.cubic.build_spline(x_data, y_data)
  x_args = [3.3, 3.4, 3.9]

  tff.math.interpolation.cubic.interpolate(x_args, spline)
  # Expected: [0.0833737 , 0.07881707, 0.06149562]
  ```

  #### References:
  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
    Link: https://api.semanticscholar.org/CorpusID:10976311
  [2]: R. Pienaar, M Choudhry. Fitting the term structure of interest rates:
    the practical implementation of cubic spline methodology.
    Link:
    http://yieldcurve.com/mktresearch/files/PienaarChoudhry_CubicSpline2.pdf

  Args:
    x_data: A real `Tensor` of shape `[..., num_points]` containing
      X-coordinates of points to fit the splines to. The values have to be
      monotonically non-decreasing along the last dimension.
    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing
      Y-coordinates of points to fit the splines to.
    boundary_condition_type: Boundary condition type for current cubic
      interpolation. Instance of BoundaryConditionType enum.
      Default value: `None` which maps to `BoundaryConditionType.NATURAL`.
    left_boundary_value: Set to non-empty value IFF boundary_condition_type is
      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's first
      derivative at `x_data[..., 0]`.
    right_boundary_value: Set to non-empty value IFF boundary_condition_type is
      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's first
      derivative at `x_data[..., num_points - 1]`.
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
  if boundary_condition_type is None:
    boundary_condition_type = BoundaryConditionType.NATURAL
  if name is None:
    name = 'cubic_spline_build'
  with tf.name_scope(name):
    x_data = tf.convert_to_tensor(x_data, dtype=dtype, name='x_data')
    y_data = tf.convert_to_tensor(y_data, dtype=dtype, name='y_data')
    # Sanity check inputs
    if validate_args:
      assert_sanity_check = [_validate_arguments(x_data)]
    else:
      assert_sanity_check = []
    x_data, y_data = tff_utils.broadcast_common_batch_shape(x_data, y_data)

    if boundary_condition_type == BoundaryConditionType.FIXED_FIRST_DERIVATIVE:
      if left_boundary_value is None or right_boundary_value is None:
        raise ValueError(
            'Expected non-empty left_boundary_value/right_boundary_value when '
            'boundary_condition_type is FIXED_FIRST_DERIVATIVE, actual '
            'left_boundary_value {0}, actual right_boundary_value {1}'.format(
                left_boundary_value, right_boundary_value))
    with tf.compat.v1.control_dependencies(assert_sanity_check):
      spline_coeffs = _calculate_spline_coeffs(x_data, y_data,
                                               boundary_condition_type,
                                               left_boundary_value,
                                               right_boundary_value)

    return SplineParameters(
        x_data=x_data, y_data=y_data, spline_coeffs=spline_coeffs)


def interpolate(x: types.RealTensor,
                spline_data: SplineParameters,
                optimize_for_tpu: bool = False,
                dtype: tf.DType = None,
                name: str = None) -> types.RealTensor:
  """Interpolates spline values for the given `x` and the `spline_data`.

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
    x: A real `Tensor` of shape `batch_shape + [num_points]`.
    spline_data: An instance of `SplineParameters`. `spline_data.x_data` should
      have the same batch shape as `x`.
    optimize_for_tpu: A Python bool. If `True`, the algorithm uses one-hot
      encoding to lookup indices of `x` in `spline_data.x_data`. This
      significantly improves performance of the algorithm on a TPU device but
      may slow down performance on the CPU.
      Default value: `False`.
    dtype: Optional dtype for `x`.
      Default value: `None` which maps to the default dtype inferred by
        TensorFlow.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which is mapped to the default name
        `cubic_spline_interpolate`.

  Returns:
      A `Tensor` of the same shape and `dtype` as `x`. Represents
      the interpolated values.

  Raises:
    ValueError:
      If `x` batch shape is different from `spline_data.x_data` batch
      shape.
  """
  name = name or 'cubic_spline_interpolate'
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    dtype = x.dtype
    # Unpack the spline data
    x_data = spline_data.x_data
    y_data = spline_data.y_data
    spline_coeffs = spline_data.spline_coeffs
    # Try broadcast batch_shapes
    x, x_data, y_data, spline_coeffs = tff_utils.broadcast_common_batch_shape(
        x, x_data, y_data, spline_coeffs)
    # Determine the splines to use.
    indices = tf.searchsorted(x_data, x, side='right') - 1
    # This selects all elements for the start of the spline interval.
    # Make sure indices lie in the permissible range
    lower_encoding = tf.maximum(indices, 0)
    # This selects all elements for the end of the spline interval.
    # Make sure indices lie in the permissible range
    upper_encoding = tf.minimum(indices + 1,
                                tff_utils.get_shape(x_data)[-1] - 1)
    # Prepare indices for `tf.gather` or `tf.one_hot`
    # TODO(b/156720909): Extract get_slice logic into a common utilities module
    # for cubic and linear interpolation
    if optimize_for_tpu:
      x_data_size = tff_utils.get_shape(x_data)[-1]
      lower_encoding = tf.one_hot(lower_encoding, x_data_size, dtype=dtype)
      upper_encoding = tf.one_hot(upper_encoding, x_data_size, dtype=dtype)
    # Calculate dx and dy.
    # Simplified logic:
    # dx = x_data[indices + 1] - x_data[indices]
    # dy = y_data[indices + 1] - y_data[indices]
    # indices is a tensor with different values per row/spline
    def get_slice(x, encoding):
      if optimize_for_tpu:
        return tf.math.reduce_sum(
            tf.expand_dims(x, axis=-2) * encoding, axis=-1)
      else:
        return tf.gather(x, encoding, axis=-1, batch_dims=x.shape.rank - 1)

    x0 = get_slice(x_data, lower_encoding)
    x1 = get_slice(x_data, upper_encoding)
    dx = x1 - x0

    y0 = get_slice(y_data, lower_encoding)
    y1 = get_slice(y_data, upper_encoding)
    dy = y1 - y0

    spline_coeffs0 = get_slice(spline_coeffs, lower_encoding)
    spline_coeffs1 = get_slice(spline_coeffs, upper_encoding)

    t = (x - x0) / dx
    t = tf.where(dx > 0, t, tf.zeros_like(t))
    df = ((t + 1.0) * spline_coeffs1 * 2.0) - ((t - 2.0) * spline_coeffs0 * 2.0)
    df1 = df * t * (t - 1) / 6.0
    result = y0 + (t * dy) + (dx * dx * df1)
    # Use constant extrapolation outside the domain
    upper_bound = tf.expand_dims(tf.reduce_max(x_data, -1),
                                 -1) + tf.zeros_like(result)
    lower_bound = tf.expand_dims(tf.reduce_min(x_data, -1),
                                 -1) + tf.zeros_like(result)
    result = tf.where(
        tf.logical_and(x <= upper_bound, x >= lower_bound),
        result, tf.where(x > upper_bound, y0, y1))
    return result


def _calculate_spline_coeffs_natural(dx, superdiag, subdiag, diag_values, rhs,
                                     dtype):
  """Calculates spline coefficients for the NATURAL boundary condition."""
  # remove duplicate
  corr_term = tf.logical_or(tf.equal(superdiag, 0), tf.equal(subdiag, 0))
  diag_values_corr = tf.where(corr_term, tf.ones_like(diag_values), diag_values)
  superdiag_corr = tf.where(
      tf.equal(subdiag, 0), tf.zeros_like(superdiag), superdiag)
  subdiag_corr = tf.where(
      tf.equal(superdiag, 0), tf.zeros_like(subdiag), subdiag)
  diagonals = tf.stack([superdiag_corr, diag_values_corr, subdiag_corr],
                       axis=-2)
  # Transform a matrix to make it invertible.
  # For the cases input x_data and y_data contains duplicate points.
  rhs = tf.where(corr_term, tf.zeros_like(rhs), rhs)

  # Partial pivoting is unnecessary since the matrix is diagonally dominant.
  spline_coeffs = tf.linalg.tridiagonal_solve(
      diagonals, rhs, partial_pivoting=False)
  # Reshape `spline_coeffs`
  zero = tf.zeros_like(dx[..., :1], dtype=dtype)
  spline_coeffs = tf.concat([zero, spline_coeffs, zero], axis=-1)
  return spline_coeffs


def _calculate_spline_coeffs_clamped_or_first_derivative(
    dx,
    dd,
    superdiag,
    subdiag,
    diag_values,
    rhs,
    dtype,
    boundary_condition_type,
    left_boundary_value=None,
    right_boundary_value=None):
  """Calculates the coefficients for the spline interpolation if the boundary condition type is CLAMPED/FIXED_FIRST_DERIVATIVE."""
  zero = tf.zeros_like(dx[..., :1], dtype=dtype)
  one = tf.ones_like(dx[..., :1], dtype=dtype)
  diag_values = tf.concat([2.0 * dx[..., :1], diag_values, zero], axis=-1)
  superdiag = tf.concat([dx[..., :1], superdiag, zero], axis=-1)
  subdiag = tf.concat([zero, subdiag, dx[..., -1:]], axis=-1)

  # locate right boundary when duplicates exists
  dx = tf.concat((one, dx, zero), axis=-1)
  dx_right = dx[..., 1:]
  dx_left = dx[..., :-1]
  right_boundary = tf.math.logical_and(
      tf.equal(dx_right, 0), tf.not_equal(dx_left, 0))

  # For diag_values, at the right boundary, fill the value as 2.0 * dx.
  # For the right padding beyond boundary, fill the default value as 1.0.
  # No need to operate on super_diag/sub_diag,
  # since dx[..., -1:] is already zero
  diag_values = tf.where(right_boundary, 2.0 * dx_left, diag_values)
  diag_values = tf.where(tf.equal(dx_left, 0), one, diag_values)

  # build diagonals
  diagonals = tf.stack([superdiag, diag_values, subdiag], axis=-2)

  # build rhs
  left_boundary_tensor = tf.zeros_like(dx[..., :1], dtype=dtype)
  right_boundary_tensor = tf.zeros_like(dx[..., :1], dtype=dtype)
  if boundary_condition_type == BoundaryConditionType.FIXED_FIRST_DERIVATIVE:
    left_boundary_tensor = tf.convert_to_tensor(
        left_boundary_value, dtype=dtype, name='left_boundary_value')
    right_boundary_tensor = tf.convert_to_tensor(
        right_boundary_value, dtype=dtype, name='right_boundary_value')
  top_rhs = 3.0 * (dd[..., :1] - left_boundary_tensor[..., :1])
  rhs = tf.concat([top_rhs, rhs, zero], axis=-1)
  # For rhs, at the right boundary, fill the value as bottom_rhs.
  # For the right padding beyond boundary, fill the default value as 0.0.
  dd_left = tf.concat((one, dd), axis=-1)
  bottom_rhs = -3.0 * (dd_left - right_boundary_tensor[..., :1])
  rhs = tf.where(right_boundary, bottom_rhs, rhs)
  rhs = tf.where(tf.equal(dd_left, 0), zero, rhs)

  spline_coeffs = tf.linalg.tridiagonal_solve(
      diagonals, rhs, partial_pivoting=False)
  return spline_coeffs


def _calculate_spline_coeffs(
    x_data,
    y_data,
    boundary_condition_type=BoundaryConditionType.NATURAL,
    left_boundary_value=None,
    right_boundary_value=None):
  """Calculates the coefficients for the spline interpolation.

  These are the values of the second derivative of the spline at `x_data`.
  See p.548 of [1].

  #### Below formula is for natural condition type.
  It is an outline of the function when number of observations if equal to 7.
  The coefficients are obtained by building and solving a tridiagonal linear
  system of equations with symmetric matrix
   1,  0,  0,    0,    0,   0,  0
  dx0  w0, dx1,  0,    0,   0,  0
   0, dx1,  w1,  dx2,  0,   0,  0
   0,  0,  dx2,  w2,  dx3,  0,  0
   0,  0,   0,   dx3,  w3, dx4, 0
   0,  0,   0,   0,   dx4,  w4, dx5
   0,  0,   0,   0,    0,   0,  1
   where:
   dxn = x_data[n+1] - x_data[n]
   wn = 2 * (dx[n] + dx[n+1])

   and the right hand side of the equation is:
   [[0],
    [3*( (y2-y1)/dx1 - (y1-y0)/dx0],
    [3*( (y3-y2)/dx2 - (y2-y1)/dx1],
    [3*( (y4-y3)/dx3 - (y3-y2)/dx2],
    [3*( (y5-y4)/dx4 - (y4-y3)/dx3],
    [3*( (y6-y5)/dx5 - (y5-y4)/dx4],
    [0]
   ]

   with yi = y_data[..., i]

   Solve for `spline_coeffs`, so that  matrix * spline_coeffs = rhs

   the solution is the `spline_coeffs` parameter of the spline equation:

   y_pred = a(spline_coeffs) * t^3 + b(spline_coeffs) * t^2
            + c(spline_coeffs) * t + d(spline_coeffs)
   with t being the proportion of the difference between the x value of
   the spline used and the nx_value of the next spline:

   t = (x - x_data[:,n]) / (x_data[:,n+1]-x_data[:,n])

   and `a`, `b`, `c`, and `d` are functions of `spline_coeffs` and `x_data` and
   are provided in the `interpolate` function.

  #### Below formula is for clamped/first_derivative condition type.
  Similar to natural condition type, let us assume the number of observations
  is equal to 7. The underlying mathematics can be found in [2].
  left hand side matrix:
  2*dx0, dx0,  0,   0,    0,   0,  0
   dx0    w0, dx1,  0,    0,   0,  0
    0,   dx1,  w1,  dx2,  0,   0,  0
    0,    0,  dx2,  w2,  dx3,  0,  0
    0,    0,   0,   dx3,  w3, dx4, 0
    0,    0,   0,   0,   dx4,  w4, dx5
    0,    0,   0,   0,    0,  dx5, 2*dx5
   where:
   dxn and wn is same as natural contition case.

   and the right hand side of the equation is:
   [[3* ((y1-y0)/dx0 - lb)],
    [3*( (y2-y1)/dx1 - (y1-y0)/dx0],
    [3*( (y3-y2)/dx2 - (y2-y1)/dx1],
    [3*( (y4-y3)/dx3 - (y3-y2)/dx2],
    [3*( (y5-y4)/dx4 - (y4-y3)/dx3],
    [3*( (y6-y5)/dx5 - (y5-y4)/dx4],
    [-3*((y6-y5)/dx5 - rb)]
   ]
   where dxn, yi is same as natural case.
   lb is specified first derivative at left boundary.
   rb is specified first derivative at right boundary.

  #### Special handling for right padding, imagine the number of observations
  is equal to 7. While there are 2 repeated points as right padding.
  The left hand matrix needs to be:
  2*dx0, dx0,  0,   0,    0,   0,  0     0,  0
   dx0    w0, dx1,  0,    0,   0,  0     0,  0
    0,   dx1,  w1,  dx2,  0,   0,  0     0,  0
    0,    0,  dx2,  w2,  dx3,  0,  0     0,  0
    0,    0,   0,   dx3,  w3, dx4, 0     0,  0
    0,    0,   0,   0,   dx4,  w4, dx5   0,  0
    0,    0,   0,   0,    0,  dx5, 2*dx5 0,  0
    0,    0,   0,   0,    0,  0,   0,    1,  0
    0,    0,   0,   0,    0,  0,   0,    0,  1

   The right hand matrix needs to be:
    [[3* ((y1-y0)/dx0 - lb)],
    [3*( (y2-y1)/dx1 - (y1-y0)/dx0],
    [3*( (y3-y2)/dx2 - (y2-y1)/dx1],
    [3*( (y4-y3)/dx3 - (y3-y2)/dx2],
    [3*( (y5-y4)/dx4 - (y4-y3)/dx3],
    [3*( (y6-y5)/dx5 - (y5-y4)/dx4],
    [-3*((y6-y5)/dx5 - rb)],
    [0],
    [0]
   ]

  #### References:
  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.
    Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf

  Args:
    x_data: A real `Tensor` of shape `[..., num_points]` containing
      X-coordinates of points to fit the splines to. The values have to be
      monotonically non-decreasing along the last dimension.
    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing
      Y-coordinates of points to fit the splines to.
    boundary_condition_type: Boundary condition type for current cubic
      interpolation.
    left_boundary_value: Set to non-empty value IFF boundary_condition_type is
      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's
      first derivative at x_data[: 0].
    right_boundary_value: Set to non-empty value IFF boundary_condition_type is
      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's
      first derivative at x_data[: num_points - 1]

  Returns:
     A `Tensor` of the same shape and `dtype` as `x_data`. Represents the
     spline coefficients for the cubic spline interpolation.
  [2]: http://macs.citadel.edu/chenm/343.dir/09.dir/lect3_4.pdf
  """

  # `dx` is the distances between the x points. It is 1 element shorter than
  # `x_data`
  dx = x_data[..., 1:] - x_data[..., :-1]

  # determine the rhs of the equation
  dd = (y_data[..., 1:] - y_data[..., :-1]) / dx
  dd = tf.where(tf.equal(dx, 0), tf.zeros_like(dd), dd)
  # rhs is a column vector:
  # [[-3((y1-y0)/dx0 - (y2-y1)/dx0], ...]
  # Its length is 2 shorter compared to number of points. Since this is the
  # "inner" part before we have finalized rhs.
  # The finalized rhs's first and last element will be
  # decided based on the boundary conditions.
  rhs = -3 * (dd[..., :-1] - dd[..., 1:])

  # `diag_values` are the diagonal values 2 * (x_data[i+1] - x_data[i-1]).
  # Its length is 2 shorter compared to number of points. Since this is the
  # "inner" part before we have finalized diagonals.
  # The finalized diagonals's first and last element will be
  # decided based on the boundary conditions.

  diag_values = 2.0 * (x_data[..., 2:] - x_data[..., :-2])
  superdiag = dx[..., 1:]
  subdiag = dx[..., :-1]
  # Compute first and last element for diagonal and rhs,
  # the computation varies per boundary condition.
  if boundary_condition_type == BoundaryConditionType.NATURAL:
    return _calculate_spline_coeffs_natural(dx, superdiag, subdiag, diag_values,
                                            rhs, x_data.dtype)
  elif boundary_condition_type in [
      BoundaryConditionType.FIXED_FIRST_DERIVATIVE,
      BoundaryConditionType.CLAMPED
  ]:
    return _calculate_spline_coeffs_clamped_or_first_derivative(
        dx, dd, superdiag, subdiag, diag_values, rhs, x_data.dtype,
        boundary_condition_type, left_boundary_value, right_boundary_value)


def _validate_arguments(x_data):
  """Checks that input arguments are in the non-decreasing order."""
  diffs = x_data[..., 1:] - x_data[..., :-1]
  return tf.compat.v1.debugging.assert_greater_equal(
      diffs,
      tf.zeros_like(diffs),
      message='x_data is not sorted in non-decreasing order.')
