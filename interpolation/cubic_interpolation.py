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
"""Cubic Spline interpolation framework.

Given a tuple of state points `x_data` and corresponding values `y_data`
creates an object that can interpolate new values for a set of state points `x`
using a Cubic interpolation algorithm.
It assumes that the second derivative of the first and last spline points
are zero.

The basic logic is explained here:
  Algorithms in C
  Robert Sedegewick
  Princeton University
  see Reference [2]

  pages 545-550

however the solution of the matrix is done using tf.linalg.tridiagonal_solve.

Alternate Source: [4]

  The algorithm calculates the first derivatives S'(x). This and the x and y
  data points together provide the necessary information to calculate the
  interpolated data y = s(x)

  ## References:
  [1]: https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation#Methods
  [2]: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf
  [3]: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf page 550
  [4]: http://yieldcurve.com/mktresearch/files/PienaarChoudhry_CubicSpline2.pdf

  Typical Usage Example:

  ```python
  import tensorflow as tf
  import numpy as np

  x_data = np.linspace(-5.0, 5.0,  num=11)

  y_data = [1.0/(1.0 + x*x) for x in x ]

  x_series = tf.constant(np.array([x_data, ..]))
  y_series = tf.constant(np.array([y_data, ..]))
  spline = cubic_interpolation.build(x_series, y_series)

  x_args = [[3.3, 3.4, 3.9],
            [2.1, 2.4, 4.4],
           ...
           ]

  y = cubic_interpolation.interpolate(x_args, spline)
  ```
Using interpolate with x_values outside of [min(spline_x), max(spline_x))
will result in an exception
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

SplineParameters = collections.namedtuple(
    "SplineParameters",
    [
        # A `Tensor` of floats containing X coordinates of  the
        # Spline points. It is of shape [numSplines,splineLength].
        "x_data",
        # A `Tensor` of floats containing Y coordinates of  the
        # Spline points. It is of shape [numSplines,splineLength].
        "y_data",
        # A `Tensor` of floats containing the second derivatives of the splines
        "beta"
    ])


def build(x_data, y_data, name=None, dtype=None, validate_args=False):
  """Builds a Cubic Spline interpolation object.

  Args:
    x_data: `Tensor` of `float` containing X coordinates of points to fit the
      splines to. It is of shape [numSplines,splineLength]. The values have to
      be strictly monotonic increasing. Duplicate values will cause an
      exception.
    y_data: `Tensor` of `float` containing y coordinates of points to fit the
      splines to. It has the same shape as `x_data`
    name: Name of the operation
    dtype: Optional dtype for both `x_data` and `y_data`
    validate_args: Python `bool` indicating whether to validate arguments such
      as that x coordinates must be strictly monotonic increasing

  Returns:
    `SplineParameters` object to be used with `cubic_interpolation.interpolate`
  """

  def _validate_constructor_arguments(x_data):
    """Validates the arguments.

    Args:
       x_data: `Tensor` of floats containing X coordinates of points to fit the
         splines to. It is of shape [numSplines,splineLength]. The values have
         to be strictly monotonic increasing. Duplicate values will cause an
         exception.

    Returns:
      An op that will evaluate true or false
    """

    # check: are the x_data strictly increasing
    diffs = x_data[:, 1:] - x_data[:, :-1]
    # diffs should all be larger than 0
    return tf.debugging.assert_greater(
        diffs,
        tf.zeros_like(diffs),
        message="x_data are not strictly increasing")

  def _calculate_beta(x_data, y_data, dtype=None):
    """Calculates the coefficients for the second derivative.

    Args:
      x_data: `Tensor` of floats containing X coordinates of points to fit the
        splines to. It is of shape [numSplines,splineLength]. The values have to
        be strictly monotonic increasing. Duplicate values will cause an
        exception.
      y_data: `Tensor` of floats containing y coordinates of points to fit the
        splines to. It has the same shape as `x_data`
      dtype: Optional dtype for both `x_data` and `y_data`

    Returns:
       The solution to the tridiagonal system of equations.

    Do this by building and solving a linear system of equaitons with matrix

     w2,  dx2,   0,   0,   0
     dx2,  w3, dx3,   0,   0
     0,  dx3,   w4, dx4,   0
     0,    0,  dx4,  w5, dx5
     0,    0,    0, dx5,  w6

     where:
     wn = 2 * (x[n-2] + x[n-1])
     dxn = x[n-1] - x[n-2]


     and the right hand side of the equation is:
     [[3*( (d2-d1)/X1 - (d1-d0)/x0],
      [3*( (d3-d2)/X2 - (d2-d1)/x1],
      ...
     ]

     with dn = self.y_points[:,n]

     Solve for beta, so that beta * matrix = rhs

     the solution is the beta parameter of the spline euqation:

     y_pred = alpha * t^3 + beta * t^2  + gamma * t + d
     with t being the proportion of the difference between the x value of
     the spline used and the nx_value of the next spline.

     t = (x_input - x_data[:,n])/(x_data[:,n+1]-x_data[:,n])

     Note: alpha and gamma are not calculated explicitly, they are derived from
           beta, d and X during the project method.
    """

    # dx is the distances between the x points. It i1 1 shorter than x_data
    dx = x_data[:, 1:] - x_data[:, :-1]

    # diag_values are the diagonal values 2 * (dx[i] + dx[i+1])
    # its length 2 shorter

    diag_values = 2.0 * (x_data[:, 2:] - x_data[:, :-2])
    superdiag = dx[:, 1:]
    subdiag = dx[:, :-1]

    diagonals = tf.stack([superdiag, diag_values, subdiag], axis=1)

    # determine the rhs of the equation
    dd = (y_data[:, 1:] - y_data[:, :-1]) / dx
    # rhs is a column vector:
    # [[-3((y1-y0)/dx0 - (y2-y1)/dx0], ...]
    rhs = tf.expand_dims((dd[:, :-1] - dd[:, 1:]) * -3.0, axis=2)
    beta = tf.linalg.tridiagonal_solve(diagonals, rhs)

    # Reshape beta
    zero = tf.zeros_like(dx[:, :1], dtype=dtype)
    beta = tf.concat([zero, tf.squeeze(beta, axis=[2]), zero], axis=1)

    return beta

  # Main body of build
  with tf.name_scope(
      name, default_name="cubic_spline_build", values=[x_data, y_data]):
    x_data = tf.convert_to_tensor(x_data, name="x_data", dtype=dtype)
    y_data = tf.convert_to_tensor(y_data, name="y_data", dtype=dtype)

    # sanity check inputs
    if validate_args:
      assert_sanity_check = [_validate_constructor_arguments(x_data)]
    else:
      assert_sanity_check = []

    with tf.control_dependencies(assert_sanity_check):
      beta = _calculate_beta(x_data, y_data, dtype=dtype)

    return SplineParameters(x_data=x_data, y_data=y_data, beta=beta)


def interpolate(x_values,
                spline_data,
                validate_args=False,
                dtype=None,
                name=None):
  """Interpolates y_values for the given x_values and the spline_data.

  Args:
       x_values   : `Tensor` of floats containing x coordinates of points
       spline_data: `SplineParameters` built by `cubic_interpolation.build`. if
         spline_data.beta is None then build will be called.
       validate_args: Python `bool` indicating whether to validate that the
         x_values are within spline boundaries
       dtype: Optional dtype for both `x_data` and `y_data`
       name: Optional name of the operation

  Returns:
      A `Tensor` of `float` that represent the y_values interpolated
      from the `x_values`
  """

  def _is_inside(x_data, to_test):
    """Test that all values in test are within x_data.

    Args:
       x_data: `Tensor` of float. shape (numSplines, spline_length)
       to_test: `Tensor` of float, shape (num_splines, n_tests)
    Returns: `Tensor` of True or False  Establishes that - any point in
      to_test[spline_idx, test_idx] >= data[spline_idx, 0], - any point in
      to_test[spline_idx, test_idx] < data[spline_idx, -1]
    """
    # Lower take the smallest value for each point and compare it
    # With points[:,0]
    lower_test = tf.reduce_min(to_test, axis=-1)
    lower = tf.greater_equal(lower_test, x_data[:, 0])

    # Get the largest value
    upper_test = tf.reduce_max(to_test, axis=-1)
    upper = tf.less(upper_test, x_data[:, -1])

    return tf.reduce_all(tf.logical_and(lower, upper))

  # Check that beta is supplied. If not call build()
  if spline_data.beta is None:
    spline_data = build(
        spline_data.x_data,
        spline_data.y_data,
        name=name,
        dtype=dtype,
        validate_args=validate_args)
  # Unpack the spline data
  x_data = spline_data.x_data
  y_data = spline_data.y_data
  beta = spline_data.beta

  x_values = tf.convert_to_tensor(x_values, name="x_values", dtype=dtype)

  # Check that all the x_values are within the boundaries
  if x_values.shape[0] != x_data.shape[0]:
    msg = ("the input tensor has a different number of rows than the "
           "number of splines: {} != {}")
    raise ValueError(msg.format(x_values.shape[0], x_data.shape[0]))

  with tf.name_scope(
      name,
      default_name="cubic_spline_interpolate",
      values=[x_data, y_data, beta, x_values]):

    # Make sure x_values are legal.
    if validate_args:
      assert_is_inside = [
          tf.Assert(_is_inside(x_data, x_values), [x_data, x_values])
      ]
    else:
      assert_is_inside = []

    with tf.control_dependencies(assert_is_inside):
      # Determine the splines to use.
      indices = tf.searchsorted(x_data, x_values, side="right") - 1

      # Prepares the indices so that it can be used in gather_nd.
      row_indices = tf.range(indices.shape[0], dtype=dtype)
      index_matrix = tf.transpose(tf.tile([row_indices], [indices.shape[1], 1]))
      # This selects all elements for the start of the spline interval.
      selection_matrix = tf.stack([index_matrix, indices], axis=-1)
      # This selects all elements for the end of the spline interval.
      selection_matrix_1 = tf.stack([index_matrix, indices + 1], axis=-1)

      # Calculate dx and dy.
      # Simplified logic:
      # dx = x_data[indices + 1] - x_data[indices]
      # dy = y_data[indices + 1] - y_data[indices]
      # indices is a tensor with different values per row/spline
      # Hence use a selection matrix with gather_nd

      x0 = tf.gather_nd(x_data, selection_matrix)
      x1 = tf.gather_nd(x_data, selection_matrix_1)
      dx = x1 - x0

      y0 = tf.gather_nd(y_data, selection_matrix)
      y1 = tf.gather_nd(y_data, selection_matrix_1)
      dy = y1 - y0
      beta0 = tf.gather_nd(beta, selection_matrix)
      beta1 = tf.gather_nd(beta, selection_matrix_1)

      # This reduces the amount of calculation effort to derive
      # alpha and gamma separately.
      # Reference: [3]

      t = (x_values - x0) / dx
      df = ((t + 1.0) * beta1 * 2.0) - ((t - 2.0) * beta0 * 2.0)
      df1 = df * t * (t - 1) / 6.0
      result = y0 + (t * dy) + (dx * dx * df1)

      return result
