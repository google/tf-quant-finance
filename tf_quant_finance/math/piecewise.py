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
"""Piecewise utility functions."""

import tensorflow.compat.v2 as tf


class PiecewiseConstantFunc(object):
  """Creates a piecewise constant function."""

  def __init__(self, jump_locations, values, dtype=None, name=None):
    r"""Initializes jumps of the piecewise constant function.

    Sets jump locations and values for a piecewise constant function.
    `jump_locations` split real line into intervals
    `[-inf, jump_locations[..., 0]], ..,
    [jump_locations[..., i], jump_locations[..., i + 1]], ...,
    [jump_locations[..., -1], inf]`
    so that the piecewise constant function takes corresponding `values` on the
    intervals, i.e.,
    ```
    f(x) = \sum_i values[..., i]
           * I_{x \in [jump_locations[..., i -1], jump_locations[..., i])}
    ```

    #### Example. Simple scalar-valued piecewise constant function.
    ```python
    dtype = np.float64
    jump_locations = [0.1, 10]
    values = [3, 4, 5]
    piecewise_func = piecewise.PiecewiseConstantFunc(jump_locations, values,
                                                     dtype=dtype)
    # Locations at which to evaluate the function assuming it is
    # left-continuous.
    x = np.array([0., 0.1, 2., 11.])
    value = piecewise_func(x)
    # Expected values: [3, 3, 4, 5]
    integral = piecewise_func.integrate(x, x + 1)
    # Expected integrals: [3.9, 4, 4, 5]
    ```

    #### Example. Matrix-valued piecewise constant function.
    ```python
    dtype = np.float64
    jump_locations = [0.1, 10]
    # The function takes values [[1, 2], [3, 4]] on (-inf, 0.1),
    # [[5, 6], [7, 8]] on (0.1, 0.5), and [[9, 10], [11, 12]] on (0.5, +inf).
    values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    piecewise_func = piecewise.PiecewiseConstantFunc(
        jump_locations, values, dtype=dtype)
    # Locations at which to evaluate the function assuming it is
    # left-continuous.
    x = np.array([0., 0.1, 2, 11])
    value = piecewise_func(x)
    # Expected values: [[[1, 2], [3, 4]], [[1, 2], [3, 4]],
    #                   [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    integral = piecewise_func.integrate(x, x + 1)
    # Expected integrals: [[[4.6, 5.6], [6.6, 7.6]],
    #                      [[5, 6], [7, 8]],
    #                      [[5, 6], [7, 8]],
    #                      [[9, 10], [11, 12]]]
    ```

    Args:
      jump_locations: A real `Tensor` of shape
        `batch_shape + [num_jump_points]`. The locations where the function
        changes its values. Note that the values are expected to be ordered
        along the last dimension. Repeated values are allowed but it is
        up to the user to ensure that the corresponding `values` are also
        repeated.
      values: A `Tensor` of the same `dtype` as `jump_locations` and shape
        `batch_shape + [num_jump_points + 1] + event_shape`. Defines
        `values[batch_rank * slice(None), i]` on intervals
        `(jump_locations[..., i - 1], jump_locations[..., i])`. Here
        `event_shape` allows for array-valued piecewise constant functions
        and `batch_rank = len(batch_shape)`.
      dtype:  Optional dtype for `jump_locations` and `values`.
        Default value: `None` which maps to the default dtype inferred from
        `jump_locations`.
      name: Python `str` name prefixed to ops created by this class.
        Default value: `None` which is mapped to the default name
        `PiecewiseConstantFunc`.

    Raises:
      ValueError:
        If `jump_locations` and `values` have different batch shapes or,
        in case of static shapes, if the event shape of `values` is different
        from `num_jump_points + 1`.
    """
    self._name = name or 'PiecewiseConstantFunc'
    # Add a property that indicates that the class instance is a
    # piecewise constant function
    self.is_piecewise_constant = True
    with tf.name_scope(self._name):
      self._jump_locations = tf.convert_to_tensor(jump_locations, dtype=dtype,
                                                  name='jump_locations')
      self._dtype = dtype or self._jump_locations.dtype
      self._values = tf.convert_to_tensor(values, dtype=self._dtype,
                                          name='values')
      shape_values = self._values.shape.as_list()

      shape_jump_locations = self._jump_locations.shape.as_list()
      batch_rank = self._jump_locations.shape.rank - 1
      self._batch_rank = batch_rank
      if None not in shape_values and None not in shape_jump_locations:
        if shape_values[:batch_rank] != shape_jump_locations[:-1]:
          raise ValueError(
              'Batch shapes of `values` and `jump_locations` should '
              'be the same but are {0} and {1}'.format(
                  shape_values[:-1], shape_jump_locations[:-1]))
        if shape_values[batch_rank] - 1 != shape_jump_locations[-1]:
          raise ValueError('Event shape of `values` should have one more '
                           'element than the event shape of `jump_locations` '
                           'but are {0} and {1}'.format(
                               shape_values[-1], shape_jump_locations[-1]))

  def dtype(self):
    """The underlying data type."""
    return self._dtype

  def values(self):
    """The value of the piecewise constant function between jump locations."""
    return self._values

  def jump_locations(self):
    """The jump locations of the piecewise constant function."""
    return self._jump_locations

  def name(self):
    """The name to give to the ops created by this class."""
    return self._name

  def __call__(self, x, left_continuous=True, name=None):
    """Computes value of the piecewise constant function.

    Returns a value of the piecewise function with jump locations and values
    given by the initializer.

    Args:
      x: A real `Tensor` of shape `batch_shape + [num_points]`. Points at which
        the function has to be evaluated.
      left_continuous: Python `bool`. Whether the function is left- or right-
        continuous, i.e., at the `jump_locations[..., i]` left-continuity means
        that the function has the same value
        `values[batch_rank * slice(None), i]`, whereas for
        right-continuity, the value is
        `values[batch_rank * slice(None), i + 1]`.
        Default value: `True` which means that the function is left-continuous.
      name: Python `str` name prefixed to ops created by this method.
        Default value: `None` which is mapped to the default name
        `self.name() + _call`.

    Returns:
      A `Tensor` of the same `dtype` as `x` and shape
      `batch_shape + [num_points] + event_shape` containing values of the
      piecewise constant function.
    """
    name = name or self._name  + '_call'
    with tf.name_scope(name):
      x = tf.convert_to_tensor(x, dtype=self.dtype(), name='x')
      batch_shape = tf.shape(self._jump_locations)[:-1]
      x = _try_broadcast_to(x, batch_shape)
      side = 'left' if left_continuous else 'right'
      return _piecewise_constant_function(
          x, self._jump_locations, self._values, self._batch_rank, side=side)

  def integrate(self, x1, x2, name=None):
    """Integrates the piecewise constant function between end points.

    Returns a value of the integral on the interval `[x1, x2]` of a piecewise
    constant function with jump locations and values given by the initializer.

    Args:
      x1: A real `Tensor` of shape `batch_shape + [num_points]`. Left end points
        at which the function has to be integrated.
      x2: A `Tensor` of the same shape and `dtype` as `x1`. Right end points at
        which the function has to be integrated.
      name: Python `str` name prefixed to ops created by this method.
        Default value: `None` which is mapped to the default name
        `self.name() + `_integrate``.

    Returns:
      A `Tensor` of the same `dtype` as `x` and shape
      `batch_shape + [num_points] + event_shape` containing values of the
      integral of the piecewise constant function between `[x1, x2]`.
    """
    name = name or self._name + '_integrate'
    with tf.name_scope(name):
      x1 = tf.convert_to_tensor(x1, dtype=self.dtype(),
                                name='x1')
      x2 = tf.convert_to_tensor(x2, dtype=self.dtype(),
                                name='x2')
      batch_shape = tf.shape(self._jump_locations)[:-1]
      x1 = _try_broadcast_to(x1, batch_shape)
      x2 = _try_broadcast_to(x2, batch_shape)
      return _piecewise_constant_integrate(
          x1, x2, self._jump_locations, self._values, self._batch_rank)


def find_interval_index(query_xs,
                        interval_lower_xs,
                        last_interval_is_closed=False,
                        dtype=None,
                        name=None):
  """Function to find the index of the interval where query points lies.

  Given a list of adjacent half-open intervals [x_0, x_1), [x_1, x_2), ...,
  [x_{n-1}, x_n), [x_n, inf), described by a list [x_0, x_1, ..., x_{n-1}, x_n].
  Return the index where the input query points lie. If x >= x_n, n is returned,
  and if x < x_0, -1 is returned. If `last_interval_is_closed` is set to `True`,
  the last interval [x_{n-1}, x_n] is interpreted as closed (including x_n).

  #### Example

  ```python
  interval_lower_xs = [0.25, 0.5, 1.0, 2.0, 3.0]
  query_xs = [0.25, 3.0, 5.0, 0.0, 0.5, 0.8]
  result = find_interval_index(query_xs, interval_lower_xs)
  # result == [0, 4, 4, -1, 1, 1]
  ```

  Args:
    query_xs: Rank 1 real `Tensor` of any size, the list of x coordinates for
      which the interval index is to be found. The values must be strictly
      increasing.
    interval_lower_xs: Rank 1 `Tensor` of the same shape and dtype as
      `query_xs`. The values x_0, ..., x_n that define the interval starts.
    last_interval_is_closed: If set to `True`, the last interval is interpreted
      as closed.
    dtype: Optional `tf.Dtype`. If supplied, the dtype for `query_xs` and
      `interval_lower_xs`.
      Default value: None which maps to the default dtype inferred from
      `query_xs`.
    name: Optional name of the operation.

  Returns:
    A tensor that matches the shape of `query_xs` with dtype=int32 containing
    the indices of the intervals containing query points. `-1` means the query
    point lies before all intervals and `n-1` means that the point lies in the
    last half-open interval (if `last_interval_is_closed` is `False`) or that
    the point lies to the right of all intervals (if `last_interval_is_closed`
    is `True`).
  """
  name = name or 'find_interval_index'
  with tf.name_scope(name):
    # TODO(b/138988951): add ability to validate that intervals are increasing.
    # TODO(b/138988951): validate that if last_interval_is_closed, input size
    # must be > 1.
    query_xs = tf.convert_to_tensor(query_xs, dtype=dtype)
    dtype = dtype or query_xs.dtype
    interval_lower_xs = tf.convert_to_tensor(interval_lower_xs, dtype=dtype)

    # Result assuming that last interval is half-open.
    indices = tf.searchsorted(interval_lower_xs, query_xs, side='right') - 1

    # Handling the branch if the last interval is closed.
    last_index = tf.shape(interval_lower_xs)[-1] - 1
    last_x = tf.gather(interval_lower_xs, [last_index], axis=-1)
    # should_cap is a tensor true where a cell is true iff indices is the last
    # index at that cell and the query x <= the right boundary of the last
    # interval.
    should_cap = tf.logical_and(
        tf.equal(indices, last_index), tf.less_equal(query_xs, last_x))

    # cap to last_index if the query x is not in the last interval, otherwise,
    # cap to last_index - 1.
    caps = last_index - tf.cast(should_cap, dtype=tf.int32)

    return tf.compat.v1.where(last_interval_is_closed,
                              tf.minimum(indices, caps), indices)


def _piecewise_constant_function(x, jump_locations, values,
                                 batch_rank, side='left'):
  """Computes value of the piecewise constant function."""
  # Initializer already verified that `jump_locations` and `values` have the
  # same shape
  batch_shape = jump_locations.shape.as_list()[:-1]
  # Check that the batch shape of `x` is the same as of `jump_locations` and
  # `values`
  batch_shape_x = x.shape.as_list()[:batch_rank]
  if batch_shape_x != batch_shape:
    raise ValueError('Batch shape of `x` is {1} but should be {0}'.format(
        batch_shape, batch_shape_x))
  if x.shape.as_list()[:batch_rank]:
    no_batch_shape = False
  else:
    no_batch_shape = True
    x = tf.expand_dims(x, 0)
    batch_rank += 1
  # Expand batch size to one if there is no batch shape
  if not batch_shape:
    jump_locations = tf.expand_dims(jump_locations, 0)
    values = tf.expand_dims(values, 0)
  indices = tf.searchsorted(jump_locations, x, side=side)
  res = tf.gather(values, indices, axis=batch_rank, batch_dims=batch_rank)
  if no_batch_shape:
    return tf.squeeze(res, 0)
  else:
    return res


def _piecewise_constant_integrate(x1, x2, jump_locations, values, batch_rank):
  """Integrates piecewise constant function between `x1` and `x2`."""
  # Initializer already verified that `jump_locations` and `values` have the
  # same shape.
  # Expand batch size to one if there is no batch shape.
  if x1.shape.as_list()[:batch_rank]:
    no_batch_shape = False
  else:
    no_batch_shape = True
    x1 = tf.expand_dims(x1, 0)
    x2 = tf.expand_dims(x2, 0)
  if not jump_locations.shape.as_list()[:-1]:
    jump_locations = tf.expand_dims(jump_locations, 0)
    values = tf.expand_dims(values, 0)
    batch_rank += 1
  # Compute integral values between the jump locations
  event_shape = tf.shape(values)[(batch_rank+1):]
  event_rank = values.shape.rank - batch_rank - 1
  num_data_points = tf.shape(values)[batch_rank]
  diff = jump_locations[..., 1:] - jump_locations[..., :-1]
  # Broadcast `diff` to the shape of
  # `batch_shape + [num_data_points - 2] + [1] * sample_rank`.
  for _ in range(event_rank):
    diff = tf.expand_dims(diff, -1)
  slice_indices = batch_rank * [slice(None)]
  slice_indices += [slice(1, num_data_points - 1)]
  integrals = tf.cumsum(values[slice_indices] * diff, batch_rank)
  # Pad integrals with zero values on left and right.
  batch_shape = tf.shape(integrals)[:batch_rank]
  pad_shape = tf.concat([batch_shape, [1], event_shape], axis=0)
  zeros = tf.zeros(pad_shape, dtype=integrals.dtype)
  integrals = tf.concat([zeros, integrals, zeros], axis=batch_rank)
  # Get jump locations and values and the integration end points
  value1, jump_location1, indices_1 = _get_indices_and_values(
      x1, jump_locations, values, 'left', batch_rank)
  value2, jump_location2, indices_2 = _get_indices_and_values(
      x2, jump_locations, values, 'right', batch_rank)
  integrals1 = tf.gather(
      integrals, indices_1, axis=batch_rank, batch_dims=batch_rank)
  integrals2 = tf.gather(
      integrals, indices_2, axis=batch_rank, batch_dims=batch_rank)
  # Broadcast `x1`, `x2`, `jump_location1`, `jump_location2` to the shape
  # `batch_shape + [num_points] + [1] * sample_rank`.
  for _ in range(event_rank):
    x1 = tf.expand_dims(x1, -1)
    x2 = tf.expand_dims(x2, -1)
    jump_location1 = tf.expand_dims(jump_location1, -1)
    jump_location2 = tf.expand_dims(jump_location2, -1)
  # Compute the value of the integrals.
  res = ((jump_location1 - x1) * value1
         + (x2 - jump_location2) * value2
         + integrals2 - integrals1)
  if no_batch_shape:
    return tf.squeeze(res, 0)
  else:
    return res


def _get_indices_and_values(x, jump_locations, values, side,
                            batch_rank):
  """Computes values and jump locations of the piecewise constant function.

  Given `jump_locations` and the `values` on the corresponding segments of the
  piecewise constant function, the function identifies the nearest jump to `x`
  from the right or left (which is determined by the `side` argument) and the
  corresponding value of the piecewise constant function at `x`

  Args:
    x: A real `Tensor` of shape `batch_shape + [num_points]`. Points at which
      the function has to be evaluated.
    jump_locations: A `Tensor` of the same `dtype` as `x` and shape
      `batch_shape + [num_jump_points]`. The locations where the function
      changes its values. Note that the values are expected to be ordered
      along the last dimension.
    values: A `Tensor` of the same `dtype` as `x` and shape
      `batch_shape + [num_jump_points + 1]`. Defines `values[..., i]` on
      `jump_locations[..., i - 1], jump_locations[..., i]`.
    side: A Python string. Whether the function is left- or right- continuous.
      The corresponding values for side should be `left` and `right`.
    batch_rank: A Python scalar stating the batch rank of `x`.

  Returns:
    A tuple of three `Tensor` of the same `dtype` as `x` and shapes
    `batch_shape + [num_points] + event_shape`, `batch_shape + [num_points]`,
    and `batch_shape + [num_points] + [2 * len(batch_shape)]`. The `Tensor`s
    correspond to the values, jump locations at `x`, and the corresponding
    indices used to obtain jump locations via `tf.gather_nd`.
  """
  indices = tf.searchsorted(jump_locations, x, side=side)
  num_data_points = tf.shape(values)[batch_rank] - 2
  if side == 'right':
    indices_jump = indices - 1
    indices_jump = tf.maximum(indices_jump, 0)
  else:
    indices_jump = tf.minimum(indices, num_data_points)
  value = tf.gather(values, indices, axis=batch_rank, batch_dims=batch_rank)
  jump_location = tf.gather(
      jump_locations, indices_jump, axis=batch_rank, batch_dims=batch_rank)
  return value, jump_location, indices_jump


def _try_broadcast_to(x, batch_shape):
  """Broadcasts batch shape of `x` to a `batch_shape` if possible."""
  broadcast_shape = tf.concat([batch_shape, tf.shape(x)[-1:]], axis=0)
  return x + tf.zeros(broadcast_shape, dtype=x.dtype)


def convert_to_tensor_or_func(x, dtype=None, name=None):
  """Returns either a `PiecewiseConstantFunc` or x converted to a `Tensor`.

  Converts the input argument into a `Tensor` unless the input is of type
  `PiecewiseConstantFunc` when the input is returned unchanged.  This function
  helps pricing models to preprocess inputs that can be either constant or time
  dependent as represented by a PiecewiseConstantFunc.

  Example:
    class GeometricBrownianMotion(object)
      def __init__(self, mu, sigma):
        self._mu, self._mu_is_constant = piecewise.convert_to_tensor_or_func(
          mu)
        self._sigma, self._sigma_is_constant =
          piecewise.convert_to_tensor_or_func(sigma)

  Args:
    x: Either a 'Tensor' or 'PiecewiseConstantFunc'.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this class.
      Default value: `None` which maps to the default name
      'PiecewiseConstantFunc_tensor_or_func'.
  Returns:
    A tuple (y, flag) where y is either a Tensor or PiecewiseConstantFunc
    and flag which is False if x is of type PiecewiseConstantFunc and True
    otherwise.
  """
  name = name or ('PiecewiseConstantFunc_tensor_or_func')
  if isinstance(x, PiecewiseConstantFunc):
    return x, False
  return tf.convert_to_tensor(x, dtype=dtype, name=name), True
