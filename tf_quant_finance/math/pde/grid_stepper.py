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
"""Scaffolding for finite difference solvers for Kolmogorov backward equation.

The [Kolmogorov Backward equation (KBE)](https://en.wikipedia.org/wiki/
Kolmogorov_backward_equations_(diffusion)) is a standard workhorse for pricing
financial derivatives. The simplest of this class of equations is the
[Black-Scholes equation](https://en.wikipedia.org/
wiki/Black%E2%80%93Scholes_equation). BS equation is the KBE associated to
the lognormal Ito process in the risk neutral measure. For more complex models
typically used in finance, there are no analytic solutions for commonly traded
instruments and one has to resort to either Monte Carlo or Finite Difference
(FD) methods to solve the PDE. This module provides convenience classes to
manage FD method for solving the equation.

The main class of this module is the `BackwardGridStepper`. It provides methods
to set final conditions and perform manipulations of the grid values
at intermediate times (for example, to set a barrier condition). Note that this
class does not perform the actual stepping itself so is not tied to any specific
finite difference method. Those have to be supplied by the user.

TODO(b/139954627): Increase test coverage for `BackwardGridStepper`.

TODO(b/139955815): Add namescopes for BackwardGridStepper methods.

TODO(b/139953871): Add functionality to vary the physical grid.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

_GridStepperState = collections.namedtuple(
    '_GridStepperState',
    [
        'time',  # The current time.
        'time_step',  # The next time step to take. Always positive or zero.
        'num_steps_performed',  # Number of steps performed so far.
        'coordinate_grid',  # The coordinates of the points on the grid.
        'value_grid'  # The function values on the grid.
    ])


class GridStepperState(_GridStepperState):
  """Represents the state of the grid stepper at a particular time."""

  def copy(self, **kwargs):
    """Creates a copy of the state with some fields overridden.

    Args:
      **kwargs: A set of keyword args whose attributes are to be overridden in
        the copy.

    Returns:
      A new instance of `GridStepperState` with some fields overridden.
    """
    cls = type(self)
    attribute_dict = super(cls, self)._asdict()
    attribute_dict.update(kwargs)
    return cls(**attribute_dict)


class BackwardGridStepper(object):
  """Steps back a final value to time 0."""

  def __init__(self,
               final_time,
               one_step_kernel,
               grid_spec,
               time_step=None,
               time_step_fn=None,
               value_dim=1,
               dtype=None,
               name=None):
    """Initializes the stepper.

    Args:
      final_time: Scalar real `Tensor`. The time of the final payout. Must be
        positive. The solver is initialized to the final time.
      one_step_kernel: A callable that moves the solution from time `t` to
        `t-dt` for a given `dt`. It accepts an instance of the
        `GridStepperState` and returns a `Tensor` of the same shape as
        `GridStepperState.value_grid` (i.e. the value_grid). The value grid is a
        `Tensor` of shape `[k1, k2, ... kn, p]` where `n` is the dimension of
        the grid, `ki` is the grid size along axis `i` and `p` is the batch of
        values. Usually, one would do independent propagation of each batch of
        values but the grid stepper is agnostic to this.
      grid_spec: An iterable convertible to a tuple containing at least the
        attributes named 'grid', 'dim' and 'sizes'. For a full description of
        the fields and expected types, see `grids.GridSpec` which provides the
        canonical specification of this object.
      time_step: A real positive scalar `Tensor` or None. The discretization
        parameter along the time dimension. Either this argument or the
        `time_step_fn` must be specified. It is an error to specify both.
      time_step_fn: A callable accepting an instance of `GridStepperState` and
        returning the size of the next time step as a real scalar tensor. This
        argument allows usage of a non-constant time step while stepping back.
        If not specified, the `time_step` parameter must be specified. It is an
        error to specify both.
      value_dim: A positive Python int. The batch size of values to be
        propagated simultaneously.
      dtype: Optional Tensorflow dtype. The dtype of the values. If not
        specified, the dtype of the coordinate grid is used.
      name: Python str. The name prefixed to the ops created by this class. If
        not supplied, the default name 'grid_stepper' is used.

    Raises:
      ValueError: If any of the following conditions are violated.
        1) Exactly one of `time_step` or `time_step_fn` is specified.
        2) The value_dim is greater than or equal to 1.
    """
    self._name = name
    if (time_step is None) == (time_step_fn is None):
      raise ValueError('Must specify exactly one of `time_step` or '
                       '`time_step_fn`.')
    if value_dim < 1:
      raise ValueError('The value dimension must be at least 1.')

    with tf.compat.v1.name_scope(name, 'grid_stepper_init',
                                 [final_time, grid_spec, time_step]):
      if dtype is None:
        dtype = tf.convert_to_tensor(grid_spec.grid).dtype
      self._dtype = dtype
      self._final_time = tf.convert_to_tensor(
          final_time, dtype=dtype, name='final_time')
      self._kernel = one_step_kernel
      self._grid_spec = grid_spec
      self._dim = self._grid_spec.dim
      if time_step is not None:
        time_step = tf.convert_to_tensor(
            time_step, dtype=dtype, name='time_step')
      self._time_step_fn = time_step_fn or (lambda _: time_step)
      self._value_dim = value_dim
      self._value_grid_shape = [self._value_dim] + self._grid_spec.sizes
      value_grid = tf.zeros(self._value_grid_shape, dtype=dtype)
      state = GridStepperState(
          time=self._final_time,
          time_step=None,
          num_steps_performed=tf.constant(0, dtype=tf.int32),
          coordinate_grid=grid_spec,
          value_grid=value_grid)
      next_time_step = self._time_step_fn(state)
      self._state = state.copy(time_step=next_time_step)

  def state(self):
    """The current state of the grid stepper."""
    return self._state

  def value_grid(self):
    """Returns the current value grid.

    The shape of the value grid is [value dimension] + shape of coordinate grid.
    The value dimension corresponds to a batch of values.

    Returns:
      The current value grid as a real `Tensor` of shape
      [value batch size] + shape of coordinate grid.
    """
    return self._state.value_grid

  @property
  def value_grid_shape(self):
    return self._value_grid_shape

  @property
  def dtype(self):
    """The dtype to use for the grid."""
    return self._dtype

  def transform_values(self, value_transform_fn):
    """Transforms the current grid values by invoking the supplied callable.

    Args:
      value_transform_fn: A Python callable that accepts an instance of
        `GridStepperState` and returns a `Tensor` of the same shape and dtype as
        the current value grid (i.e. `GridStepperState.value_grid`). The
        supplied argument is the current state of the grid.
    """
    value_grid = value_transform_fn(self._state)
    if value_grid.dtype != self._state.value_grid.dtype:
      raise ValueError('Expected value_transform_fn to return dtype %s. '
                       'Found %s.' %
                       (self._state.value_grid.dtype, value_grid.dtype))
    self._state = self._state.copy(value_grid=value_grid)

  def step_back(self):
    """Performs a single step back in time."""
    raw_time_step = self._state.time_step
    # Check that the raw time step won't put us below time 0.
    time_step = tf.minimum(raw_time_step, self._state.time)
    next_value_grid = self._kernel(self._state.copy(time_step=time_step))
    next_state = self._state.copy(
        time=self._state.time - time_step,
        num_steps_performed=self._state.num_steps_performed + 1,
        value_grid=next_value_grid)
    self._state = next_state.copy(time_step=self._time_step_fn(next_state))

  def step_back_to_time(self,
                        time,
                        value_transform_fn=None,
                        swap_memory=True,
                        extra_time_points=None):
    """Propagates the grid back in time to the supplied time.

    Args:
      time: Positive scalar `Tensor`. The time to step back to. If the time is
        greater than the current time on the grid, this method will have no
        effect. If the time is less than 0, the step back is performed only to
        time 0. The dtype of the tensor should match the dtype of the grid
        stepper (see,  GridStepper.dtype attribute).
      value_transform_fn: Python callable or None. If supplied, it should accept
        the current grid state (instance of `GridStepperState and return a
        transformed value grid as a `Tensor` of the same shape as
        `state.value_grid`. Note that this function should have no side effects.
        If it does, it will lead to bugs in the graph mode. This is because the
        function will only be invoked at graph building time and not through the
        actual backward propagation and hence any side-effects will not be
        executed after the first call. Also note that the transform is applied
        *after* the kernel is applied to the state. This means that the
        transform will not affect the starting values on the grid but will
        affect the ending values.
      swap_memory: Whether GPU-CPU memory swap is enabled for this op. See
        equivalent flag in `tf.while_loop` documentation for more details.
        Useful when computing a gradient of the op.
      extra_time_points: Positive rank 1 `Tensor`. The time points where the
        stepper would stop by before continue stepping back to reduce numerical
        errors. The dtype of this tensor should match the dtype of the grid
        stepper. And the values need not be sorted.
    """
    time = tf.convert_to_tensor(time, dtype=self._dtype, name='time')
    target_time = tf.maximum(time, 0)

    if extra_time_points is None:
      extra_time_points = tf.constant([], dtype=self._dtype)
      index = tf.constant(-1)
    else:
      extra_time_points = tf.convert_to_tensor(
          extra_time_points, dtype=self._dtype, name='extra_time_points')
      extra_time_points = tf.sort(extra_time_points)
      length = tf.shape(extra_time_points)[0]
      # A time point is valid if it's smaller than final time as the direction
      # of the movement of this stepper is backward.
      is_time_points_valid = extra_time_points < self._final_time

      def find_largest_valid_index():
        return tf.argmax(
            tf.cast(is_time_points_valid, tf.int32) * tf.range(length),
            output_type=tf.int32)

      # Find the largest index which has a valid time point. If there's no valid
      # time points, set index to -1.
      index = tf.cond(
          tf.reduce_any(is_time_points_valid), find_largest_valid_index,
          lambda: -1)

    def _loop_cond(should_stop, state, index):
      del state, index
      return tf.logical_not(should_stop)

    def _loop_body(should_stop, state, index):
      """Propagates the grid backward in time."""
      raw_time_step = state.time_step
      time_step = tf.minimum(raw_time_step, state.time - target_time)

      next_time_point = tf.cond(index >= 0, lambda: extra_time_points[index],
                                lambda: target_time)
      time_to_next_time_point = state.time - next_time_point
      index = tf.where(time_to_next_time_point <= time_step, index - 1, index)
      time_step = tf.minimum(time_step, time_to_next_time_point)

      updated_state = state.copy(time_step=time_step)
      next_value_grid = self._kernel(updated_state)
      if value_transform_fn is not None:
        updated_state = updated_state.copy(value_grid=next_value_grid)
        next_value_grid = value_transform_fn(updated_state)
      next_state = state.copy(
          time=state.time - time_step,
          num_steps_performed=self._state.num_steps_performed + 1,
          value_grid=next_value_grid)
      next_time_step = self._time_step_fn(next_state)
      should_stop = next_state.time <= target_time
      return should_stop, next_state.copy(time_step=next_time_step), index

    initial_args = (self._state.time <= target_time, self._state, index)
    _, next_state, index = tf.compat.v1.while_loop(
        _loop_cond, _loop_body, initial_args, swap_memory=swap_memory)
    self._state = next_state
