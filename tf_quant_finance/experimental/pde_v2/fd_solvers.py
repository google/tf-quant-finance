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
"""Helper functions for solving linear parabolic PDEs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def step_back(start_time,
              end_time,
              coord_grid,
              values_grid,
              num_steps=None,
              time_step=None,
              one_step_fn=None,
              boundary_conditions=None,
              values_transform_fn=None,
              quadratic_coeff_fn=None,
              linear_coeff_fn=None,
              constant_coeff_fn=None,
              maximum_steps=None,
              swap_memory=True,
              dtype=None,
              name=None):
  """Evolves a grid of function values backward in time.

  Evolves a discretized solution of following second order linear
  partial differential equation:

  ```None
    V_t + Sum[mu_i(t, x) V_i, 1<=i<=n] +
      (1/2) Sum[ D_{ij} V_{ij}, 1 <= i,j <= n] - r(t, x) V = 0
  ```
  from time `t0` to time `t1<t0` (i.e. backwards in time).
  The solution `V(t,x)` is assumed to be discretized on an `n`-dimensional
  rectangular grid. A rectangular grid, G, in n-dimensions may be described
  by specifying the coordinates of the points along each axis. For example,
  a 2 x 4 grid in two dimensions can be specified by taking the cartesian
  product of [1, 3] and [5, 6, 7, 8] to yield the grid points with
  coordinates: `[(1, 5), (1, 6), (1, 7), (1, 8), (3, 5) ... (3, 8)]`.

  This function allows batching of solutions. In this context, batching means
  the ability to represent and evolve multiple independent functions `V`
  (e.g. V1, V2 ...) simultaneously. A single discretized solution is specified
  by stating its values at each grid point. This can be represented as a
  `Tensor` of shape [d1, d2, ... dn] where di is the grid size along the `i`th
  axis. A batch of such solutions is represented by a `Tensor` of shape:
  [K, d1, d2, ... dn] where `K` is the batch size. This method only requires
  that the input parameter `values_grid` be broadcastable with shape
  [K, d1, ... dn].

  The evolution of the solution from `t0` to `t1` is done by discretizing the
  differential equation to a difference equation along the spatial and
  temporal axes. The temporal discretization is given by a (sequence of)
  time steps [dt_1, dt_2, ... dt_k] such that the sum of the time steps is
  equal to the total time step `t0 - t1`. If a uniform time step is used,
  it may equivalently be specified by stating the number of steps (n_steps)
  to take. This method provides both options via the `time_step`
  and `num_steps` parameters.

  The mapping between the arguments of this method and the above
  equation are described in the Args section below.

  Args:
    start_time: Real positive scalar `Tensor`. The start time of the grid.
      Corresponds to time `t0` above.
    end_time: Real scalar `Tensor` smaller than the `start_time` and greater
      than zero. The time to step back to. Corresponds to time `t1` above.
    coord_grid: List of `n` rank 1 real `Tensor`s. `n` is the dimension of the
      domain. The i-th `Tensor` has shape, `[d_i]` where `d_i` is the size of
      the grid along axis `i`. The coordinates of the grid points. Corresponds
      to the spatial grid `G` above.
    values_grid: Real `Tensor` containing the function values at time
      `start_time` which have to be stepped back to time `end_time`. The shape
      of the `Tensor` must broadcast with `[K, d_1, d_2, ..., d_n]`. The first
      axis of size `K` is the values batch dimension and allows multiple
      functions (with potentially different boundary/final conditions) to be
      stepped back simultaneously.
    num_steps: Positive int scalar `Tensor`. The number of time steps to take
      when moving from `start_time` to `end_time`. Either this argument or the
      `time_step` argument must be supplied (but not both). If num steps is
      `k>=1`, uniform time steps of size `(t0 - t1)/k` are taken to evolve the
      solution from `t0` to `t1`. Corresponds to the `n_steps` parameter above.
    time_step: The time step to take. Either this argument or the `num_steps`
      argument must be supplied (but not both). The type of this argument may
      be one of the following (in order of generality): (a) None in which case
        `num_steps` must be supplied. (b) A positive real scalar `Tensor`. The
        maximum time step to take. If the value of this argument is `dt`, then
        the total number of steps taken is N = (t0 - t1) / dt rounded up to the
        nearest integer. The first N-1 steps are of size dt and the last step is
        of size `t0 - t1 - (N-1) * dt`. (c) A callable accepting the current
        time and returning the size of the step to take. The input and the
        output are real scalar `Tensor`s.
    one_step_fn: The transition kernel. A callable that consumes the following
      arguments by keyword:
        1. 'time': Current time
        2. 'next_time': The next time to step to. For the backwards in time
          evolution, this time will be smaller than the current time.
        3. 'coord_grid': The coordinate grid.
        4. 'values_grid': The values grid.
        5. 'quadratic_coeff': A callable returning the quadratic coefficients of
          the PDE (i.e. `(1/2)D_{ij}(t, x)` above). The callable accepts the
          time and  coordinate grid as keyword arguments and returns a `Tensor`
          with shape that broadcasts with `[dim, dim]`.
        6. 'linear_coeff': A callable returning the linear coefficients of the
          PDE (i.e. `mu_i(t, x)` above). Accepts time and coordinate grid as
          keyword arguments and returns a `Tensor` with shape that broadcasts
          with `[dim]`.
        7. 'constant_coeff': A callable returning the coefficient of the linear
          homogenous term (i.e. `r(t,x)` above). Same spec as above. The
          `one_step_fn` callable returns a 2-tuple containing the next
          coordinate grid, next values grid.
    boundary_conditions: The boundary conditions.
    values_transform_fn: An optional callable applied to transform the solution
      values at each time step. The callable is invoked after the time step has
      been performed. The callable should accept the time of the grid, the
      coordinate grid and the values grid and should return the values grid. All
      input arguments to be passed by keyword.
    quadratic_coeff_fn: The quadratic coefficient.
    linear_coeff_fn: The linear coefficient.
    constant_coeff_fn: The constant coefficient.
    maximum_steps: Optional int `Tensor`. The maximum number of time steps that
      might be taken. This argument is only used if the `num_steps` is not used
      and `time_step` is a callable otherwise it is ignored. It is useful to
      supply this argument to ensure that the time stepping loop can be
      optimized. If the argument is supplied and used, the time loop with
      execute at most these many steps so it is important to ensure that this
      parameter is an upper bound on the number of expected steps.
    swap_memory: Whether GPU-CPU memory swap is enabled for this op. See
      equivalent flag in `tf.while_loop` documentation for more details. Useful
      when computing a gradient of the op.
    dtype: The dtype to use.
    name: The name to give to the ops.
      Default value: None which means `step_back` is used.

  Returns:
    The final time, final coordinate grid and the final values grid.

  Raises:
    ValueError if neither num steps nor time steps are provided or if both
    are provided.
  """
  if (num_steps is None) == (time_step is None):
    raise ValueError('Exactly one of num_steps or time_step'
                     ' should be supplied.')

  with tf.name_scope(
      name,
      default_name='step_back',
      values=[
          start_time,
          end_time,
          coord_grid,
          values_grid,
          num_steps,
          time_step,
      ]):
    start_time = tf.convert_to_tensor(
        start_time, dtype=dtype, name='start_time')
    end_time = tf.math.maximum(
        tf.math.minimum(
            tf.convert_to_tensor(end_time, dtype=dtype, name='end_time'),
            start_time), 0)

    time_step_fn, est_max_steps = _get_time_steps_info(start_time, end_time,
                                                       num_steps, time_step)
    if est_max_steps is None and maximum_steps is not None:
      est_max_steps = maximum_steps

    def loop_cond(should_stop, time, x_grid, f_grid):
      del time, x_grid, f_grid
      return tf.logical_not(should_stop)

    def loop_body(should_stop, time, x_grid, f_grid):
      """Propagates the grid backward in time."""
      del should_stop
      next_should_stop, t_next = time_step_fn(time)
      next_xs, next_fs = one_step_fn(
          time=time,
          next_time=t_next,
          coord_grid=x_grid,
          values_grid=f_grid,
          boundary_conditions=boundary_conditions,
          quadratic_coeff=quadratic_coeff_fn,
          linear_coeff=linear_coeff_fn,
          constant_coeff=constant_coeff_fn)

      if values_transform_fn is not None:
        next_xs, next_fs = values_transform_fn(t_next, next_xs, next_fs)
      return next_should_stop, t_next, next_xs, next_fs

      # If the start time is already equal to or before the end time,
      # no stepping is needed.

    should_already_stop = (start_time <= end_time)
    initial_args = (should_already_stop, start_time, coord_grid, values_grid)
    _, final_time, final_coords, final_values = tf.compat.v1.while_loop(
        loop_cond,
        loop_body,
        initial_args,
        swap_memory=swap_memory,
        max_iterations=est_max_steps)
    return final_time, final_coords, final_values


def _is_callable(var_or_fn):
  """Returns whether an object is callable or not."""
  # Python 2.7 as well as Python 3.x with x > 2 support 'callable'.
  # In between, callable was removed hence we need to do a more expansive check
  if hasattr(var_or_fn, '__call__'):
    return True
  try:
    return callable(var_or_fn)
  except NameError:
    return False


def _get_time_steps_info(start_time, end_time, num_steps, time_step):
  """Creates a callable to step through time and estimates the max steps."""
  # Assume end_time <= start_time
  dt = None
  estimated_max_steps = None
  if num_steps is not None:
    dt = (start_time - end_time) / tf.cast(num_steps, dtype=start_time.dtype)
    estimated_max_steps = num_steps
  if time_step is not None and not _is_callable(time_step):
    dt = time_step
    estimated_max_steps = tf.cast(
        tf.math.ceil((start_time - end_time) / dt), dtype=tf.int32)
  if dt is not None:
    raw_time_step_fn = lambda _: dt
  else:
    raw_time_step_fn = time_step

  def step_fn(t):
    # t is the current time.
    # t_next is the next time
    dt = raw_time_step_fn(t)
    t_next = tf.math.maximum(end_time, t - dt)
    return t_next > end_time, t_next

  return step_fn, estimated_max_steps
