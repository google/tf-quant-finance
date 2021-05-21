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
"""Functions for solving linear parabolic PDEs."""

import tensorflow.compat.v2 as tf
from tf_quant_finance.math.pde.steppers.douglas_adi import douglas_adi_step
from tf_quant_finance.math.pde.steppers.oscillation_damped_crank_nicolson import oscillation_damped_crank_nicolson_step


def solve_backward(start_time,
                   end_time,
                   coord_grid,
                   values_grid,
                   num_steps=None,
                   start_step_count=0,
                   time_step=None,
                   one_step_fn=None,
                   boundary_conditions=None,
                   values_transform_fn=None,
                   second_order_coeff_fn=None,
                   first_order_coeff_fn=None,
                   zeroth_order_coeff_fn=None,
                   inner_second_order_coeff_fn=None,
                   inner_first_order_coeff_fn=None,
                   maximum_steps=None,
                   swap_memory=True,
                   dtype=None,
                   name=None):
  """Evolves a grid of function values backwards in time according to a PDE.

  Evolves a discretized solution of following second order linear
  partial differential equation:

  ```None
    dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 0 <= i, j <=n-1] +
       Sum[b_i d(B_i V)/dx_i, 0 <= i <= n-1] + c V = 0.
  ```
  from time `t0` to time `t1<t0` (i.e. backwards in time). Here `a_ij`,
  `A_ij`, `b_i`, `B_i` and `c` are coefficients that may depend on spatial
  variables `x` and time `t`.

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

  #### Example. European call option pricing.
  ```python
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff
  pde = tff.math.pde

  num_equations = 2  # Number of PDE
  num_grid_points = 1024  # Number of grid points
  dtype = tf.float64

  # Build a log-uniform grid
  s_min = 0
  s_max = 200
  grid = pde.grids.uniform_grid(minimums=[s_min],
                                maximums=[s_max],
                                sizes=[num_grid_points],
                                dtype=dtype)

  # Specify volatilities and interest rates for the options
  strike = tf.constant([[50], [100]], dtype)
  volatility = tf.constant([[0.3], [0.15]], dtype)
  rate = tf.constant([[0.01], [0.03]], dtype)
  expiry = 1.0

  # For batching multiple PDEs, we need to stack the grid values
  # so that final_values[i] is the grid for the ith strike.
  s = grid[0]
  final_value_grid = tf.nn.relu(s - strike)

  # Define parabolic equation coefficients. In this case the coefficients
  # can be computed exactly but the same functions as below can be used to
  # get approximate values for general case.
  # We again need to use `tf.meshgrid` to batch the coefficients.
  def second_order_coeff_fn(t, grid):
    del t
    s = grid[0]
    return [[volatility**2 * s**2 / 2]]

  def first_order_coeff_fn(t, grid):
    del t
    s = grid[0]
    return [rate * s]

  def zeroth_order_coeff_fn(t, grid):
    del t, grid
    return -rate

  @pde.boundary_conditions.dirichlet
  def lower_boundary_fn(t, grid):
    del t, grid
    return 0

  @pde.boundary_conditions.dirichlet
  def upper_boundary_fn(t, grid):
    del grid
    return tf.squeeze(s_max - strike * tf.exp(-rate * (expiry - t)))


  # Estimate European call option price:
  estimate = pde.fd_solvers.solve_backward(
    start_time=expiry,
    end_time=0,
    coord_grid=grid,
    values_grid=final_value_grid,
    time_step=0.01,
    boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
    second_order_coeff_fn=second_order_coeff_fn,
    first_order_coeff_fn=first_order_coeff_fn,
    zeroth_order_coeff_fn=zeroth_order_coeff_fn,
    dtype=dtype)[0]

  # Extract estimates for some of the grid locations and compare to the
  # true option price:
  value_grid_first_option = estimate[0, :]
  value_grid_second_option = estimate[1, :]

  # As an alternative, user can use a default BC for the lower bound by setting
  # lower_boundary_fn to `None`, which corresponds to
  # `V_t - r V_s - rV = 0`.
  # Estimate European call option price using default BC for the lower bound:
  estimate_with_default_bc = pde.fd_solvers.solve_backward(
    start_time=expiry,
    end_time=0,
    coord_grid=grid,
    values_grid=final_value_grid,
    time_step=0.01,
    boundary_conditions=[(None, upper_boundary_fn)],
    second_order_coeff_fn=second_order_coeff_fn,
    first_order_coeff_fn=first_order_coeff_fn,
    zeroth_order_coeff_fn=zeroth_order_coeff_fn,
    dtype=dtype)[0]
  ```

  See more examples in `pde_solvers.pdf`.

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
      `start_time` which have to be evolved to time `end_time`. The shape of the
      `Tensor` must broadcast with `B + [d_1, d_2, ..., d_n]`. `B` is the batch
      dimensions (one or more), which allow multiple functions (with potentially
      different boundary/final conditions and PDE coefficients) to be evolved
      simultaneously.
    num_steps: Positive int scalar `Tensor`. The number of time steps to take
      when moving from `start_time` to `end_time`. Either this argument or the
      `time_step` argument must be supplied (but not both). If num steps is
      `k>=1`, uniform time steps of size `(t0 - t1)/k` are taken to evolve the
      solution from `t0` to `t1`. Corresponds to the `n_steps` parameter above.
    start_step_count: A scalar integer `Tensor`. Number of steps performed so
      far.
    time_step: The time step to take. Either this argument or the `num_steps`
      argument must be supplied (but not both). The type of this argument may
      be one of the following (in order of generality):
        (a) None in which case `num_steps` must be supplied.
        (b) A positive real scalar `Tensor`. The maximum time step to take.
          If the value of this argument is `dt`, then the total number of steps
          taken is N = (t0 - t1) / dt rounded up to the nearest integer. The
          first N-1 steps are of size dt and the last step is of size
          `t0 - t1 - (N-1) * dt`.
        (c) A callable accepting the current time and returning the size of the
          step to take. The input and the output are real scalar `Tensor`s.
    one_step_fn: The transition kernel. A callable that consumes the following
      arguments by keyword:
        1. 'time': Current time
        2. 'next_time': The next time to step to. For the backwards in time
          evolution, this time will be smaller than the current time.
        3. 'coord_grid': The coordinate grid.
        4. 'values_grid': The values grid.
        5. 'second_order_coeff_fn': Callable returning the coefficients of the
          second order terms of the PDE. See the spec of the
          `second_order_coeff_fn` argument below.
        6. 'first_order_coeff_fn': Callable returning the coefficients of the
          first order terms of the PDE. See the spec of the
          `first_order_coeff_fn` argument below.
        7. 'zeroth_order_coeff_fn': Callable returning the coefficient of the
          zeroth order term of the PDE. See the spec of the
          `zeroth_order_coeff_fn` argument below.
        8. 'num_steps_performed': A scalar integer `Tensor`. Number of steps
          performed so far.
       The callable should return a sequence of two `Tensor`s. The first one
       is a `Tensor` of the same `dtype` and `shape` as `coord_grid` and
       represents a new coordinate grid after one iteration. The second `Tensor`
       is of the same shape and `dtype` as`values_grid` and represents an
       approximate solution of the equation after one iteration.
       Default value: None, which means Crank-Nicolson scheme with oscillation
       damping is used for 1D problems, and Douglas ADI scheme with `theta=0.5`
       - for multidimensional problems.
    boundary_conditions: The boundary conditions. Only rectangular boundary
      conditions are supported. A list of tuples of size `n` (space dimension
      of the PDE). The elements of the Tuple can be either a Python Callable or
      `None` representing the boundary conditions at the minimum and maximum
      values of the spatial variable indexed by the position in the list. E.g.,
      for `n=2`, the length of `boundary_conditions` should be 2,
      `boundary_conditions[0][0]` describes the boundary `(y_min, x)`, and
      `boundary_conditions[1][0]`- the boundary `(y, x_min)`. `None` values mean
      that the second order terms for that dimension on the boundary are assumed
      to be zero, i.e., if `boundary_conditions[k][0]` is None,
      'dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <= n, i!=k+1, j!=k+1] +
         Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.'
      For not `None` values, the boundary conditions are accepted in the form
      `alpha(t, x) V + beta(t, x) V_n = gamma(t, x)`, where `V_n` is the
      derivative with respect to the exterior normal to the boundary.
      Each callable receives the current time `t` and the `coord_grid` at the
      current time, and should return a tuple of `alpha`, `beta`, and `gamma`.
      Each can be a number, a zero-rank `Tensor` or a `Tensor` whose shape is
      the grid shape with the corresponding dimension removed.
      For example, for a two-dimensional grid of shape `(b, ny, nx)`, where `b`
      is the batch size, `boundary_conditions[0][i]` with `i = 0, 1` should
      return a tuple of either numbers, zero-rank tensors or tensors of shape
      `(b, nx)`. Similarly for `boundary_conditions[1][i]`, except the tensor
      shape should be `(b, ny)`. `alpha` and `beta` can also be `None` in case
      of Neumann and Dirichlet conditions, respectively.
      Default value: `None`. Unlike setting `None` to individual elements of
      `boundary_conditions`, setting the entire `boundary_conditions` object to
      `None` means Dirichlet conditions with zero value on all boundaries are
      applied.
    values_transform_fn: An optional callable applied to transform the solution
      values at each time step. The callable is invoked after the time step has
      been performed. The callable should accept the time of the grid, the
      coordinate grid, and the values grid and should return a tuple of the
      the coordinate grid and updated value grid.
    second_order_coeff_fn: Callable returning the coefficients of the second
      order terms of the PDE (i.e. `a_{ij}(t, x)` above) at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `coord_grid`: a `Tensor` representing a grid of locations `r` at which
          the coefficient should be evaluated.
      Returns the object `a` such that `a[i][j]` is defined and
      `a[i][j]=a_{ij}(r, t)`, where `0 <= i < n_dims` and `i <= j < n_dims`.
      For example, the object may be a list of lists or a rank 2 Tensor.
      `a[i][j]` is assumed to be symmetrical, and only the elements with
      `j >= i` will be used, so elements with `j < i` can be `None`.
      Each `a[i][j]` should be a Number, a `Tensor` broadcastable to the shape
      of `coord_grid`, or `None` if corresponding term is absent in the
      equation. Also, the callable itself may be None, meaning there are no
      second-order derivatives in the equation.
      For example, for a 2D equation with the following second order terms
      ```
      a_xx V_xx + 2 a_xy V_xy + a_yy V_yy
      ```
       the callable may return either
      `[[a_yy, a_xy], [a_xy, a_xx]]` or `[[a_yy, a_xy], [None, a_xx]]`.
      Default value: None. If both `second_order_coeff_fn` and
        `inner_second_order_coeff_fn` are None, it means the second-order term
        is absent. If only one of them is `None`, it is assumed to be `1`.
    first_order_coeff_fn: Callable returning the coefficients of the
      first order terms of the PDE (i.e. `mu_i(t, r)` above) evaluated at given
      time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Returns a list or an 1D `Tensor`, `i`-th element of which represents
      `b_i(t, r)`. Each element is a `Tensor` broadcastable to the shape of
      `locations_grid`, or None if corresponding term is absent in the
      equation. The callable itself may be None, meaning there are no
      first-order derivatives in the equation.
      Default value: None. If both `first_order_coeff_fn` and
        `inner_first_order_coeff_fn` are None, it means the first-order term is
        absent. If only one of them is `None`, it is assumed to be `1`.
    zeroth_order_coeff_fn: Callable returning the coefficient of the
      zeroth order term of the PDE (i.e. `c(t, r)` above) evaluated at given
      time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Should return a `Tensor` broadcastable to the shape of `locations_grid`.
      May return None or be None if the shift term is absent in the equation.
      Default value: None, meaning absent zeroth order term.
    inner_second_order_coeff_fn: Callable returning the coefficients under the
      second derivatives (i.e. `A_ij(t, x)` above) at given time `t`. The
      requirements are the same as for `second_order_coeff_fn`.
    inner_first_order_coeff_fn: Callable returning the coefficients under the
      first derivatives (i.e. `B_i(t, x)` above) at given time `t`. The
      requirements are the same as for `first_order_coeff_fn`.
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
      Default value: None, which means dtype will be inferred from
      `values_grid`.
    name: The name to give to the ops.
      Default value: None which means `solve_backward` is used.

  Returns:
    The final values grid, final coordinate grid, final time and number of steps
    performed.

  Raises:
    ValueError if neither num steps nor time steps are provided or if both
    are provided.
  """
  values_grid = tf.convert_to_tensor(values_grid, dtype=dtype)
  start_time = tf.convert_to_tensor(
      start_time, dtype=values_grid.dtype, name='start_time')
  end_time = tf.math.maximum(
      tf.math.minimum(
          tf.convert_to_tensor(end_time, dtype=values_grid.dtype,
                               name='end_time'),
          start_time), 0)
  return _solve(_time_direction_backward_fn,
                start_time,
                end_time,
                coord_grid,
                values_grid,
                num_steps,
                start_step_count,
                time_step,
                one_step_fn,
                boundary_conditions,
                values_transform_fn,
                second_order_coeff_fn,
                first_order_coeff_fn,
                zeroth_order_coeff_fn,
                inner_second_order_coeff_fn,
                inner_first_order_coeff_fn,
                maximum_steps,
                swap_memory,
                name or 'solve_backward')


def solve_forward(start_time,
                  end_time,
                  coord_grid,
                  values_grid,
                  num_steps=None,
                  start_step_count=0,
                  time_step=None,
                  one_step_fn=None,
                  boundary_conditions=None,
                  values_transform_fn=None,
                  second_order_coeff_fn=None,
                  first_order_coeff_fn=None,
                  zeroth_order_coeff_fn=None,
                  inner_second_order_coeff_fn=None,
                  inner_first_order_coeff_fn=None,
                  maximum_steps=None,
                  swap_memory=True,
                  dtype=None,
                  name=None):
  """Evolves a grid of function values forward in time according to a PDE.

  Evolves a discretized solution of following second order linear
  partial differential equation:

  ```None
    dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <=n] +
       Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.
  ```
  from time `t0` to time `t1 > t0` (i.e. forward in time). Here `a_ij`,
  `A_ij`, `b_i`, `B_i` and `c` are coefficients that may depend on spatial
  variables `x` and time `t`.

  See more details in `solve_backwards()`: other than the forward time
  direction, the specification is the same.

  Args:
    start_time: Real scalar `Tensor`. The start time of the grid.
      Corresponds to time `t0` above.
    end_time: Real scalar `Tensor` larger than the `start_time`.
       The time to evolve forward to. Corresponds to time `t1` above.
    coord_grid: List of `n` rank 1 real `Tensor`s. `n` is the dimension of the
      domain. The i-th `Tensor` has shape, `[d_i]` where `d_i` is the size of
      the grid along axis `i`. The coordinates of the grid points. Corresponds
      to the spatial grid `G` above.
    values_grid: Real `Tensor` containing the function values at time
      `start_time` which have to be evolved to time `end_time`. The shape of the
      `Tensor` must broadcast with `B + [d_1, d_2, ..., d_n]`. `B` is the batch
      dimensions (one or more), which allow multiple functions (with potentially
      different boundary/final conditions and PDE coefficients) to be evolved
      simultaneously.
    num_steps: Positive int scalar `Tensor`. The number of time steps to take
      when moving from `start_time` to `end_time`. Either this argument or the
      `time_step` argument must be supplied (but not both). If num steps is
      `k>=1`, uniform time steps of size `(t1 - t0)/k` are taken to evolve the
      solution from `t0` to `t1`. Corresponds to the `n_steps` parameter above.
    start_step_count: A scalar integer `Tensor`. Number of steps performed so
      far.
    time_step: The time step to take. Either this argument or the `num_steps`
      argument must be supplied (but not both). The type of this argument may
      be one of the following (in order of generality):
        (a) None in which case `num_steps` must be supplied.
        (b) A positive real scalar `Tensor`. The maximum time step to take.
          If the value of this argument is `dt`, then the total number of steps
          taken is N = (t1 - t0) / dt rounded up to the nearest integer. The
          first N-1 steps are of size dt and the last step is of size
          `t1 - t0 - (N-1) * dt`.
        (c) A callable accepting the current time and returning the size of the
          step to take. The input and the output are real scalar `Tensor`s.
    one_step_fn: The transition kernel. A callable that consumes the following
      arguments by keyword:
        1. 'time': Current time
        2. 'next_time': The next time to step to. For the backwards in time
          evolution, this time will be smaller than the current time.
        3. 'coord_grid': The coordinate grid.
        4. 'values_grid': The values grid.
        5. 'second_order_coeff_fn': Callable returning the coefficients of the
          second order terms of the PDE. See the spec of the
          `second_order_coeff_fn` argument below.
        6. 'first_order_coeff_fn': Callable returning the coefficients of the
          first order terms of the PDE. See the spec of the
          `first_order_coeff_fn` argument below.
        7. 'zeroth_order_coeff_fn': Callable returning the coefficient of the
          zeroth order term of the PDE. See the spec of the
          `zeroth_order_coeff_fn` argument below.
        8. 'num_steps_performed': A scalar integer `Tensor`. Number of steps
          performed so far.
       The callable should return a sequence of two `Tensor`s. The first one
       is a `Tensor` of the same `dtype` and `shape` as `coord_grid` and
       represents a new coordinate grid after one iteration. The second `Tensor`
       is of the same shape and `dtype` as`values_grid` and represents an
       approximate solution of the equation after one iteration.
       Default value: None, which means Crank-Nicolson scheme with oscillation
       damping is used for 1D problems, and Douglas ADI scheme with `theta=0.5`
       - for multidimensional problems.
    boundary_conditions: The boundary conditions. Only rectangular boundary
      conditions are supported. A list of tuples of size `n` (space dimension
      of the PDE). The elements of the Tuple can be either a Python Callable or
      `None` representing the boundary conditions at the minimum and maximum
      values of the spatial variable indexed by the position in the list. E.g.,
      for `n=2`, the length of `boundary_conditions` should be 2,
      `boundary_conditions[0][0]` describes the boundary `(y_min, x)`, and
      `boundary_conditions[1][0]`- the boundary `(y, x_min)`. `None` values mean
      that the second order terms for that dimension on the boundary are assumed
      to be zero, i.e., if `boundary_conditions[k][0]` is None,
      'dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <=n, i!=k+1, j!=k+1] +
         Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.'
      For not `None` values, the boundary conditions are accepted in the form
      `alpha(t, x) V + beta(t, x) V_n = gamma(t, x)`, where `V_n` is the
      derivative with respect to the exterior normal to the boundary.
      Each callable receives the current time `t` and the `coord_grid` at the
      current time, and should return a tuple of `alpha`, `beta`, and `gamma`.
      Each can be a number, a zero-rank `Tensor` or a `Tensor` whose shape is
      the grid shape with the corresponding dimension removed.
      For example, for a two-dimensional grid of shape `(b, ny, nx)`, where `b`
      is the batch size, `boundary_conditions[0][i]` with `i = 0, 1` should
      return a tuple of either numbers, zero-rank tensors or tensors of shape
      `(b, nx)`. Similarly for `boundary_conditions[1][i]`, except the tensor
      shape should be `(b, ny)`. `alpha` and `beta` can also be `None` in case
      of Neumann and Dirichlet conditions, respectively.
      Default value: `None`. Unlike setting `None` to individual elements of
      `boundary_conditions`, setting the entire `boundary_conditions` object to
      `None` means Dirichlet conditions with zero value on all boundaries are
      applied.
    values_transform_fn: An optional callable applied to transform the solution
      values at each time step. The callable is invoked after the time step has
      been performed. The callable should accept the time of the grid, the
      coordinate grid and the values grid and should return the values grid. All
      input arguments to be passed by keyword.
      It returns the updated value grid and the coordinate grid, which may be
      updated as well.
    second_order_coeff_fn: Callable returning the coefficients of the second
      order terms of the PDE (i.e. `a_{ij}(t, x)` above) at given time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `coord_grid`: a `Tensor` representing a grid of locations `r` at which
          the coefficient should be evaluated.
      Returns the object `a` such that `a[i][j]` is defined and
      `a[i][j]=a_{ij}(r, t)`, where `0 <= i < n_dims` and `i <= j < n_dims`.
      For example, the object may be a list of lists or a rank 2 Tensor.
      `a[i][j]` is assumed to be symmetrical, and only the elements with
      `j >= i` will be used, so elements with `j < i` can be `None`.
      Each `a[i][j]` should be a Number, a `Tensor` broadcastable to the shape
      of `coord_grid`, or `None` if corresponding term is absent in the
      equation. Also, the callable itself may be None, meaning there are no
      second-order derivatives in the equation.
      For example, for a 2D equation with the following second order terms
      ```
      a_xx V_xx + 2 a_xy V_xy + a_yy V_yy
      ```
       the callable may return either
      `[[a_yy, a_xy], [a_xy, a_xx]]` or `[[a_yy, a_xy], [None, a_xx]]`.
      Default value: None. If both `second_order_coeff_fn` and
        `inner_second_order_coeff_fn` are None, it means the second-order term
        is absent. If only one of them is `None`, it is assumed to be `1`.
    first_order_coeff_fn: Callable returning the coefficients of the
      first order terms of the PDE (i.e. `mu_i(t, r)` above) evaluated at given
      time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Returns a list or an 1D `Tensor`, `i`-th element of which represents
      `b_i(t, r)`. Each element is a `Tensor` broadcastable to the shape of
      `locations_grid`, or None if corresponding term is absent in the
      equation. The callable itself may be None, meaning there are no
      first-order derivatives in the equation.
      Default value: None. If both `first_order_coeff_fn` and
        `inner_first_order_coeff_fn` are None, it means the first-order term is
        absent. If only one of them is `None`, it is assumed to be `1`.
    zeroth_order_coeff_fn: Callable returning the coefficient of the
      zeroth order term of the PDE (i.e. `c(t, r)` above) evaluated at given
      time `t`.
      The callable accepts the following arguments:
        `t`: The time at which the coefficient should be evaluated.
        `locations_grid`: a `Tensor` representing a grid of locations `r` at
          which the coefficient should be evaluated.
      Should return a `Tensor` broadcastable to the shape of `locations_grid`.
      May return None or be None if the shift term is absent in the equation.
      Default value: None, meaning absent zeroth order term.
    inner_second_order_coeff_fn: Callable returning the coefficients under the
      second derivatives (i.e. `A_ij(t, x)` above) at given time `t`. The
      requirements are the same as for `second_order_coeff_fn`.
    inner_first_order_coeff_fn: Callable returning the coefficients under the
      first derivatives (i.e. `B_i(t, x)` above) at given time `t`. The
      requirements are the same as for `first_order_coeff_fn`.
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
      Default value: None, which means dtype will be inferred from
      `values_grid`.
    name: The name to give to the ops.
      Default value: None which means `solve_forward` is used.

  Returns:
    The final values grid, final coordinate grid, final time and number of steps
    performed.

  Raises:
    ValueError if neither num steps nor time steps are provided or if both
    are provided.
  """
  values_grid = tf.convert_to_tensor(values_grid, dtype=dtype)
  start_time = tf.convert_to_tensor(
      start_time, dtype=values_grid.dtype, name='start_time')
  end_time = tf.math.maximum(
      tf.convert_to_tensor(end_time, dtype=values_grid.dtype, name='end_time'),
      start_time)
  return _solve(_time_direction_forward_fn,
                start_time,
                end_time,
                coord_grid,
                values_grid,
                num_steps,
                start_step_count,
                time_step,
                one_step_fn,
                boundary_conditions,
                values_transform_fn,
                second_order_coeff_fn,
                first_order_coeff_fn,
                zeroth_order_coeff_fn,
                inner_second_order_coeff_fn,
                inner_first_order_coeff_fn,
                maximum_steps,
                swap_memory,
                name or 'solve_forward')


def _solve(
    time_direction_fn,
    start_time,
    end_time,
    coord_grid,
    values_grid,
    num_steps=None,
    start_step_count=0,
    time_step=None,
    one_step_fn=None,
    boundary_conditions=None,
    values_transform_fn=None,
    second_order_coeff_fn=None,
    first_order_coeff_fn=None,
    zeroth_order_coeff_fn=None,
    inner_second_order_coeff_fn=None,
    inner_first_order_coeff_fn=None,
    maximum_steps=None,
    swap_memory=True,
    name=None):
  """Common code for solve_backward and solve_forward."""
  if (num_steps is None) == (time_step is None):
    raise ValueError('Exactly one of num_steps or time_step'
                     ' should be supplied.')
  coord_grid = [
      tf.convert_to_tensor(dim_grid, dtype=values_grid.dtype)
      for dim_grid in coord_grid
  ]

  n_dims = len(coord_grid)
  if one_step_fn is None:
    if n_dims == 1:
      one_step_fn = oscillation_damped_crank_nicolson_step()
    else:
      one_step_fn = douglas_adi_step(theta=0.5)

  if boundary_conditions is None:

    def zero_dirichlet(t, grid):
      del t, grid
      return 1, None, tf.constant(0, dtype=values_grid.dtype)

    boundary_conditions = [(zero_dirichlet, zero_dirichlet)] * n_dims

  with tf.compat.v1.name_scope(
      name,
      default_name='solve',
      values=[
          start_time,
          end_time,
          coord_grid,
          values_grid,
          num_steps,
          time_step,
      ]):
    time_step_fn, est_max_steps = _get_time_steps_info(start_time, end_time,
                                                       num_steps, time_step,
                                                       time_direction_fn)
    if est_max_steps is None and maximum_steps is not None:
      est_max_steps = maximum_steps

    def loop_cond(should_stop, time, x_grid, f_grid, steps_performed):
      del time, x_grid, f_grid, steps_performed
      return tf.logical_not(should_stop)

    def loop_body(should_stop, time, x_grid, f_grid, steps_performed):
      """Propagates the grid in time."""
      del should_stop
      next_should_stop, t_next = time_step_fn(time)
      next_xs, next_fs = one_step_fn(
          time=time,
          next_time=t_next,
          coord_grid=x_grid,
          value_grid=f_grid,
          boundary_conditions=boundary_conditions,
          second_order_coeff_fn=second_order_coeff_fn,
          first_order_coeff_fn=first_order_coeff_fn,
          zeroth_order_coeff_fn=zeroth_order_coeff_fn,
          inner_second_order_coeff_fn=inner_second_order_coeff_fn,
          inner_first_order_coeff_fn=inner_first_order_coeff_fn,
          num_steps_performed=steps_performed)

      if values_transform_fn is not None:
        next_xs, next_fs = values_transform_fn(t_next, next_xs, next_fs)
      return next_should_stop, t_next, next_xs, next_fs, steps_performed + 1

    # If the start time is already equal to end time, no stepping is needed.
    # solve_backward, solve_forward already took care of the case when end_time
    # is on the "wrong side" of start_time.
    should_already_stop = (start_time == end_time)
    initial_args = (should_already_stop, start_time, coord_grid, values_grid,
                    start_step_count)
    (_, final_time, final_coords, final_values,
     steps_performed) = tf.while_loop(
         loop_cond,
         loop_body,
         initial_args,
         swap_memory=swap_memory,
         maximum_iterations=est_max_steps)
    return final_values, final_coords, final_time, steps_performed


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


def _get_time_steps_info(start_time, end_time, num_steps, time_step,
                         time_direction_fn):
  """Creates a callable to step through time and estimates the max steps."""
  # time_direction_fn must be one of _time_step_forward_fn and
  # _time_step_backward_fn
  dt = None
  estimated_max_steps = None
  interval = tf.math.abs(end_time - start_time)
  if num_steps is not None:
    dt = interval / tf.cast(num_steps, dtype=start_time.dtype)
    estimated_max_steps = num_steps
  if time_step is not None and not _is_callable(time_step):
    dt = time_step
    estimated_max_steps = tf.cast(tf.math.ceil(interval / dt), dtype=tf.int32)
  if dt is not None:
    raw_time_step_fn = lambda _: dt
  else:
    raw_time_step_fn = time_step

  def time_step_fn(t):
    # t is the current time.
    # t_next is the next time
    dt = raw_time_step_fn(t)
    should_stop, t_next = time_direction_fn(t, dt, end_time)
    return should_stop, t_next

  return time_step_fn, estimated_max_steps


def _time_direction_forward_fn(t, dt, end_time):
  t_next = tf.math.minimum(end_time, t + dt)
  return t_next >= end_time, t_next


def _time_direction_backward_fn(t, dt, end_time):
  t_next = tf.math.maximum(end_time, t - dt)
  return t_next <= end_time, t_next


__all__ = ['solve_backward', 'solve_forward']
