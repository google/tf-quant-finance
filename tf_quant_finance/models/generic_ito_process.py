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
"""Defines class to describe any Ito processes.

Uses Euler scheme for sampling and ADI scheme for solving the associated
Feynman-Kac equation.
"""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.pde import fd_solvers
from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import ito_process


class GenericItoProcess(ito_process.ItoProcess):
  """Generic Ito process defined from a drift and volatility function."""

  def __init__(self, dim, drift_fn, volatility_fn, dtype=None, name=None):
    """Initializes the Ito process with given drift and volatility functions.

    Represents a general Ito process:

    ```None
      dX_i = a_i(t, X) dt + Sum(S_{ij}(t, X) dW_j for 1 <= j <= n), 1 <= i <= n
    ```

    The vector coefficient `a_i` is referred to as the drift of the process and
    the matrix `b_{ij}` as the volatility of the process. For the process to be
    well defined, these coefficients need to satisfy certain technical
    conditions which may be found in Ref. [1]. The vector `dW_j` represents
    independent Brownian increments.

    #### Example. Sampling from 2-dimensional Ito process of the form:

    ```none
    dX_1 = mu_1 * sqrt(t) dt + s11 * dW_1 + s12 * dW_2
    dX_2 = mu_2 * sqrt(t) dt + s21 * dW_1 + s22 * dW_2
    ```

    ```python
    mu = np.array([0.2, 0.7])
    s = np.array([[0.3, 0.1], [0.1, 0.3]])
    num_samples = 10000
    dim = 2
    dtype=tf.float64

    # Define drift and volatility functions
    def drift_fn(t, x):
      return mu * tf.sqrt(t) * tf.ones([num_samples, dim], dtype=dtype)

    def vol_fn(t, x):
      return s * tf.ones([num_samples, dim, dim], dtype=dtype)

    # Initialize `GenericItoProcess`
    process = GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn,
                                dtype=dtype)
    # Set starting location
    x0 = np.array([0.1, -1.1])
    # Sample `num_samples` paths at specified `times` locations using built-in
    # Euler scheme.
    times = [0.1, 1.0, 2.0]
    paths = process.sample_paths(
              times,
              num_samples=num_samples,
              initial_state=x0,
              time_step=0.01,
              seed=42)
    ```

    #### References
    [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
      Applications. Springer. 2010.

    Args:
      dim: Python int greater than or equal to 1. The dimension of the Ito
        process.
      drift_fn: A Python callable to compute the drift of the process. The
        callable should accept two real `Tensor` arguments of the same dtype.
        The first argument is the scalar time t, the second argument is the
        value of Ito process X - `Tensor` of shape
        `batch_shape + sample_shape + [dim]`, where `batch_shape` represents
        a batch of models and `sample_shape` represents samples for each of the
        models. The result is value of drift a(t, X). The return value of the
        callable is a real `Tensor` of the same dtype as the input arguments and
        of shape `batch_shape + sample_shape + [dim]`.
      volatility_fn: A Python callable to compute the volatility of the process.
        The callable should accept two real `Tensor` arguments of the same dtype
        and shape `times_shape`. The first argument is the scalar time t, the
        second argument is the value of Ito process X - `Tensor` of shape
        `batch_shape + sample_shape + [dim]`, where `batch_shape` represents
        a batch of models and `sample_shape` represents samples for each of the
        models. The result is value of volatility S_{ij}(t, X). The return value
        of the callable is a real `Tensor` of the same dtype as the input
        arguments and of shape `batch_shape + sample_shape + [dim, dim]`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: None which means that default dtypes inferred by
          TensorFlow are used.
      name: str. The name scope under which ops created by the methods of this
        class are nested.
        Default value: None which maps to the default name
          `generic_ito_process`.

    Raises:
      ValueError if the dimension is less than 1, or if either `drift_fn`
        or `volatility_fn` is not supplied.
    """
    if dim < 1:
      raise ValueError('Dimension must be 1 or greater.')
    if drift_fn is None or volatility_fn is None:
      raise ValueError('Both drift and volatility functions must be supplied.')
    self._dim = dim
    self._drift_fn = drift_fn
    self._volatility_fn = volatility_fn
    self._dtype = dtype
    self._name = name or 'generic_ito_process'

  def dim(self):
    """The dimension of the process."""
    return self._dim

  def dtype(self):
    """The data type of process realizations."""
    return self._dtype

  def name(self):
    """The name to give to ops created by this class."""
    return self._name

  def drift_fn(self):
    """Python callable calculating instantaneous drift.

    The callable should accept two real `Tensor` arguments of the same dtype.
    The first argument is the scalar time t, the second argument is the value of
    Ito process X - `Tensor` of shape `batch_shape + [dim]`. The result is the
    value of drift a(t, X). The return value of the callable is a real `Tensor`
    of the same dtype as the input arguments and of shape `batch_shape + [dim]`.

    Returns:
      The instantaneous drift rate callable.
    """
    return self._drift_fn

  def volatility_fn(self):
    """Python callable calculating the instantaneous volatility.

    The callable should accept two real `Tensor` arguments of the same dtype and
    shape `times_shape`. The first argument is the scalar time t, the second
    argument is the value of Ito process X - `Tensor` of shape `batch_shape +
    [dim]`. The result is value of volatility `S_ij`(t, X). The return value of
    the callable is a real `Tensor` of the same dtype as the input arguments and
    of shape `batch_shape + [dim, dim]`.

    Returns:
      The instantaneous volatility callable.
    """
    return self._volatility_fn

  def sample_paths(self,
                   times,
                   num_samples=1,
                   initial_state=None,
                   random_type=None,
                   seed=None,
                   swap_memory=True,
                   name=None,
                   time_step=None,
                   num_time_steps=None,
                   skip=0,
                   precompute_normal_draws=True,
                   times_grid=None,
                   normal_draws=None,
                   watch_params=None,
                   validate_args=False):
    """Returns a sample of paths from the process using Euler sampling.

    The default implementation uses the Euler scheme. However, for particular
    types of Ito processes more efficient schemes can be used.

    Args:
      times: Rank 1 `Tensor` of increasing positive real values. The times at
        which the path points are to be evaluated.
      num_samples: Positive scalar `int`. The number of paths to draw.
        Default value: 1.
      initial_state: `Tensor` of shape broadcastable
        `batch_shape + [num_samples, dim]`. The initial state of the process.
        `batch_shape` represents the shape of the independent batches of the
        stochastic process as in the `drift_fn` and `volatility_fn` of the
        underlying class. Note that the `batch_shape` is inferred from
        the `initial_state` and hence when sampling is requested for a batch of
        stochastic processes, the shape of `initial_state` should be as least
        `batch_shape + [1, 1]`.
        Default value: None which maps to a zero initial state.
      random_type: Enum value of `RandomType`. The type of (quasi)-random number
        generator to use to generate the paths.
        Default value: None which maps to the standard pseudo-random numbers.
      seed: Seed for the random number generator. The seed is
        only relevant if `random_type` is one of
        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
        `HALTON_RANDOMIZED` the seed should be an Python integer. For
        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
        `Tensor` of shape `[2]`.
        Default value: `None` which means no seed is set.
      swap_memory: A Python bool. Whether GPU-CPU memory swap is enabled for
        this op. See an equivalent flag in `tf.while_loop` documentation for
        more details. Useful when computing a gradient of the op since
        `tf.while_loop` is used to propagate stochastic process in time.
        Default value: True.
      name: Python string. The name to give this op.
        Default value: `None` which maps to `sample_paths` is used.
      time_step: An optional scalar real `Tensor` - maximal distance between
        points in the time grid.
        Either this or `num_time_steps` should be supplied.
        Default value: `None`.
      num_time_steps: An optional Scalar integer `Tensor` - a total number of
        time steps performed by the algorithm. The maximal distance between
        points in grid is bounded by
        `times[-1] / (num_time_steps - times.shape[0])`.
        Either this or `time_step` should be supplied.
        Default value: `None`.
      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
        Halton sequence to skip. Used only when `random_type` is 'SOBOL',
        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
        Default value: `0`.
      precompute_normal_draws: Python bool. Indicates whether the noise
        increments in Euler scheme are precomputed upfront (see
        `models.euler_sampling.sample`). For `HALTON` and `SOBOL` random types
        the increments are always precomputed. While the resulting graph
        consumes more memory, the performance gains might be significant.
        Default value: `True`.
      times_grid: An optional rank 1 `Tensor` representing time discretization
        grid. If `times` are not on the grid, then the nearest points from the
        grid are used.
        Default value: `None`, which means that times grid is computed using
        `time_step` and `num_time_steps`.
      normal_draws: A `Tensor` of shape
        `batch_shape + [num_samples, num_time_points, dim]`
        and the same `dtype` as `times`. Represents random normal draws to
        compute increments `N(0, t_{n+1}) - N(0, t_n)`. `batch_shape` is the
        shape of the independent batches of the stochastic process. When
        supplied, `num_sample`, `time_step` and `num_time_steps` arguments are
        ignored and the first dimensions of `normal_draws` are used instead.
      watch_params: An optional list of zero-dimensional `Tensor`s of the same
        `dtype` as `initial_state`. If provided, specifies `Tensor`s with
        respect to which the differentiation of the sampling function will
        happen. A more efficient algorithm is used when `watch_params` are
        specified. Note the the function becomes differentiable onlhy wrt to
        these `Tensor`s and the `initial_state`. The gradient wrt any other
        `Tensor` is set to be zero.
      validate_args: Python `bool`. When `True` and `normal_draws` are supplied,
        checks that `tf.shape(normal_draws)[1]` is equal to `num_time_steps`
        that is either supplied as an argument or computed from `time_step`.
        When `False` invalid dimension may silently render incorrect outputs.
        Default value: `False`.

    Returns:
     A real `Tensor` of shape `batch_shape + [num_samples, k, n]` where `k`
     is the size of the `times`, and `n` is the dimension of the process.

    Raises:
      ValueError:
        (a) When `times_grid` is not supplied, and neither `num_time_steps` nor
          `time_step` are supplied or if both are supplied.
        (b) If `normal_draws` is supplied and `dim` is mismatched.
      tf.errors.InvalidArgumentError: If `normal_draws` is supplied and
        `num_time_steps` is mismatched.
    """
    name = name or (self._name + '_sample_path')
    with tf.name_scope(name):
      return euler_sampling.sample(
          dim=self._dim,
          drift_fn=self._drift_fn,
          volatility_fn=self._volatility_fn,
          times=times,
          num_samples=num_samples,
          initial_state=initial_state,
          random_type=random_type,
          time_step=time_step,
          num_time_steps=num_time_steps,
          seed=seed,
          swap_memory=swap_memory,
          skip=skip,
          precompute_normal_draws=precompute_normal_draws,
          times_grid=times_grid,
          normal_draws=normal_draws,
          watch_params=watch_params,
          dtype=self._dtype,
          name=name)

  def fd_solver_backward(self,
                         start_time,
                         end_time,
                         coord_grid,
                         values_grid,
                         discounting=None,
                         one_step_fn=None,
                         boundary_conditions=None,
                         start_step_count=0,
                         num_steps=None,
                         time_step=None,
                         values_transform_fn=None,
                         dtype=None,
                         name=None,
                         **kwargs):
    """Returns a solver for Feynman-Kac PDE associated to the process.

    This method applies a finite difference method to solve the final value
    problem as it appears in the Feynman-Kac formula associated to this Ito
    process. The Feynman-Kac PDE is closely related to the backward Kolomogorov
    equation associated to the stochastic process and allows for the inclusion
    of a discounting function.

    For more details of the Feynman-Kac theorem see [1]. The PDE solved by this
    method is:

    ```None
      V_t + Sum[mu_i(t, x) V_i, 1<=i<=n] +
        (1/2) Sum[ D_{ij} V_{ij}, 1 <= i,j <= n] - r(t, x) V = 0
    ```

    In the above, `V_t` is the derivative of `V` with respect to `t`,
    `V_i` is the partial derivative with respect to `x_i` and `V_{ij}` the
    (mixed) partial derivative with respect to `x_i` and `x_j`. `mu_i` is the
    drift of this process and `D_{ij}` are the components of the diffusion
    tensor:

    ```None
      D_{ij}(t,x) = (Sigma(t,x) . Transpose[Sigma(t,x)])_{ij}
    ```

    This method evolves a spatially discretized solution of the above PDE from
    time `t0` to time `t1 < t0` (i.e. backwards in time).
    The solution `V(t,x)` is assumed to be discretized on an `n`-dimensional
    rectangular grid. A rectangular grid, G, in n-dimensions may be described
    by specifying the coordinates of the points along each axis. For example,
    a 2 x 4 grid in two dimensions can be specified by taking the cartesian
    product of [1, 3] and [5, 6, 7, 8] to yield the grid points with
    coordinates: `[(1, 5), (1, 6), (1, 7), (1, 8), (3, 5) ... (3, 8)]`.

    This method allows batching of solutions. In this context, batching means
    the ability to represent and evolve multiple independent functions `V`
    (e.g. V1, V2 ...) simultaneously. A single discretized solution is specified
    by stating its values at each grid point. This can be represented as a
    `Tensor` of shape [d1, d2, ... dn] where di is the grid size along the `i`th
    axis. A batch of such solutions is represented by a `Tensor` of shape:
    `batch_shape + payoff_shape + [d1, d2, ... dn]` where `batch_shape` is the
    batch of processes as in the underlying `drift_fn` and `volatility_fn` and
    `payoff_shape` are the equations to be solved for each batch element.

    The evolution of the solution from `t0` to `t1` is often done by
    discretizing the differential equation to a difference equation along
    the spatial and temporal axes. The temporal discretization is given by a
    (sequence of) time steps [dt_1, dt_2, ... dt_k] such that the sum of the
    time steps is equal to the total time step `t0 - t1`. If a uniform time
    step is used, it may equivalently be specified by stating the number of
    steps (n_steps) to take. This method provides both options via the
    `time_step` and `num_steps` parameters. However, not all methods need
    discretization along time direction (e.g. method of lines) so this argument
    may not be applicable to some implementations.

    The workhorse of this method is the `one_step_fn`. For the commonly used
    methods, see functions in `math.pde.steppers` module.

    The mapping between the arguments of this method and the above
    equation are described in the Args section below.

    For a simple instructive example of implementation of this method, see
    `models.GenericItoProcess.fd_solver_backward`.

    #### Examples
    ```python
    import tensorflow as tf
    import numpy as np

    import tf_quant_finance as tff
    dtype = tf.float64

    # Specify volatilities, interest rates and strikes for the options
    volatilities = tf.constant([[0.3], [0.15], [0.1]], dtype)
    rates = tf.constant([[0.01], [0.03], [0.01]], dtype)
    expiries = 1.0

    # Define Generic Ito Process

    # Process dimensionality
    dim = 1

    # Batch size of the process
    num_processes = 3

    def drift_fn(t, x):
      del t
      # `x` is expected to be of shape [num_processes] + sample_shape + [dim]
      # We need to expand rank of rates to
      # `[num_processes] + extra_rank * [1] + [1]`
      expand_rank = x.shape.rank - 2
      rates_expand = tf.reshape(
          rates, [num_processes] + (expand_rank + 1) * [1])
      # Output is of shape [num_processes] + sample_shape + [dim]
      return rates_expand * x

    def vol_fn(t, x):
      del t
      # `x` is expected to be of shape [num_processes] + sample_shape + [dim]
      # As before, need to expand rank of volatilities to
      # `[num_processes] + extra_rank * [1] + [1]`
      expand_rank = x.shape.rank - 2
      volatilities_expand = tf.reshape(
          volatilities, [num_processes] + (expand_rank + 1) * [1])
      # Output is of shape [num_processes] + sample_shape + [dim, dim]
      return (tf.expand_dims(volatilities_expand * x, axis=-1)
              * tf.eye(dim, batch_shape=x.shape.as_list()[:-1], dtype=x.dtype))

    process = tff.models.GenericItoProcess(dim=dim,
                                           drift_fn=drift_fn,
                                           volatility_fn=vol_fn,
                                           dtype=dtype)
    # Define a 2 strikes for each batch process,
    num_strikes = 2
    # Shape [num_processes, num_strikes, 1]. Here 1 at the end is just for
    # convenience
    strikes = tf.constant([[[50], [60]], [[100], [90]], [[120], [90]]], dtype)

    # Price a batch of European call options
    @tff.math.pde.boundary_conditions.dirichlet
    def upper_boundary_fn(t, grid):
      del grid
      # Shape (num_processes, num_strikes)
      return tf.squeeze(s_max - strikes * tf.exp(-rates  * (expiries - t)))

    # Define discounting function
    def discounting(t, x):
      del t, x
      rates_expand = tf.expand_dims(rates, axis=-1)
      # Shape compatible with (num_processes, num_strikes)
      return rates_expand

    # Build a uniform grid
    s_min = 0
    s_max = 200
    num_grid_points = 256  # Number of grid points

    grid = tff.math.pde.grids.uniform_grid(minimums=[s_min],
                                           maximums=[s_max],
                                           sizes=[num_grid_points],
                                           dtype=dtype)

    # Shape [num_processes, num_strikes, num_grid_points]
    final_value_grid = tf.nn.relu(grid[0] - strikes)

    # Estimated prices for the European options
    process.fd_solver_backward(
        start_time=expiries,
        end_time=0,
        time_step=0.1,
        coord_grid=grid,
        values_grid=final_value_grid,
        discounting=discounting,
        boundary_condtions=[(None, upper_boundary_fn)])[0]
    # Shape of the output is [num_processes, num_strikes, num_grid_points]
    ```

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
        of the `Tensor` must broadcast with
        `batch_shape + payoff_shape + [d_1, d_2, ..., d_n]`. `batch_shape`
        represents the batch of the processes as in the underlying `drift_fn`
        and `volatility_fn`. `payoff_shape` specifies equations to be solved for
        each batch element (with potentially different boundary/final conditions
        and for various coordinate grids). When the batch dimensions
        `batch_shape` or `payoff_shape` are present, the shape of values_grid`
        must be at least `batch_shape + payoff_shape + dim * [1]`.
      discounting: Callable corresponding to `r(t,x)` above. If not supplied,
        zero discounting is assumed.
      one_step_fn: The transition kernel. A callable that consumes the following
        arguments by keyword:
          1. 'time': Current time
          2. 'next_time': The next time to step to. For the backwards in time
            evolution, this time will be smaller than the current time.
          3. 'coord_grid': The coordinate grid.
          4. 'values_grid': The values grid.
          5. 'boundary_conditions': The boundary conditions.
          6. 'quadratic_coeff': A callable returning the quadratic coefficients
            of the PDE (i.e. `(1/2)D_{ij}(t, x)` above). The callable accepts
            the time and  coordinate grid as keyword arguments and returns a
            `Tensor` with shape that broadcasts with `[dim, dim]`.
          7. 'linear_coeff': A callable returning the linear coefficients of the
            PDE (i.e. `mu_i(t, x)` above). Accepts time and coordinate grid as
            keyword arguments and returns a `Tensor` with shape that broadcasts
            with `[dim]`.
          8. 'constant_coeff': A callable returning the coefficient of the
            linear homogeneous term (i.e. `r(t,x)` above). Same spec as above.
            The `one_step_fn` callable returns a 2-tuple containing the next
            coordinate grid, next values grid.
      boundary_conditions: The boundary conditions. Only rectangular boundary
        conditions are supported. A list of tuples of size `n` (space dimension
        of the PDE). The elements of the Tuple can be either a Python Callable
        or `None` representing the boundary conditions at the minimum and
        maximum values of the spatial variable indexed by the position in the
        list. E.g., for `n=2`, the length of `boundary_conditions` should be 2,
        `boundary_conditions[0][0]` describes the boundary `(y_min, x)`, and
        `boundary_conditions[1][0]`- the boundary `(y, x_min)`. `None` values
        mean that the second order terms for that dimension on the boundary are
        assumed to be zero, i.e., if `boundary_conditions[k][0]` is `None`,
        'dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <= n, i!=k+1, j!=k+1]
           + Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.'
        For not `None` values, the boundary conditions are accepted in the form
        `alpha(t, x) V + beta(t, x) V_n = gamma(t, x)`, where `V_n` is the
        derivative with respect to the exterior normal to the boundary.
        Each callable receives the current time `t` and the `coord_grid` at the
        current time, and should return a tuple of `alpha`, `beta`, and `gamma`.
        Each can be a number, a zero-rank `Tensor` or a `Tensor` whose shape is
        the grid shape with the corresponding dimension removed.
        For example, for a two-dimensional grid of shape `(b, ny, nx)`, where
        `b` is the batch size, `boundary_conditions[0][i]` with `i = 0, 1`
        should return a tuple of either numbers, zero-rank tensors or tensors of
        shape `(b, nx)`. Similarly for `boundary_conditions[1][i]`, except the
        tensor shape should be `(b, ny)`. `alpha` and `beta` can also be `None`
        in case of Neumann and Dirichlet conditions, respectively.
        Default value: `None`. Unlike setting `None` to individual elements of
        `boundary_conditions`, setting the entire `boundary_conditions` object
        to `None` means Dirichlet conditions with zero value on all boundaries
        are applied.
      start_step_count: Scalar integer `Tensor`. Initial value for the number of
        time steps performed.
        Default value: 0 (i.e. no previous steps performed).
      num_steps: Positive int scalar `Tensor`. The number of time steps to take
        when moving from `start_time` to `end_time`. Either this argument or the
        `time_step` argument must be supplied (but not both). If num steps is
        `k>=1`, uniform time steps of size `(t0 - t1)/k` are taken to evolve the
        solution from `t0` to `t1`. Corresponds to the `n_steps` parameter
        above.
      time_step: The time step to take. Either this argument or the `num_steps`
        argument must be supplied (but not both). The type of this argument may
        be one of the following (in order of generality): (a) None in which case
          `num_steps` must be supplied. (b) A positive real scalar `Tensor`. The
          maximum time step to take. If the value of this argument is `dt`, then
          the total number of steps taken is N = (t0 - t1) / dt rounded up to
          the nearest integer. The first N-1 steps are of size dt and the last
          step is of size `t0 - t1 - (N-1) * dt`. (c) A callable accepting the
          current time and returning the size of the step to take. The input and
          the output are real scalar `Tensor`s.
      values_transform_fn: An optional callable applied to transform the
        solution values at each time step. The callable is invoked after the
        time step has been performed. The callable should accept the time of the
        grid, the coordinate grid and the values grid and should return the
        values grid. All input arguments to be passed by keyword.
      dtype: The dtype to use.
      name: The name to give to the ops.
        Default value: None which means `solve_backward` is used.
      **kwargs: Additional keyword args:
        (1) pde_solver_fn: Function to solve the PDE that accepts all the above
          arguments by name and returns the same tuple object as required below.
          Defaults to `tff.math.pde.fd_solvers.solve_backward`.

    Returns:
      A tuple object containing at least the following attributes:
        final_values_grid: A `Tensor` of same shape and dtype as `values_grid`.
          Contains the final state of the values grid at time `end_time`.
        final_coord_grid: A list of `Tensor`s of the same specification as
          the input `coord_grid`. Final state of the coordinate grid at time
          `end_time`.
        step_count: The total step count (i.e. the sum of the `start_step_count`
          and the number of steps performed in this call.).
        final_time: The final time at which the evolution stopped. This value
          is given by `max(min(end_time, start_time), 0)`.
    """
    pde_solver_fn = kwargs.get('pde_solver_fn', fd_solvers.solve_backward)

    values_grid_shape = _get_static_shape(values_grid)
    # batch shape includes payff_shape in the description
    batch_shape = values_grid_shape[:-self.dim()]

    second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn = (
        _backward_pde_coeffs(self._drift_fn, self._volatility_fn, discounting,
                             batch_shape=batch_shape,
                             dim=self.dim()))

    return pde_solver_fn(
        start_time=start_time,
        end_time=end_time,
        coord_grid=coord_grid,
        values_grid=values_grid,
        num_steps=num_steps,
        start_step_count=start_step_count,
        time_step=time_step,
        one_step_fn=one_step_fn,
        boundary_conditions=boundary_conditions,
        values_transform_fn=values_transform_fn,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn,
        dtype=dtype,
        name=name)

  def fd_solver_forward(self,
                        start_time,
                        end_time,
                        coord_grid,
                        values_grid,
                        one_step_fn=None,
                        boundary_conditions=None,
                        start_step_count=0,
                        num_steps=None,
                        time_step=None,
                        values_transform_fn=None,
                        dtype=None,
                        name=None,
                        **kwargs):
    r"""Returns a solver for the Fokker Plank equation of this process.

    The Fokker Plank equation (also known as the Kolmogorov Forward equation)
    associated to this Ito process is given by:

    ```None
      V_t + Sum[(mu_i(t, x) V)_i, 1<=i<=n]
        - (1/2) Sum[ (D_{ij} V)_{ij}, 1 <= i,j <= n] = 0
    ```

    with the initial value condition $$V(0, x) = u(x)$$.

    This method evolves a spatially discretized solution of the above PDE from
    time `t0` to time `t1 > t0` (i.e. forwards in time).
    The solution `V(t,x)` is assumed to be discretized on an `n`-dimensional
    rectangular grid. A rectangular grid, G, in n-dimensions may be described
    by specifying the coordinates of the points along each axis. For example,
    a 2 x 4 grid in two dimensions can be specified by taking the cartesian
    product of [1, 3] and [5, 6, 7, 8] to yield the grid points with
    coordinates: `[(1, 5), (1, 6), (1, 7), (1, 8), (3, 5) ... (3, 8)]`.

    This method allows batching of solutions. In this context, batching means
    the ability to represent and evolve multiple independent functions `V`
    (e.g. V1, V2 ...) simultaneously. A single discretized solution is specified
    by stating its values at each grid point. This can be represented as a
    `Tensor` of shape [d1, d2, ... dn] where di is the grid size along the `i`th
    axis. A batch of such solutions is represented by a `Tensor` of shape:
    `batch_shape + payoff_shape + [d1, d2, ... dn]` where `batch_shape` is the
    batch of processes as in the underlying `drift_fn` and `volatility_fn` and
    `payoff_shape` are the equations to be solved for each batch element.

    The evolution of the solution from `t0` to `t1` is often done by
    discretizing the differential equation to a difference equation along
    the spatial and temporal axes. The temporal discretization is given by a
    (sequence of) time steps [dt_1, dt_2, ... dt_k] such that the sum of the
    time steps is equal to the total time step `t1 - t0`. If a uniform time
    step is used, it may equivalently be specified by stating the number of
    steps (n_steps) to take. This method provides both options via the
    `time_step` and `num_steps` parameters. However, not all methods need
    discretization along time direction (e.g. method of lines) so this argument
    may not be applicable to some implementations.

    The workhorse of this method is the `one_step_fn`. For the commonly used
    methods, see functions in `math.pde.steppers` module.

    The mapping between the arguments of this method and the above
    equation are described in the Args section below.

    For a simple instructive example of implementation of this method, see
    `models.GenericItoProcess.fd_solver_forward`.

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
        of the `Tensor` must broadcast with
        `batch_shape + payoff_shape + [d_1, d_2, ..., d_n]`. `batch_shape`
        represents the batch of the processes as in the underlying `drift_fn`
        and `volatility_fn`. `payoff_shape` specifies equations to be solved for
        each batch element (with potentially different boundary/final conditions
        and for various coordinate grids). When the batch dimensions
        `batch_shape` or `payoff_shape` are present, the shape of values_grid`
        must be at least `batch_shape + payoff_shape + dim * [1]`.
      one_step_fn: The transition kernel. A callable that consumes the following
        arguments by keyword:
          1. 'time': Current time
          2. 'next_time': The next time to step to. For the backwards in time
            evolution, this time will be smaller than the current time.
          3. 'coord_grid': The coordinate grid.
          4. 'values_grid': The values grid.
          5. 'quadratic_coeff': A callable returning the quadratic coefficients
            of the PDE (i.e. `(1/2)D_{ij}(t, x)` above). The callable accepts
            the time and  coordinate grid as keyword arguments and returns a
            `Tensor` with shape that broadcasts with `[dim, dim]`.
          6. 'linear_coeff': A callable returning the linear coefficients of the
            PDE (i.e. `mu_i(t, x)` above). Accepts time and coordinate grid as
            keyword arguments and returns a `Tensor` with shape that broadcasts
            with `[dim]`.
          7. 'constant_coeff': A callable returning the coefficient of the
            linear homogeneous term (i.e. `r(t,x)` above). Same spec as above.
            The `one_step_fn` callable returns a 2-tuple containing the next
            coordinate grid, next values grid.
      boundary_conditions: The boundary conditions. Only rectangular boundary
        conditions are supported. A list of tuples of size `n` (space dimension
        of the PDE). The elements of the Tuple can be either a Python Callable
        or `None` representing the boundary conditions at the minimum and
        maximum values of the spatial variable indexed by the position in the
        list. E.g., for `n=2`, the length of `boundary_conditions` should be 2,
        `boundary_conditions[0][0]` describes the boundary `(y_min, x)`, and
        `boundary_conditions[1][0]`- the boundary `(y, x_min)`. `None` values
        mean that the second order terms for that dimension on the boundary are
        assumed to be zero, i.e., if `boundary_conditions[k][0]` is `None`,
        'dV/dt + Sum[a_ij d2(A_ij V)/dx_i dx_j, 1 <= i, j <= n, i!=k+1, j!=k+1]
           + Sum[b_i d(B_i V)/dx_i, 1 <= i <= n] + c V = 0.'
        For not `None` values, the boundary conditions are accepted in the form
        `alpha(t, x) V + beta(t, x) V_n = gamma(t, x)`, where `V_n` is the
        derivative with respect to the exterior normal to the boundary.
        Each callable receives the current time `t` and the `coord_grid` at the
        current time, and should return a tuple of `alpha`, `beta`, and `gamma`.
        Each can be a number, a zero-rank `Tensor` or a `Tensor` whose shape is
        the grid shape with the corresponding dimension removed.
        For example, for a two-dimensional grid of shape `(b, ny, nx)`, where
        `b` is the batch size, `boundary_conditions[0][i]` with `i = 0, 1`
        should return a tuple of either numbers, zero-rank tensors or tensors of
        shape `(b, nx)`. Similarly for `boundary_conditions[1][i]`, except the
        tensor shape should be `(b, ny)`. `alpha` and `beta` can also be `None`
        in case of Neumann and Dirichlet conditions, respectively.
        Default value: `None`. Unlike setting `None` to individual elements of
        `boundary_conditions`, setting the entire `boundary_conditions` object
        to `None` means Dirichlet conditions with zero value on all boundaries
        are applied.
      start_step_count: Scalar integer `Tensor`. Initial value for the number of
        time steps performed.
        Default value: 0 (i.e. no previous steps performed).
      num_steps: Positive int scalar `Tensor`. The number of time steps to take
        when moving from `start_time` to `end_time`. Either this argument or the
        `time_step` argument must be supplied (but not both). If num steps is
        `k>=1`, uniform time steps of size `(t0 - t1)/k` are taken to evolve the
        solution from `t0` to `t1`. Corresponds to the `n_steps` parameter
        above.
      time_step: The time step to take. Either this argument or the `num_steps`
        argument must be supplied (but not both). The type of this argument may
        be one of the following (in order of generality): (a) None in which case
          `num_steps` must be supplied. (b) A positive real scalar `Tensor`. The
          maximum time step to take. If the value of this argument is `dt`, then
          the total number of steps taken is N = (t1 - t0) / dt rounded up to
          the nearest integer. The first N-1 steps are of size dt and the last
          step is of size `t1 - t0 - (N-1) * dt`. (c) A callable accepting the
          current time and returning the size of the step to take. The input and
          the output are real scalar `Tensor`s.
      values_transform_fn: An optional callable applied to transform the
        solution values at each time step. The callable is invoked after the
        time step has been performed. The callable should accept the time of the
        grid, the coordinate grid and the values grid and should return the
        values grid. All input arguments to be passed by keyword.
      dtype: The dtype to use.
      name: The name to give to the ops.
        Default value: None which means `solve_forward` is used.
      **kwargs: Additional keyword args:
        (1) pde_solver_fn: Function to solve the PDE that accepts all the above
          arguments by name and returns the same tuple object as required below.
          Defaults to `tff.math.pde.fd_solvers.solve_forward`.

    Returns:
      A tuple object containing at least the following attributes:
        final_values_grid: A `Tensor` of same shape and dtype as `values_grid`.
          Contains the final state of the values grid at time `end_time`.
        final_coord_grid: A list of `Tensor`s of the same specification as
          the input `coord_grid`. Final state of the coordinate grid at time
          `end_time`.
        step_count: The total step count (i.e. the sum of the `start_step_count`
          and the number of steps performed in this call.).
        final_time: The final time at which the evolution stopped. This value
          is given by `max(min(end_time, start_time), 0)`.
    """
    pde_solver_fn = kwargs.get('pde_solver_fn', fd_solvers.solve_forward)

    values_grid_shape = _get_static_shape(values_grid)
    # batch shape includes payff_shape in the description
    batch_shape = values_grid_shape[:-self.dim()]

    backward_second_order, backward_first_order, backward_zeroth_order = (
        _backward_pde_coeffs(self._drift_fn, self._volatility_fn,
                             discounting=None,
                             batch_shape=batch_shape,
                             dim=self.dim()))

    # Transform backward to forward equation.
    inner_second_order_coeff_fn = lambda t, x: -backward_second_order(t, x)
    inner_first_order_coeff_fn = backward_first_order
    zeroth_order_coeff_fn = backward_zeroth_order

    return pde_solver_fn(
        start_time=start_time,
        end_time=end_time,
        coord_grid=coord_grid,
        values_grid=values_grid,
        num_steps=num_steps,
        start_step_count=start_step_count,
        time_step=time_step,
        one_step_fn=one_step_fn,
        boundary_conditions=boundary_conditions,
        values_transform_fn=values_transform_fn,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn,
        dtype=dtype,
        name=name)


def _backward_pde_coeffs(drift_fn, volatility_fn, discounting,
                         batch_shape, dim):
  """Returns coeffs of the backward PDE."""
  def second_order_coeff_fn(t, coord_grid):
    coord_grid = _broadcast_batch_shape(_coord_grid_to_mesh_grid(coord_grid),
                                        batch_shape, dim)
    sigma = volatility_fn(t, coord_grid)
    sigma_times_sigma_t = tf.linalg.matmul(sigma, sigma, transpose_b=True)
    # We currently have [dim, dim] as innermost dimensions, but the returned
    # tensor must have [dim, dim] as outermost dimensions.
    rank = len(sigma.shape.as_list())
    perm = [rank - 2, rank - 1] + list(range(rank - 2))
    sigma_times_sigma_t = tf.transpose(sigma_times_sigma_t, perm)
    return sigma_times_sigma_t / 2

  def first_order_coeff_fn(t, coord_grid):
    coord_grid = _broadcast_batch_shape(_coord_grid_to_mesh_grid(coord_grid),
                                        batch_shape, dim)
    mu = drift_fn(t, coord_grid)

    # We currently have [dim] as innermost dimension, but the returned
    # tensor must have [dim] as outermost dimension.
    rank = len(mu.shape.as_list())
    perm = [rank - 1] + list(range(rank - 1))
    mu = tf.transpose(mu, perm)
    return mu

  def zeroth_order_coeff_fn(t, coord_grid):
    coord_grid = _broadcast_batch_shape(_coord_grid_to_mesh_grid(coord_grid),
                                        batch_shape, dim)
    if not discounting:
      return None
    return -discounting(t, coord_grid)

  return second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn


def _coord_grid_to_mesh_grid(coord_grid):
  if len(coord_grid) == 1:
    return tf.expand_dims(coord_grid[0], -1)
  return tf.stack(values=tf.meshgrid(*coord_grid, indexing='ij'), axis=-1)


# TODO(b/192220570): Move this to the utility module
def _get_static_shape(t):
  t_shape = t.shape.as_list()
  if None in t_shape:
    t_shape = tf.shape(t)
  return t_shape


def _broadcast_batch_shape(x, batch_shape, dim):
  # `x` is of shape `batch_shape + [d1,.., dn, dim]`, where n == dim
  x_shape_no_batch = _get_static_shape(x)[-dim - 1:]
  if isinstance(x_shape_no_batch, list) and isinstance(batch_shape, list):
    return tf.broadcast_to(x, batch_shape + x_shape_no_batch)
  else:
    output_shape = tf.concat([batch_shape, x_shape_no_batch], axis=0)
    return tf.broadcast_to(x, output_shape)
