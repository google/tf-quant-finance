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
"""Defines an interface for Ito processes.

Ito processes underlie most quantitative finance models. This module defines
a framework for describing Ito processes. An Ito process is usually defined
via an Ito SDE:

```
  dX = a(t, X_t) dt + b(t, X_t) dW_t

```

where `a(t, x)` is a function taking values in `R^n`, `b(t, X_t)` is a function
taking values in `n x n` matrices. For a complete mathematical definition,
including the regularity conditions that must be imposed on the coefficients
`a(t, X)` and `b(t, X)`, see Ref [1].

#### References:
  [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.
"""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ItoProcess:
  """Interface for specifying Ito processes.

    Interface for defining stochastic process defined by the Ito SDE:

    ```None
      dX_i = a_i(t, X) dt + Sum(S_{ij}(t, X) dW_j for 1 <= j <= n), 1 <= i <= n
    ```

  The vector coefficient `a_i` is referred to as the drift of the process and
  the matrix `S_{ij}` as the volatility of the process. For the process to be
  well defined, these coefficients need to satisfy certain technical conditions
  which may be found in Ref. [1]. The vector `dW_j` represents independent
  Brownian increments.

  For a simple and instructive example of the implementation of this interface,
  see `models.GenericItoProcess`.

  #### References
  [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.
  """

  @abc.abstractmethod
  def name(self):
    """The name to give to ops created by this class."""
    pass

  @abc.abstractmethod
  def dim(self):
    """The dimension of the process. A positive python integer."""
    pass

  @abc.abstractmethod
  def dtype(self):
    """The data type of process realizations."""
    pass

  @abc.abstractmethod
  def drift_fn(self):
    """Python callable calculating instantaneous drift.

    The callable should accept two real `Tensor` arguments of the same dtype.
    The first argument is the scalar time t, the second argument is the value of
    Ito process X as a tensor of shape `batch_shape + [dim]`. The result is
    value of drift a(t, X). The return value of the callable is a real `Tensor`
    of the same dtype as the input arguments and of shape `batch_shape + [dim]`.
    """
    pass

  @abc.abstractmethod
  def volatility_fn(self):
    """Python callable calculating the instantaneous volatility matrix.

    The callable should accept two real `Tensor` arguments of the same dtype.
    The first argument is the scalar time t and the second argument is the value
    of Ito process X as a tensor of shape `batch_shape + [dim]`. The result is
    the instantaneous volatility matrix at time t and location X: S(t, X). The
    return value of the callable is a real `Tensor` of the same dtype as the
    input arguments and of shape `batch_shape + [dim, dim]`.
    """
    pass

  @abc.abstractmethod
  def sample_paths(self,
                   times,
                   num_samples=1,
                   initial_state=None,
                   random_type=None,
                   seed=None,
                   **kwargs):
    """Returns a sample of paths from the process.

    Args:
      times: A `Tensor` of positive real values of a shape `[T, k]`, where
        `T` is either empty or a shape which is broadcastable to `batch_shape`
        (as defined when this ItoProcess was initialised) and `k` is the
        number of time points. The definition of `batch_shape` and how it is set
        is dictated by the individual child classes. Typically `batch_shape`
        is set based on the shapes of the input parameters that define the
        ItoProcess, see `GeometricBrownianMotion` for an example. The times at
        which the path points are to be evaluated.
      num_samples: Positive scalar `int`. The number of paths to draw.
      initial_state: `Tensor` of shape `[dim]`. The initial state of the
        process.
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
      **kwargs: Any other keyword args needed by an implementation.

    Returns:
     A real `Tensor` of shape [`batch_shape`, num_samples, k, n] where `k` is
     the size of `times`, `n` is the dimension of the process and `batch_shape`
     is the batch dimensions of the ItoProcess.
    """
    pass

  @abc.abstractmethod
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
    [K, d1, d2, ... dn] where `K` is the batch size. This method only requires
    that the input parameter `values_grid` be broadcastable with shape
    [K, d1, ... dn].

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

    TODO(b/142309558): Complete documentation.

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
            linear homogenous term (i.e. `r(t,x)` above). Same spec as above.
            The `one_step_fn` callable returns a 2-tuple containing the next
            coordinate grid, next values grid.
      boundary_conditions: A list of size `dim` containing boundary conditions.
        The i'th element of the list is a 2-tuple containing the lower and upper
        boundary condition for the boundary along the i`th axis.
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
      **kwargs: Any other keyword args needed by an implementation.

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
    pass

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

    Batching of solutions is supported. In this context, batching means
    the ability to represent and evolve multiple independent functions `V`
    (e.g. V1, V2 ...) simultaneously. A single discretized solution is specified
    by stating its values at each grid point. This can be represented as a
    `Tensor` of shape [d1, d2, ... dn] where di is the grid size along the `i`th
    axis. A batch of such solutions is represented by a `Tensor` of shape:
    [K, d1, d2, ... dn] where `K` is the batch size. This method only requires
    that the input parameter `values_grid` be broadcastable with shape
    [K, d1, ... dn].

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

    TODO(b/142309558): Complete documentation.

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
            linear homogenous term (i.e. `r(t,x)` above). Same spec as above.
            The `one_step_fn` callable returns a 2-tuple containing the next
            coordinate grid, next values grid.
      boundary_conditions: A list of size `dim` containing boundary conditions.
        The i'th element of the list is a 2-tuple containing the lower and upper
        boundary condition for the boundary along the i`th axis.
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
      **kwargs: Any other keyword args needed by an implementation.

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
    pass
