# Lint as: python3
# Copyright 2021 Google LLC
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

"""Geometric Brownian Motion model."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math import piecewise as pw
from tf_quant_finance.math.pde import fd_solvers
from tf_quant_finance.models import ito_process
from tf_quant_finance.models import utils


class GeometricBrownianMotion(ito_process.ItoProcess):
  """Geometric Brownian Motion.

  Represents the 1-dimensional Ito process:

  ```None
    dX(t) = means(t) * X(t) * dt + volatilities(t) * X(t) * dW(t),
  ```

  where `W(t)` is a 1D Brownian motion, `mean(t)` and `volatility(t)` are either
  constant `Tensor`s or piecewise constant functions of time.

  Supports batching which enables modelling multiple univariate geometric
  brownian motions (GBMs) efficiently. No guarantee is made about the
  relationships between the batched univariate GMBs. To control the correlation
  between multiple GBMs use `MultivariateGeometricBrownianMotion`.

  ## Example

  ```python
  import tensorflow as tf
  import tf_quant_finance as tff
  process = tff.models.GeometricBrownianMotion(0.05, 1.0, dtype=tf.float64)
  times = [0.1, 0.2, 1.0]
  # Use SOBOL sequence to draw trajectories
  samples_sobol = process.sample_paths(
      times=times,
      initial_state=1.5,
      random_type=tff.math.random.RandomType.SOBOL,
      num_samples=100000)

  # You can also supply the random normal draws directly to the sampler
  normal_draws = tf.random.stateless_normal(
      [100000, 3, 1], seed=[4, 2], dtype=tf.float64)
  samples_custom = process.sample_paths(
      times=times,
      initial_state=1.5,
      normal_draws=normal_draws)
  ```
  """

  def __init__(self,
               mean,
               volatility,
               dtype=None,
               name=None):
    """Initializes the Geometric Brownian Motion.

    Args:
      mean: A real `Tensor` broadcastable to `batch_shape + [1]` or an instance
        of left-continuous `PiecewiseConstantFunc` with `batch_shape + [1]`
        dimensions. Here `batch_shape` represents a batch of independent
        GBMs. Corresponds to the mean drift of the Ito process.
      volatility: A real `Tensor` broadcastable to `batch_shape + [1]` or an
        instance of left-continuous `PiecewiseConstantFunc` of the same `dtype`
        and `batch_shape` as set by `mean`. Corresponds to the volatility of the
        process and should be positive.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred from
          `mean` is used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        'geometric_brownian_motion'.
    """
    self._name = name or 'geometric_brownian_motion'
    with tf.name_scope(self._name):
      self._mean, self._mean_is_constant = pw.convert_to_tensor_or_func(
          mean, dtype=dtype, name='mean')
      self._dtype = dtype or self._mean.dtype
      (
          self._volatility,
          self._volatility_is_constant
      ) = pw.convert_to_tensor_or_func(volatility, dtype=self._dtype,
                                       name='volatility')
      self._volatility_squared = self._volatility_squared_from_volatility(
          self._volatility,
          self._volatility_is_constant,
          dtype=self._dtype,
          name='volatility_squared')
      self._dim = 1

  def dim(self):
    """The dimension of the process."""
    return self._dim

  def dtype(self):
    """The data type of process realizations."""
    return self._dtype

  def name(self):
    """The name to give to ops created by this class."""
    return self._name

  def drift_is_constant(self):
    """Returns True if the drift of the process is a constant."""
    return self._mean_is_constant

  def volatility_is_constant(self):
    """Returns True is the volatility of the process is a constant."""
    return self._volatility_is_constant

  def drift_fn(self):
    """Python callable calculating instantaneous drift."""
    def _constant_fn(t, x):
      """Drift function of the GBM with constant mean."""
      del t
      return self._mean * x

    def _piecewise_fn(t, x):
      """Drift function of the GBM with piecewise constant mean."""
      return self._mean(t) * x

    return _constant_fn if self.drift_is_constant() else _piecewise_fn

  def volatility_fn(self):
    """Python callable calculating the instantaneous volatility."""
    def _constant_fn(t, x):
      """Volatility function of the GBM with constant volatility."""
      del t
      vol = self._volatility * tf.expand_dims(x, -1)
      return vol

    def _piecewise_fn(t, x):
      """Volatility function of the GBM with piecewise constant volatility."""
      vol = self._volatility(t) * tf.expand_dims(x, -1)
      return vol

    return _constant_fn if self.volatility_is_constant() else _piecewise_fn

  def sample_paths(self,
                   times,
                   initial_state=None,
                   num_samples=1,
                   random_type=None,
                   seed=None,
                   skip=0,
                   normal_draws=None,
                   name=None):
    """Returns a sample of paths from the process.

    If `mean` and `volatility` were specified with batch dimensions the sample
    paths will be generated for all batch dimensions for the specified `times`
    using a single set of random draws.

    Args:
      times: A `Tensor` of positive real values of a shape `[T, k]`, where
        `T` is either empty or a shape which is broadcastable to `batch_shape`
        (as defined by the shape of `mean` or `volatility` which were set when
        this instance of GeometricBrownianMotion was initialised) and `k` is the
        number of time points. The times at which the path points are to be
        evaluated.
      initial_state: A `Tensor` of the same `dtype` as `times` and of shape
        broadcastable to `[batch_shape, num_samples]`. Represents the initial
        state of the Ito process.
        Default value: `None` which maps to a initial state of ones.
      num_samples: Positive scalar `int`. The number of paths to draw.
      random_type: Enum value of `RandomType`. The type of (quasi)-random
        number generator to use to generate the paths.
        Default value: None which maps to the standard pseudo-random numbers.
      seed: Seed for the random number generator. The seed is
        only relevant if `random_type` is one of
        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
        `HALTON_RANDOMIZED` the seed should be an Python integer. For
        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
        `Tensor` of shape `[2]`.
        Default value: `None` which means no seed is set.
      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
        Halton sequence to skip. Used only when `random_type` is 'SOBOL',
        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
        Default value: 0.
      normal_draws: A `Tensor` of shape `[num_samples, num_time_points, 1]`
        and the same `dtype` as `times`. Represents random normal draws to
        compute increments `N(0, t_{n+1}) - N(0, t_n)`. When supplied,
        `num_samples` argument is ignored and the first dimensions of
        `normal_draws` is used instead. `num_time_points` should be equal to
        `tf.shape(times)[0]`.
        Default value: `None` which means that the draws are generated by the
        algorithm.
      name: Str. The name to give this op.
        Default value: `sample_paths`.

    Returns:
      A `Tensor`s of shape [batch_shape, num_samples, k, 1] where `k` is the
      the number of `time points`.

    Raises:
      ValueError: If `normal_draws` is supplied and does not have shape
      broadcastable to `[num_samples, num_time_points, 1]`.
    """
    name = name or (self._name + '_sample_path')

    with tf.name_scope(name):
      times = tf.convert_to_tensor(times, self._dtype)
      if normal_draws is not None:
        normal_draws = tf.convert_to_tensor(normal_draws, dtype=self._dtype,
                                            name='normal_draws')
      initial_state = utils.convert_to_tensor_with_default(
          initial_state,
          tf.ones([1], dtype=self._dtype),
          dtype=self._dtype,
          name='initial_state')

      num_requested_times = times.shape[-1]
      return self._sample_paths(
          times=times,
          num_requested_times=num_requested_times,
          initial_state=initial_state,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
          skip=skip,
          normal_draws=normal_draws)

  def _integrate_parameter(self, x, x_is_constant, t0, t1, name=None):
    """Returns the integral of x(t).dt over the interval [t0, t1].

    Args:
      x: Scalar real `Tensor` of shape [`batch_shape`] or an instance of a
        left-continuous `PiecewiseConstantFunc`. The function to be integrated.
      x_is_constant: 'bool' which is True if x is a Scalar real `Tensor`.
      t0: A `Tensor` which is broadcastable to [`batch_shape`, `k`], where `k`
        is the number of intervals to evaluate the integral over. The start
        times of the `k` intervals.
      t1: A `Tensor` which is broadcastable to [`batch_shape`, `k`], where `k`
        is the number of intervals to evaluate the integral over. The end
        times of the `k` intervals.
      name: Str. The name to give this op.

    Returns:
      A `Tensor` of shape [`batch_shape`, `k`] with the integrals of x over the
      intervals [`t0`, `t1`].
    """
    return x * (t1 - t0) if x_is_constant else x.integrate(t0, t1, name)

  def _sample_paths(self,
                    times,
                    num_requested_times,
                    initial_state,
                    num_samples,
                    random_type,
                    seed,
                    skip,
                    normal_draws):
    """Returns a sample of paths from the process."""
    if normal_draws is None:
      # Normal draws needed for sampling
      normal_draws = utils.generate_mc_normal_draws(
          num_normal_draws=1, num_time_steps=num_requested_times,
          num_sample_paths=num_samples, random_type=random_type,
          seed=seed,
          dtype=self._dtype, skip=skip)
    else:
      # Shape [num_time_points, num_samples, dim]
      normal_draws = tf.transpose(normal_draws, [1, 0, 2])
      num_samples = tf.shape(normal_draws)[1]
      draws_dim = normal_draws.shape[2]
      if draws_dim != 1:
        raise ValueError(
            '`dim` should be equal to `1` but is {0}'.format(draws_dim))
    # Create a set of zeros that is the right shape to add a '0' as the first
    # element for each series of times.
    zeros = tf.zeros(tf.concat([times.shape[:-1], [1]], 0), dtype=self._dtype)
    times = tf.concat([zeros, times], -1)
    mean_integral = self._integrate_parameter(
        self._mean, self._mean_is_constant, times[..., :-1], times[..., 1:])
    # mean_integral has shape [batch_shape, k-1], where self._mean has shape
    # [batch_shape, 1] and times has shape [k].
    mean_integral = tf.expand_dims(mean_integral, -2)
    volatility_sq_integral = self._integrate_parameter(
        self._volatility_squared, self._volatility_is_constant,
        times[..., :-1], times[..., 1:])
    volatility_sq_integral = tf.expand_dims(volatility_sq_integral, -2)
    # Giving mean_integral and volatility_sq_integral
    # shape = `batch_shape + [1, k-1]`,
    # where self._mean has shape `batch_shape + [1]` and times has shape `[k]`.

    # The logarithm of all the increments between the times.
    log_increments = ((mean_integral - volatility_sq_integral / 2)
                      + tf.sqrt(volatility_sq_integral)
                      * tf.transpose(tf.squeeze(normal_draws, -1)))
    # Since the implementation of tf.math.cumsum is single-threaded we
    # use lower-triangular matrix multiplication instead
    once = tf.ones([num_requested_times, num_requested_times],
                   dtype=self._dtype)
    lower_triangular = tf.linalg.band_part(once, -1, 0)
    cumsum = tf.linalg.matvec(lower_triangular,
                              log_increments)

    samples = tf.expand_dims(initial_state, [-1]) * tf.math.exp(cumsum)
    return tf.expand_dims(samples, -1)

  def _volatility_squared_from_volatility(
      self, volatility, volatility_is_constant, dtype=None, name=None):
    """Returns volatility squared as either a `PiecewiseConstantFunc` or a `Tensor`.

    Args:
      volatility: Either a 'Tensor' or 'PiecewiseConstantFunc'.
      volatility_is_constant: `bool` which is True if volatility is of type
        `Tensor`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        '_volatility_squared'.
    """
    name = name or (self._name + '_volatility_squared')
    if volatility_is_constant:
      return volatility ** 2
    else:
      return pw.PiecewiseConstantFunc(
          volatility.jump_locations(), volatility.values()**2,
          dtype=dtype, name=name)

  # TODO(b/152967694): Remove the duplicate methods.
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
      dV(t)/dt + mean(t, x) dV_i/dx
      + (1/2) volatility^2(t, x) d^2V_i/dx^2 - r(t, x) V = 0
    ```
    This method evolves a spatially discretized solution of the above PDE from
    time `t0` to time `t1 < t0` (i.e. backwards in time).
    The solution `V(t,x)` is assumed to be discretized on a grid.

    This method allows batching of solutions. In this context, batching means
    the ability to represent and evolve multiple independent functions `V`
    (e.g. V1, V2 ...) simultaneously corresponding to `mean_1, mean_2 ...` and
    `volatility_1, volatility_2 ....`.

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
            PDE (i.e. `mean_i(t, x)` above). Accepts time and coordinate grid as
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

    second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn = (
        _backward_pde_coeffs(self._drift_fn, self._volatility_fn, discounting))

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
      dV/dt + d(mean_i(t, x) V) / dx
        - (1/2) d^2(volatility^2(t, x) V) / dx^2 =  = 0
    ```

    with the initial value condition $$V(0, x) = u(x)$$.

    This method evolves a spatially discretized solution of the above PDE from
    time `t0` to time `t1 < t0` (i.e. backwards in time).
    The solution `V(t,x)` is assumed to be discretized on a grid.

    This method allows batching of solutions. In this context, batching means
    the ability to represent and evolve multiple independent functions `V`
    (e.g. V1, V2 ...) simultaneously corresponding to `mean_1, mean_2 ...` and
    `volatility_1, volatility_2 ....`.

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
            PDE (i.e. `mean_i(t, x)` above). Accepts time and coordinate grid as
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

    backward_second_order, backward_first_order, backward_zeroth_order = (
        _backward_pde_coeffs(self._drift_fn, self._volatility_fn,
                             discounting=None))

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


def _backward_pde_coeffs(drift_fn, volatility_fn, discounting):
  """Returns coeffs of the backward PDE."""
  def second_order_coeff_fn(t, coord_grid):
    volatility = volatility_fn(t, _coord_grid_to_mesh_grid(coord_grid))
    volatility_times_volatility_t = tf.linalg.matmul(
        volatility, volatility, transpose_b=True)

    # We currently have [dim, dim] as innermost dimensions, but the returned
    # tensor must have [dim, dim] as outermost dimensions.
    rank = len(volatility.shape.as_list())
    perm = [rank - 2, rank - 1] + list(range(rank - 2))
    volatility_times_volatility_t = tf.transpose(
        volatility_times_volatility_t, perm)
    return volatility_times_volatility_t / 2

  def first_order_coeff_fn(t, coord_grid):
    mean = drift_fn(t, _coord_grid_to_mesh_grid(coord_grid))

    # We currently have [dim] as innermost dimension, but the returned
    # tensor must have [dim] as outermost dimension.
    rank = len(mean.shape.as_list())
    perm = [rank - 1] + list(range(rank - 1))
    mean = tf.transpose(mean, perm)
    return mean

  def zeroth_order_coeff_fn(t, coord_grid):
    if not discounting:
      return None
    return -discounting(t, _coord_grid_to_mesh_grid(coord_grid))

  return second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn


def _coord_grid_to_mesh_grid(coord_grid):
  if len(coord_grid) == 1:
    return tf.expand_dims(coord_grid[0], -1)
  return tf.stack(values=tf.meshgrid(*coord_grid, indexing='ij'), axis=-1)
