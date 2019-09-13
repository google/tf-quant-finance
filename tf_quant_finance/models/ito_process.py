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
"""Defines Ito processes.

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

### References:
  [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

from tf_quant_finance.math import random


@six.add_metaclass(abc.ABCMeta)
class ItoProcess(object):
  """Base class for Ito processes.

    Represents a general Ito process:

    ```None
      dX_i = a_i(t, X) dt + Sum(S_{ij}(t, X) dW_j for 1 <= j <= n), 1 <= i <= n
    ```

    `a_i(t, S)` is the instantaneous drift rate and the `S_{ij}(t)` is the
    volatility matrix. While this is how the process is usually presented
    mathematically, the initializer requires a slightly different
    specification. Instead of the instantaneous drift and volatility, it is
    computationally more efficient to use the total drift and covariance. These
    are defined as:

    ```None
      total_drift_{i}(t1, t2, X) = Integrate(a_{i}(t, X), t1 <= t <= t2)
      total_covariance_{ij}(t1, t2, X) = Integrate(inst_covariance_{ij}(t, X),
                                                t1 <= t <= t2)
      inst_covariance_{ij}(t, X) = (S.S^T)_{ij}
    ```

  ### Example. 2-dimensional Ito process of the form

  ```none
  dX_1 = mu_1 * sqrt(t) dt + s11 * dW_1 + s12 * dW_2
  dX_2 = mu_2 * sqrt(t) dt + s21 * dW_1 + s22 * dW_2
  ```

  class SimpleItoProcess(ito_process.ItoProcess):
    def __init__(self, dim, drift_fn, vol_fn, dtype=tf.float64):
      self._dim = dim
      self._drift_fn = drift_fn
      self._vol_fn = vol_fn
      self._dtype = dtype

    def dim(self):
      return self._dim

    def drift_fn(self):
      return self._drift_fn

    def volatility_fn(self):
      return self._vol_fn

    def dtype(self):
      return self._dtype

    def name(self):
      return 'ito_process'

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

  # Initialize `SimpleItoProcess`
  process = TestItoProcess(dim=2, drift_fn=drift_fn, vol_fn=vol_fn, dtype=dtype)
  # Set starting location
  x0 = np.array([0.1, -1.1])
  # Sample `num_samples` paths at specified `times` locations using built-in
  # Euler scheme.
  paths = process.sample_paths(
            times,
            num_samples=num_samples,
            initial_state=x0,
            grid_step=0.01,
            seed=42)
  """

  @abc.abstractmethod
  def dim(self):
    """The dimension of the process."""
    return None

  @abc.abstractmethod
  def dtype(self):
    """The data type of process realizations."""
    return None

  @abc.abstractmethod
  def name(self):
    """The name to give to the ops created by this class."""
    return None

  def drift_fn(self):
    """Python callable calculating instantaneous drift.

    The callable should accept two real `Tensor` arguments of the same dtype.
    The first argument is the scalar time t, the second argument is the value of
    Ito process X - tensor of shape `batch_shape + [dim]`. The result is value
    of drift a(t, X). The return value of the callable is a real `Tensor` of the
    same dtype as the input arguments and of shape `batch_shape + [dim]`.
    """
    return None

  def volatility_fn(self):
    """Python callable calculating the instantaneous volatility.

    The callable should accept two real `Tensor` arguments of the same dtype and
    shape `times_shape`. The first argument is the scalar time t, the second
    argument is the value of Ito process X - tensor of shape `batch_shape +
    [dim]`. The result is value of drift S(t, X). The return value of the
    callable is a real `Tensor` of the same dtype as the input arguments and of
    shape `batch_shape + [dim, dim]`.
    """
    return None

  def total_drift_fn(self):
    """Computes integrated drift of the process between two times.

    Returns:
      A Python callable returning the integrated drift rate
      between two times. The callable accepts three real `Tensor`
      arguments. The first argument is the left end point
      and the second is the right end point of the time interval for which
      the total drift is needed. The third argument contains the values of
      the state at which the drift is to be computed. In most cases where
      the instantaneous drift depends on the state the time step for which
      the total drift is needed is very small. However, the interface allows
      asking for a finite time step at a fixed value of the state (i.e. the
      state at the start of the time interval). Implementations
      are free to either raise an error for larger time steps or give the best
      approximation they can for the given parameters. The two main advantages
      of working with total drifts/covariances vs instantaneous quantities are
        (a) If needed, the instantaneous values can be derived using either
          automatic differentiation or finite difference.
        (b) For cases where there is no non-trivial state dependence
          (e.g. Brownian motion with time dependent drifts and covariance) or
          the state dependence can be easily removed by a transformation of the
          process (e.g. geometric brownian motion), it
          is possible to directly generate samples from the joint distributions
          of the process and these depend only on the total drift and
          covariances.

      The precise definition of total drift function is as follows.
      Consider a general `dim` dimensional Ito process:

      ```None
        dX_i = mu_i(t, X) dt + Sum(S_{ij}(t, X) dW_j for 1 <= j <= dim)

      ```

      `mu_i(t, S)` is the instantaneous drift rate and the `S_{ij}(t)` is the
      volatility matrix. The total drift function `M_{ij}(t1, t2, X)`
      is then defined by

      ```None
        M_{i}(t1, t2, X) = Integral[ mu_{i}(t, X), t1 <= t <= t2]

      ```
      The shapes of the inputs and outputs of the callable are as follows:

      Input `Tensor`s.
        1. `start time` and `end time` are of shape `time_batch_shape`.
          Here `time_batch_shape` is the shape of the times `Tensor` at which
          to evaluate the integrated drift.
        2. The state is of shape `batch_shape + time_batch_shape + [dim]` where
          `dim` and `batch_shape` are the dimension and batch shape of the
          process.

      The output of this callable is a `Tensor` of shape
      `batch_shape + time_batch_shape + [dim]` containing the integrated drift
      between the start times and end times.
    """
    return None

  def total_covariance_fn(self):
    """The total covariance of the process between two times.

      A Python callable returning the integrated covariances between two times.
      The callable accepts three real `Tensor` arguments. The first argument is
      the left end point and the second is the right end point of the time
      interval for which the total covariance is needed. The third argument
      contains the values of the state at which the covariance is to be
      computed. In most cases where the instantaneous covariance depends on
      the state the time step for which the total covariance is needed is
      small (e.g. while doing finite differences or generating
      Monte Carlo paths). However, the interface allows asking for a finite
      time step at a fixed value of the state (i.e. the state at the start
      of the time interval). Implementations are free to either raise an error
      for larger time steps or give the best approximation they can for the
      given parameters. The two main advantages of working with total
      covariances/drifts vs instantaneous quantities are
        (a) If needed, the instantaneous values can be derived using either
          automatic differentiation or finite difference.
        (b) For cases where there is no non-trivial state dependence
          (e.g. Brownian motion with time dependent drifts and covariance)  or
          the state dependence can be easily removed by a transformation of the
          process (e.g. geometric brownian motion), it
          is possible to directly generate samples from the joint distributions
          of the process and these depend only on the total drift and
          covariances.

      The precise definition of total covariance function is as follows.
      Consider a general `dim` dimensional Ito process:

      ```None
        dX_i = mu_i(t, X) dt + Sum(S_{ij}(t, X) dW_j for 1 <= j <= dim)

      ```

      `mu_i(t, S)` is the instantaneous drift rate and the `S_{ij}(t)` is the
      volatility matrix. The total covariance function `V_{ij}(t1, t2, X)`
      is then defined by

      ```None
        V_{ij}(t1, t2, X) = Integral[ (S.S^T)_{ij}(t), t1 <= t <= t2]

      ```
      The shapes of the inputs and outputs of the callable are as follows:

      Input `Tensor`s.
        1. `start time` and `end time` are of shape `time_batch_shape`.
          Here `time_batch_shape` is the shape of the times `Tensor` at which
          to evaluate the integrated covariances.
        2. The state is of shape `batch_shape + time_batch_shape + [dim]` where
          `dim` and `batch_shape` are the dimension and batch shape of the
          process.

      The output of this callable is a `Tensor` of shape
      `batch_shape + time_batch_shape + [dim]` containing the integrated
      covariances between the start times and end times.
    """
    return None

  def sample_paths(self,
                   times,
                   num_samples=1,
                   initial_state=None,
                   random_type=None,
                   seed=None,
                   swap_memory=True,
                   name=None,
                   **kwargs):
    """Returns a sample of paths from the process.

    The default implementation uses Euler schema. However, for particular types
    of Ito processes more efficient schemes can be used.

    Args:
      times: Rank 1 `Tensor` of increasing positive real values. The times at
        which the path points are to be evaluated.
      num_samples: Positive scalar `int`. The number of paths to draw.
      initial_state: `Tensor` of shape `[dim]`. The initial state of the
        process.
        Default value: None which maps to a zero initial state.
      random_type: Enum value of `RandomType`. The type of (quasi)-random
        number generator to use to generate the paths.
        Default value: None which maps to the standard pseudo-random numbers.
      seed: Python `int`. The random seed to use. If not supplied, no seed is
        set.
      swap_memory: Whether GPU-CPU memory swap is enabled for this op. See
        equivalent flag in `tf.while_loop` documentation for more details.
        Useful when computing a gradient of the op since `tf.while_loop` is
        used to propagate stochastic process in time.
      name: str. The name to give this op. If not supplied, default name of
        `sample_paths` is used.
      **kwargs: parameters, specific to Euler schema:
        `grid_step` is rank 0 real `Tensor` - maximal distance between points
          in grid in Euler schema.

    Returns:
     A real `Tensor` of shape [num_samples, k, n] where `k` is the size of the
        `times`, `n` is the dimension of the process.
    """
    if self.drift_fn() is None or self.volatility_fn() is None:
      raise NotImplementedError(
          'In order to use Euler scheme, both drift_fn and volatility_fn '
          'should be provided.')
    default_name = self.name() + '_sample_paths'
    with tf.compat.v1.name_scope(
        name, default_name=default_name, values=[times, initial_state]):
      if initial_state is None:
        initial_state = tf.zeros(self._dim, dtype=self._dtype)
      times = tf.convert_to_tensor(times, dtype=self._dtype)
      initial_state = tf.convert_to_tensor(initial_state, dtype=self._dtype,
                                           name='initial_state')
      times_size = tf.shape(times)[-1]
      grid_step = kwargs['grid_step']
      times, keep_mask = self._prepare_grid(times, grid_step)
      return self._sample_paths(
          times, grid_step, keep_mask, times_size, num_samples, initial_state,
          random_type, seed, swap_memory)

  def fd_solver_backward(self,
                         final_time,
                         discounting_fn=None,
                         grid_spec=None,
                         time_step=None,
                         time_step_fn=None,
                         values_batch_size=1,
                         name=None,
                         **kwargs):
    """Returns a solver for solving Feynman-Kac PDE associated to the process.

    Constructs a finite differences grid solver for solving the final value
    problem as it appears in the Feynman-Kac formula associated to this Ito
    process. The Feynman-Kac PDE is closely related to the backward Kolomogorov
    equation associated to the stochastic process and allows for the inclusion
    of a discount rate.

    A limited version of Feynman-Kac formula states the following.
    Consider the expectation

    ```None
      V(x, t, T) = E_Q[e^{-R(t, T)} f(X_T) | X_t = x]
    ```

    where `Q` is a probability measure under which `W_j` is an n-dimensional
    Wiener process, `X` is an n-dimensional Ito process satisfying:

    ```None
      dX_i = mu_i dt + Sum[Sigma_{ij} dW_j, 1 <= j <= n]
    ```

    and `R(t, T)` is a positive stochastic process given by:

    ```None
      R(t,T) = Integral[ r(s, X_s) ds, t <= s <= T]
    ```

    This expectation is the solution of the following second order linear
    partial differential equation:

    ```None
      V_t + Sum[mu_i(t, x) V_i, 1<=i<=n] +
        (1/2) Sum[ D_{ij} V_{ij}, 1 <= i,j <= n] - r(t, x) V = 0
    ```

    In the above, `V_t` is the derivative of `V` with respect to `t`,
    `V_i` is the partial derivative with respect to `x_i` and `V_{ij}` the
    (mixed) partial derivative with respect to `x_i` and `x_j`. `D_{ij}` are
    the components of the diffusion tensor:

    ```None
      D_{ij} = (Sigma . Transpose[Sigma])_{ij}
    ```

    This method provides a finite difference solver to solve the above
    differential equation. Whereas the coefficients `mu` and `D` are properties
    of the SDE itself, the function `r(t, x)` may be arbitrarily specified
    by the user (the parameter `discounting_fn` to this method).

    Args:
      final_time: Positive scalar real `Tensor`. The time of the final value.
        The solver is initialized to this final time.
      discounting_fn: Python callable corresponding to the function `r(t, x)` in
        the description above. The callable accepts two positional arguments.
        The first argument is the time at which the discount rate function is
        needed. The second argument contains the values of the state at which
        the discount is to be computed.
        Default value: None which maps to `r(t, x) = 0`.
      grid_spec: An iterable convertible to a tuple containing at least the
        attributes named 'grid', 'dim' and 'sizes'. For a full description of
        the fields and expected types, see `grids.GridSpec` which provides the
        canonical specification of this object.
      time_step: A real positive scalar `Tensor` or None. The fixed
        discretization parameter along the time dimension. Either this argument
        or the `time_step_fn` must be specified. It is an error to specify both.
        Default value: None.
      time_step_fn: A callable accepting an instance of `grids.GridStepperState`
        and returning the size of the next time step as a real scalar tensor.
        This argument allows usage of a non-constant time step while stepping
        back. If not specified, the `time_step` parameter must be specified. It
        is an error to specify both.
        Default value: None.
      values_batch_size: A positive Python int. The batch size of values to be
        propagated simultaneously.
        Default value: 1.
      name: Python str. The name to give this op.
        Default value: None which maps to `fd_solver_backward`.
      **kwargs: Any other keyword args needed.

    Returns:
      An instance of `BackwardGridStepper` configured for solving the
      Feynman-Kac PDE associated to this process.
    """
    raise NotImplementedError('Backward Finite difference solver not '
                              'implemented for ItoProcess.')

  def _sample_paths(self, times, grid_step, keep_mask, times_size, num_samples,
                    initial_state, random_type, seed, swap_memory):
    """Returns a sample of paths from the process."""
    dt = times[1:] - times[:-1]
    sqrt_dt = tf.sqrt(dt)
    current_state = initial_state + tf.zeros([num_samples, self.dim()],
                                             dtype=initial_state.dtype)
    steps_num = tf.shape(dt)[-1]
    wiener_mean = tf.zeros((self.dim(), 1), dtype=self._dtype)

    cond_fn = lambda i, *args: i < steps_num

    def step_fn(i, written_count, current_state, result):
      """Performs one step of Euler scheme."""
      current_time = times[i + 1]
      dw = random.mv_normal_sample(
          (num_samples,), mean=wiener_mean, random_type=random_type,
          seed=seed)
      dw = dw * sqrt_dt[i]
      dt_inc = dt[i] * self.drift_fn()(current_time, current_state)  # pylint: disable=not-callable
      dw_inc = tf.squeeze(
          tf.matmul(self.volatility_fn()(current_time, current_state), dw), -1)  # pylint: disable=not-callable
      next_state = current_state + dt_inc + dw_inc

      # Keep only states for times, requested by user.
      result = tf.cond(keep_mask[i + 1],
                       (lambda: result.write(written_count, next_state)),
                       (lambda: result))
      written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
      return (i + 1, written_count, next_state, result)

    # Maximum number iterations is passed to the while loop below. It improves
    # performance of the while loop on a GPU and is needed for XLA-compilation
    # comptatiblity
    maximum_iterations = (tf.cast(1. / grid_step, dtype=tf.int32)
                          + tf.size(times))
    result = tf.TensorArray(dtype=self._dtype, size=times_size)
    _, _, _, result = tf.compat.v1.while_loop(
        cond_fn, step_fn, (0, 0, current_state, result),
        maximum_iterations=maximum_iterations,
        swap_memory=swap_memory)

    return tf.transpose(result.stack(), (1, 0, 2))

  def _prepare_grid(self, times, grid_step):
    """Prepares grid of times for path generation.

    Args:
      times:  Rank 1 `Tensor` of increasing positive real values. The times at
        which the path points are to be evaluated.
      grid_step: Rank 0 real `Tensor`. Maximal distance between points in
        resulting grid.

    Returns:
      Tuple `(all_times, mask)`.
      `all_times` is 1-D real `Tensor` containing all points from 'times` and
      whose intervals are at most `grid_step`.
      `mask` is a boolean 1-D tensor of the same shape as 'all_times', showing
      which elements of 'all_times' correspond to values from `times`.
      Guarantees that times[0]=0 and grid_step[0]=False.
      'all_times` is sorted ascending and may contain duplicates.
    """
    grid = tf.range(0.0, times[-1], grid_step, dtype=self._dtype)
    all_times = tf.concat([grid, times], axis=0)
    mask = tf.concat([
        tf.zeros_like(grid, dtype=tf.bool),
        tf.ones_like(times, dtype=tf.bool)
    ],
                     axis=0)
    perm = tf.argsort(all_times, stable=True)
    all_times = tf.gather(all_times, perm)
    mask = tf.gather(mask, perm)
    return all_times, mask
