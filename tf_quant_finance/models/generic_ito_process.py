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

import tensorflow as tf

from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import ito_process
from tf_quant_finance.math.pde import fd_solvers


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

    ### Example. Sampling from 2-dimensional Ito process of the form:

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

    ### References
    [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
      Applications. Springer. 2010.

    Args:
      dim: Python int greater than or equal to 1. The dimension of the Ito
        process.
      drift_fn: A Python callable to compute the drift of the process. The
        callable should accept two real `Tensor` arguments of the same dtype.
        The first argument is the scalar time t, the second argument is the
        value of Ito process X - `Tensor` of shape `batch_shape + [dim]`. The
        result is value of drift a(t, X). The return value of the callable is a
        real `Tensor` of the same dtype as the input arguments and of shape
        `batch_shape + [dim]`.
      volatility_fn: A Python callable to compute the volatility of the process.
        The callable should accept two real `Tensor` arguments of the same dtype
        and shape `times_shape`. The first argument is the scalar time t, the
        second argument is the value of Ito process X - `Tensor` of shape
        `batch_shape + [dim]`. The result is value of volatility S_{ij}(t, X).
        The return value of the callable is a real `Tensor` of the same dtype as
        the input arguments and of shape `batch_shape + [dim, dim]`.
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
                   time_step=None):
    """Returns a sample of paths from the process using Euler sampling.

    The default implementation uses the Euler scheme. However, for particular
    types of Ito processes more efficient schemes can be used.

    Args:
      times: Rank 1 `Tensor` of increasing positive real values. The times at
        which the path points are to be evaluated.
      num_samples: Positive scalar `int`. The number of paths to draw.
        Default value: 1.
      initial_state: `Tensor` of shape `[dim]`. The initial state of the
        process.
        Default value: None which maps to a zero initial state.
      random_type: Enum value of `RandomType`. The type of (quasi)-random number
        generator to use to generate the paths.
        Default value: None which maps to the standard pseudo-random numbers.
      seed: Python `int`. The random seed to use. If not supplied, no seed is
        set.
      swap_memory: A Python bool. Whether GPU-CPU memory swap is enabled for
        this op. See an equivalent flag in `tf.while_loop` documentation for
        more details. Useful when computing a gradient of the op since
        `tf.while_loop` is used to propagate stochastic process in time.
        Default value: True.
      name: Python string. The name to give this op.
        Default value: `None` which maps to `sample_paths` is used.
      time_step: Real scalar `Tensor`. The maximal distance between time points
        in grid in Euler scheme.

    Returns:
     A real `Tensor` of shape `[num_samples, k, n]` where `k` is the size of the
     `times`, and `n` is the dimension of the process.
    """
    default_name = self._name + '_sample_path'
    with tf.compat.v1.name_scope(
        name, default_name=default_name, values=[times, initial_state]):
      return euler_sampling.sample(
          self._dim,
          self._drift_fn,
          self._volatility_fn,
          times,
          num_samples=num_samples,
          initial_state=initial_state,
          random_type=random_type,
          time_step=time_step,
          seed=seed,
          swap_memory=swap_memory,
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
    """See base class."""
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
    """See base class."""
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
    sigma = volatility_fn(t, _coord_grid_to_mesh_grid(coord_grid))
    sigma_times_sigma_t = tf.linalg.matmul(sigma, sigma, transpose_b=True)

    # We currently have [dim, dim] as innermost dimensions, but the returned
    # tensor must have [dim, dim] as outermost dimensions.
    rank = len(sigma.shape.as_list())
    perm = [rank - 2, rank - 1] + list(range(rank - 2))
    sigma_times_sigma_t = tf.transpose(sigma_times_sigma_t, perm)
    return sigma_times_sigma_t / 2

  def first_order_coeff_fn(t, coord_grid):
    mu = drift_fn(t, _coord_grid_to_mesh_grid(coord_grid))

    # We currently have [dim] as innermost dimension, but the returned
    # tensor must have [dim] as outermost dimension.
    rank = len(mu.shape.as_list())
    perm = [rank - 1] + list(range(rank - 1))
    mu = tf.transpose(mu, perm)
    return mu

  def zeroth_order_coeff_fn(t, coord_grid):
    if not discounting:
      return None
    return -discounting(t, _coord_grid_to_mesh_grid(coord_grid))

  return second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn


def _coord_grid_to_mesh_grid(coord_grid):
  if len(coord_grid) == 1:
    return tf.expand_dims(coord_grid[0], -1)
  return tf.stack(values=tf.meshgrid(*coord_grid, indexing='ij'), axis=-1)
