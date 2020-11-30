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

"""Heston model with piecewise constant parameters."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import generic_ito_process
from tf_quant_finance.models import utils


_SQRT_2 = np.sqrt(2., dtype=np.float64)


class HestonModel(generic_ito_process.GenericItoProcess):
  """Heston Model with piecewise constant parameters.

  Represents the Ito process:

  ```None
    dX(t) = -V(t) / 2 * dt + sqrt(V(t)) * dW_{X}(t),
    dV(t) = kappa(t) * (theta(t) - V(t)) * dt
            + epsilon(t) * sqrt(V(t)) * dW_{V}(t)
  ```

  where `W_{X}` and `W_{V}` are 1D Brownian motions with a correlation
  `rho(t)`. `kappa`, `theta`, `epsilon`, and `rho` are positive piecewise
  constant functions of time. Here `V(t)` represents the process variance at
  time `t` and `X` represents logarithm of the spot price at time `t`.

  `kappa` corresponds to the mean reversion rate, `theta` is the long run
  price variance, and `epsilon` is the volatility of the volatility.

  See [1] and [2] for details.

  #### Example

  ```python
  import tf_quant_finance as tff
  import numpy as np
  epsilon = PiecewiseConstantFunc(
      jump_locations=[0.5], values=[1, 1.1], dtype=np.float64)
  process = HestonModel(kappa=0.5, theta=0.04, epsilon=epsilon, rho=0.1,
                        dtype=np.float64)
  times = np.linspace(0.0, 1.0, 1000)
  num_samples = 10000  # number of trajectories
  sample_paths = process.sample_paths(
      times,
      time_step=0.01,
      num_samples=num_samples,
      initial_state=np.array([1.0, 0.04]),
      random_type=random.RandomType.SOBOL)
  ```

  #### References:
    [1]: Cristian Homescu. Implied volatility surface: construction
      methodologies and characteristics.
      arXiv: https://arxiv.org/pdf/1107.1834.pdf
    [2]: Leif Andersen. Efficient Simulation of the Heston Stochastic
      Volatility Models. 2006.
      Link:
      http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/d512ad5b22d73cc1c1257052003f1aed/1826b88b152e65a7c12574b000347c74/$FILE/LeifAndersenHeston.pdf
  """

  def __init__(self,
               kappa,
               theta,
               epsilon,
               rho,
               dtype=None,
               name=None):
    """Initializes the Heston Model.

    #### References:
      [1]: Leif Andersen. Efficient Simulation of the Heston Stochastic
        Volatility Models. 2006.
        Link:
        http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/d512ad5b22d73cc1c1257052003f1aed/1826b88b152e65a7c12574b000347c74/$FILE/LeifAndersenHeston.pdf
    Args:
      kappa: Scalar real `Tensor` or an instant of batch-free left-continuous
        `PiecewiseConstantFunc`. Should contain a positive value.
        Corresponds to the mean reversion rate.
      theta: Scalar real `Tensor` or an instant of batch-free left-continuous
        `PiecewiseConstantFunc`. Should contain positive a value of the same
        `dtype` as `kappa`.
        Corresponds to the lond run price variance.
      epsilon: Scalar real `Tensor` or an instant of batch-free left-continuous
        `PiecewiseConstantFunc`. Should contain positive a value of the same
        `dtype` as `kappa`.
        Corresponds to the volatility of the volatility.
      rho: Scalar real `Tensor` or an instant of batch-free left-continuous
        `PiecewiseConstantFunc`. Should contain a value in range (-1, 1) of the
        same `dtype` as `kappa`.
        Corresponds to the correlation between dW_{X}` and `dW_{V}`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name `heston_model`.
    """
    self._name = name or 'heston_model'
    with tf.compat.v1.name_scope(self._name,
                                 values=[kappa, theta, epsilon, rho]):
      self._dtype = dtype or None
      self._kappa = kappa if isinstance(
          kappa, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(
              kappa, dtype=self._dtype, name='kappa')
      self._theta = theta if isinstance(
          theta, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(
              theta, dtype=self._dtype, name='theta')
      self._epsilon = epsilon if isinstance(
          epsilon, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(
              epsilon, dtype=self._dtype, name='epsilon')
      self._rho = rho if isinstance(
          rho, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(
              rho, dtype=self._dtype, name='rho')

    def _vol_fn(t, x):
      """Volatility function of the Heston Process."""
      # For correlated brownian motions W_{X} and W_{V} with correlation
      # `rho(t)`, one can write
      # W_{V}(t) = rho(t) * W_{X}(t) + sqrt(1 - rho(t)**2) * W_{Z}(t)
      # where W_{Z}(t) is an independent from W_{X} and W{V} Brownian motion
      # Volatility matrix for Heston model is then
      # [[sqrt(V(t)), 0],
      #  [epsilon(t) * rho(t) * sqrt(V(t)), epsilon(t) * sqrt(1-rho**2) * V(t)]]
      vol = tf.sqrt(tf.abs(x[..., 1]))
      zeros = tf.zeros_like(vol)
      # Get parameter values at time `t`
      rho, epsilon = _get_parameters([t], self._rho, self._epsilon)  # pylint: disable=unbalanced-tuple-unpacking
      rho, epsilon = rho[0], epsilon[0]
      # First column of the volatility matrix
      vol_matrix_1 = tf.stack([vol, epsilon * rho * vol], -1)
      # Second column of the volatility matrix
      vol_matrix_2 = tf.stack([zeros, epsilon * tf.sqrt(1 - rho**2) * vol], -1)
      vol_matrix = tf.stack([vol_matrix_1, vol_matrix_2], -1)
      return vol_matrix

    def _drift_fn(t, x):
      var = x[..., 1]
      # Get parameter values at time `t`
      kappa, theta = _get_parameters([t], self._kappa, self._theta)  # pylint: disable=unbalanced-tuple-unpacking
      kappa, theta = kappa[0], theta[0]
      log_spot_drift = -var / 2
      var_drift = kappa * (theta - var)
      drift = tf.stack([log_spot_drift, var_drift], -1)
      return drift

    super(HestonModel, self).__init__(2, _drift_fn, _vol_fn, dtype, name)

  def sample_paths(self,
                   times,
                   initial_state,
                   num_samples=1,
                   random_type=None,
                   seed=None,
                   time_step=None,
                   skip=0,
                   tolerance=1e-6,
                   name=None):
    """Returns a sample of paths from the process.

    Using Quadratic-Exponential (QE) method described in [1] generates samples
    paths started at time zero and returns paths values at the specified time
    points.

    Args:
      times: Rank 1 `Tensor` of positive real values. The times at which the
        path points are to be evaluated.
      initial_state: A rank 1 `Tensor` with two elements where the first element
        corresponds to the initial value of the log spot `X(0)` and the second
        to the starting variance value `V(0)`.
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
      time_step: Positive Python float to denote time discretization parameter.
      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
        Halton sequence to skip. Used only when `random_type` is 'SOBOL',
        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      tolerance: Scalar positive real `Tensor`. Specifies minimum time tolerance
        for which the stochastic process `X(t) != X(t + tolerance)`.
      Default value: 1e-6.
      name: Str. The name to give this op.
        Default value: `sample_paths`.

    Returns:
      A `Tensor`s of shape [num_samples, k, 2] where `k` is the size
      of the `times`. For each sample and time the first dimension represents
      the simulated log-state trajectories of the spot price `X(t)`, whereas the
      second one represents the simulated variance trajectories `V(t)`.

    Raises:
      ValueError: If `time_step` is not supplied.

    #### References:
      [1]: Leif Andersen. Efficient Simulation of the Heston Stochastic
        Volatility Models. 2006.
    """
    if random_type is None:
      random_type = random.RandomType.PSEUDO
    if time_step is None:
      raise ValueError('`time_step` can not be `None` when calling '
                       'sample_paths of HestonModel.')
    # Note: all the notations below are the same as in [1].
    name = name or (self._name + '_sample_path')
    with tf.name_scope(name):
      time_step = tf.convert_to_tensor(time_step, self._dtype)
      times = tf.convert_to_tensor(times, self._dtype)
      current_log_spot = (
          tf.convert_to_tensor(initial_state[..., 0], dtype=self._dtype)
          + tf.zeros([num_samples], dtype=self._dtype))
      current_vol = (
          tf.convert_to_tensor(initial_state[..., 1], dtype=self._dtype)
          + tf.zeros([num_samples], dtype=self._dtype))
      num_requested_times = times.shape[0]
      times, keep_mask = _prepare_grid(
          times, time_step, times.dtype,
          self._kappa, self._theta, self._epsilon, self._rho)
      return self._sample_paths(
          times, num_requested_times,
          current_log_spot, current_vol,
          num_samples, random_type, keep_mask, seed, skip, tolerance)

  def _sample_paths(self,
                    times,
                    num_requested_times,
                    current_log_spot,
                    current_vol,
                    num_samples,
                    random_type,
                    keep_mask,
                    seed,
                    skip,
                    tolerance):
    """Returns a sample of paths from the process."""
    # Note: all the notations below are the same as in [1].
    dt = times[1:] - times[:-1]
    # Compute the parameters at `times`. Here + tf.reduce_min(dt) / 2 ensures
    # that the value is constant between `times`.
    kappa, theta, epsilon, rho = _get_parameters(  # pylint: disable=unbalanced-tuple-unpacking
        times + tf.reduce_min(dt) / 2,
        self._kappa, self._theta, self._epsilon, self._rho)
    # In order random_type which is not PSEUDO,  sequence of independent random
    # normals should be generated upfront.
    if dt.shape.is_fully_defined():
      steps_num = dt.shape.as_list()[-1]
    else:
      steps_num = tf.shape(dt)[-1]
      # TODO(b/148133811): Re-enable Sobol test when TF 2.2 is released.
      if random_type == random.RandomType.SOBOL:
        raise ValueError('Sobol sequence for Euler sampling is temporarily '
                         'unsupported when `time_step` or `times` have a '
                         'non-constant value')
    if random_type != random.RandomType.PSEUDO:
      # Note that at each iteration we need 2 random draws.
      normal_draws = utils.generate_mc_normal_draws(
          num_normal_draws=2, num_time_steps=steps_num,
          num_sample_paths=num_samples, random_type=random_type,
          seed=seed,
          dtype=self.dtype(), skip=skip)
    else:
      normal_draws = None
    cond_fn = lambda i, *args: i < steps_num
    def body_fn(i, written_count, current_vol, current_log_spot, vol_paths,
                log_spot_paths):
      """Simulate Heston process to the next time point."""
      time_step = dt[i]
      if normal_draws is None:
        normals = random.mv_normal_sample(
            (num_samples,),
            mean=tf.zeros([2], dtype=kappa.dtype), seed=seed)
      else:
        normals = normal_draws[i]
      def _next_vol_fn():
        return _update_variance(
            kappa[i], theta[i], epsilon[i], rho[i],
            current_vol, time_step, normals[..., 0])
      # Do not update variance if `time_step > tolerance`
      next_vol = tf.cond(time_step > tolerance,
                         _next_vol_fn,
                         lambda: current_vol)
      def _next_log_spot_fn():
        return _update_log_spot(
            kappa[i], theta[i], epsilon[i], rho[i],
            current_vol, next_vol, current_log_spot, time_step,
            normals[..., 1])
      # Do not update state if `time_step > tolerance`
      next_log_spot = tf.cond(time_step > tolerance,
                              _next_log_spot_fn,
                              lambda: current_log_spot)
      # Update volatility paths
      vol_paths = utils.maybe_update_along_axis(
          tensor=vol_paths,
          do_update=keep_mask[i + 1],
          ind=written_count,
          axis=1,
          new_tensor=tf.expand_dims(next_vol, axis=1))
      # Update log-spot paths
      log_spot_paths = utils.maybe_update_along_axis(
          tensor=log_spot_paths,
          do_update=keep_mask[i + 1],
          ind=written_count,
          axis=1,
          new_tensor=tf.expand_dims(next_log_spot, axis=1))
      written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
      return (i + 1, written_count,
              next_vol, next_log_spot, vol_paths, log_spot_paths)

    shape = (num_samples, num_requested_times)
    log_spot_paths = tf.zeros(shape, dtype=self._dtype)
    vol_paths = tf.zeros(shape, dtype=self._dtype)
    _, _, _, _, vol_paths, log_spot_paths = tf.while_loop(
        cond_fn, body_fn, (0, 0, current_vol, current_log_spot,
                           vol_paths, log_spot_paths),
        maximum_iterations=steps_num)
    return tf.stack([log_spot_paths, vol_paths], -1)


def _get_parameters(times, *params):
  """Gets parameter values at at specified `times`."""
  result = []
  for param in params:
    if isinstance(param, piecewise.PiecewiseConstantFunc):
      result.append(param(times))
    else:
      result.append(param * tf.ones_like(times))
  return result


def _update_variance(
    kappa, theta, epsilon, rho,
    current_vol, time_step, normals, psi_c=1.5):
  """Updates variance value."""
  del rho
  psi_c = tf.convert_to_tensor(psi_c, dtype=kappa.dtype)
  scaled_time = tf.exp(-kappa * time_step)
  epsilon_squared = epsilon**2
  m = theta + (current_vol - theta) * scaled_time
  s_squared = (
      current_vol * epsilon_squared * scaled_time / kappa
      * (1 - scaled_time) + theta * epsilon_squared / 2 / kappa
      * (1 - scaled_time)**2)
  psi = s_squared / m**2
  uniforms = 0.5 * (1 + tf.math.erf(normals / _SQRT_2))
  cond = psi < psi_c
  # Result where `cond` is true
  psi_inv = 2 / psi
  b_squared = psi_inv - 1 + tf.sqrt(psi_inv * (psi_inv - 1))

  a = m / (1 + b_squared)
  next_var_true = a * (tf.sqrt(b_squared) + tf.squeeze(normals))**2
  # Result where `cond` is false
  p = (psi - 1) / (psi + 1)
  beta = (1 - p) / m
  next_var_false = tf.where(uniforms > p,
                            tf.math.log(1 - p) - tf.math.log(1 - uniforms),
                            tf.zeros_like(uniforms)) / beta
  next_var = tf.where(cond, next_var_true, next_var_false)
  return next_var


def _update_log_spot(
    kappa, theta, epsilon, rho,
    current_vol, next_vol, current_log_spot, time_step, normals,
    gamma_1=0.5, gamma_2=0.5):
  """Updates log-spot value."""
  k_0 = - rho * kappa * theta / epsilon * time_step
  k_1 = (gamma_1 * time_step
         * (kappa * rho / epsilon - 0.5)
         - rho / epsilon)
  k_2 = (gamma_2 * time_step
         * (kappa * rho / epsilon - 0.5)
         + rho / epsilon)
  k_3 = gamma_1 * time_step * (1 - rho**2)
  k_4 = gamma_2 * time_step * (1 - rho**2)

  next_log_spot = (
      current_log_spot + k_0 + k_1 * current_vol + k_2 * next_vol
      + tf.sqrt(k_3 * current_vol + k_4 * next_vol) * normals)
  return next_log_spot


def _prepare_grid(times, time_step, dtype, *params):
  """Prepares grid of times for path generation.

  Args:
    times:  Rank 1 `Tensor` of increasing positive real values. The times at
      which the path points are to be evaluated.
    time_step: Rank 0 real `Tensor`. Maximal distance between points in
      resulting grid.
    dtype: `tf.Dtype` of the input and output `Tensor`s.
    *params: Parameters of the Heston model. Either scalar `Tensor`s of the
      same `dtype` or instances of `PiecewiseConstantFunc`.

  Returns:
    Tuple `(all_times, mask)`.
    `all_times` is a 1-D real `Tensor` containing all points from 'times`, the
    uniform grid of points between `[0, times[-1]]` with grid size equal to
    `time_step`, and jump locations of piecewise constant parameters The
    `Tensor` is sorted in ascending order and may contain duplicates.
    `mask` is a boolean 1-D `Tensor` of the same shape as 'all_times', showing
    which elements of 'all_times' correspond to THE values from `times`.
    Guarantees that times[0]=0 and mask[0]=False.
  """
  grid = tf.range(0.0, times[-1], time_step, dtype=dtype)
  additional_times = []
  for param in params:
    if isinstance(param, piecewise.PiecewiseConstantFunc):
      additional_times.append(param.jump_locations())
  all_times = tf.concat([grid, times] + additional_times, axis=0)
  additional_times_mask = [
      tf.zeros_like(times, dtype=tf.bool) for times in additional_times]
  mask = tf.concat([
      tf.zeros_like(grid, dtype=tf.bool),
      tf.ones_like(times, dtype=tf.bool)
  ] + additional_times_mask, axis=0)
  perm = tf.argsort(all_times, stable=True)
  all_times = tf.gather(all_times, perm)
  mask = tf.gather(mask, perm)
  return all_times, mask
