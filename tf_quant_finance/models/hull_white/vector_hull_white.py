# Lint as: python3
# Copyright 2020 Google LLC
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

"""Vector of correlated Hull-White models with time-dependent parameters."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math import gradient
from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import generic_ito_process
from tf_quant_finance.models import utils


class VectorHullWhiteModel(generic_ito_process.GenericItoProcess):
  r"""Ensemble of correlated Hull-White Models.

  Represents the Ito process:

  ```None
    dr_i(t) = (theta_i(t) - a_i(t) * r_i(t)) dt + sigma_i(t) * dW_{r_i}(t),
    1 <= i <= n,
  ```
  where `W_{r_i}` are 1D Brownian motions with a correlation matrix `Rho(t)`.
  For each `i`, `r_i` is the Hull-White process.
  `theta_i`, `a_i`, `sigma_i`, `Rho` are positive functions of time.
  `a_i` correspond to the mean-reversion rate, `sigma_i` is the volatility of
  the process, `theta_i(t)` is the function that determines long run behaviour
  of the process `r(t) = (r_1(t), ..., r_n(t))`
  and is defined to match the market data through the instantaneous forward
  rate matching:

  ```None
  \theta_i = df_i(t) / dt + a_i * f_i(t) + 0.5 * sigma_i**2 / a_i
             * (1 - exp(-2 * a_i *t)), 1 <= i <= n
  ```
  where `f_i(t)` is the instantaneous forward rate at time `0` for a maturity
  `t` and `df_i(t)/dt` is the gradient of `f_i` with respect to the maturity.
  See Section 3.3.1 of [1] for details.

  If the parameters `a_i`, `sigma_i` and `Rho` are piecewise constant functions,
  the process is sampled exactly. Otherwise, Euler sampling is be used.

  For `n=1` this class represents Hull-White Model (see
  tff.models.hull_white.HullWhiteModel1F).

  #### Example. Two correlated Hull-White processes.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  # Mean-reversion is a piecewise constant function with different jumps for
  # the two processes. `mean_reversion(t)` has shape `[dim] + t.shape`.
  mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
      jump_locations=[[1, 2, 3, 4], [1, 2, 3, 3]],
      values=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.2, 0.3, 0.4, 0.4]],
      dtype=dtype)
  # Volatility is a piecewise constant function with jumps at the same locations
  # for both Hull-White processes. `volatility(t)` has shape `[dim] + t.shape`.
  volatility = tff.math.piecewise.PiecewiseConstantFunc(
      jump_locations=[0.1, 2.],
      values=[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
      dtype=dtype)
  # Correlation matrix is constant
  corr_matrix = [[1., 0.1], [0.1, 1.]]
  instant_forward_rate_fn = lambda *args: [0.01, 0.02]
  process = tff.models.hull_white.VectorHullWhiteModel(
      dim=2, mean_reversion=mean_reversion,
      volatility=volatility, instant_forward_rate_fn=instant_forward_rate_fn,
      corr_matrix=None,
      dtype=dtype)
  # Sample 10000 paths using Sobol numbers as a random type.
  times = np.linspace(0., 1.0, 10)
  num_samples = 10000  # number of trajectories
  paths = process.sample_paths(
      times,
      num_samples=num_samples,
      initial_state=[0.1, 0.2],
      random_type=tff.math.random.RandomType.SOBOL)
  # Compute mean for each Hull-White process at the terminal value
  tf.math.reduce_mean(paths[:, -1, :], axis=0)
  # Expected value: [0.09594861, 0.14156537]
  # Check that the correlation is recovered
  np.corrcoef(paths[:, -1, 0], paths[:, -1, 1])
  # Expected value: [[1.       , 0.0914114],
  #                  [0.0914114, 1.       ]]
  ```

  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models. 2007.
  """

  def __init__(self,
               dim,
               mean_reversion,
               volatility,
               instant_forward_rate_fn,
               corr_matrix=None,
               dtype=None,
               name=None):
    """Initializes the Correlated Hull-White Model.

    Args:
      dim: A Python scalar which corresponds to the number of correlated
        Hull-White Models.
      mean_reversion: A real positive `Tensor` of shape `[dim]` or a Python
        callable. The callable can be one of the following:
          (a) A left-continuous piecewise constant object (e.g.,
          `tff.math.piecewise.PiecewiseConstantFunc`) that has a property
          `is_piecewise_constant` set to `True`. In this case the object
          should have a method `jump_locations(self)` that returns a
          `Tensor` of shape `[dim, num_jumps]` or `[num_jumps]`
          In the first case, `mean_reversion(t)` should return a `Tensor`
          of shape `[dim] + t.shape`, and in the second, `t.shape + [dim]`,
          where `t` is a rank 1 `Tensor` of the same `dtype` as the output.
          See example in the class docstring.
         (b) A callable that accepts scalars (stands for time `t`) and returns a
         `Tensor` of shape `[dim]`.
        Corresponds to the mean reversion rate.
      volatility: A real positive `Tensor` of the same `dtype` as
        `mean_reversion` or a callable with the same specs as above.
        Corresponds to the lond run price variance.
      instant_forward_rate_fn: A Python callable that accepts expiry time as a
        scalar `Tensor` of the same `dtype` as `mean_reversion` and returns a
        `Tensor` of shape `[dim]`.
        Corresponds to the instanteneous forward rate at the present time for
        the input expiry time.
      corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as
        `mean_reversion` or a Python callable. The callable can be one of
        the following:
          (a) A left-continuous piecewise constant object (e.g.,
          `tff.math.piecewise.PiecewiseConstantFunc`) that has a property
          `is_piecewise_constant` set to `True`. In this case the object
          should have a method `jump_locations(self)` that returns a
          `Tensor` of shape `[num_jumps]`. `corr_matrix(t)` should return a
          `Tensor` of shape `t.shape + [dim]`, where `t` is a rank 1 `Tensor`
          of the same `dtype` as the output.
         (b) A callable that accepts scalars (stands for time `t`) and returns a
         `Tensor` of shape `[dim, dim]`.
        Corresponds to the correlation matrix `Rho`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name `hull_white_model`.

    Raises:
      ValueError:
        (a) If either `mean_reversion`, `volatility`, or `corr_matrix` is
          a piecewise constant function where `jump_locations` have batch shape
          of rank > 1.
        (b): If batch rank of the `jump_locations` is `[n]` with `n` different
          from `dim`.
    """
    self._name = name or 'hull_white_model'
    with tf.name_scope(self._name):
      self._dtype = dtype or None
      # If the parameter is callable but not a piecewise constant use
      # generic sampling method (e.g., Euler).
      self._sample_with_generic = False
      def _instant_forward_rate_fn(t):
        # Add dependency on `t` in order to make `instant_forward_rate_fn`
        # differentiable w.r.t. `t`.
        return (tf.convert_to_tensor(instant_forward_rate_fn(t), dtype=dtype)
                + 0 * t)
      self._instant_forward_rate_fn = _instant_forward_rate_fn
      self._mean_reversion, sample_with_generic = _input_type(
          mean_reversion, dim=dim, dtype=dtype, name='mean_reversion')
      # Update flag to whether to sample with a generic sampler.
      self._sample_with_generic |= sample_with_generic
      # Get the volatility type
      self._volatility, sample_with_generic = _input_type(
          volatility, dim=dim, dtype=dtype, name='volatility')
      # Update flag to whether to sample with a generic sampler.
      self._sample_with_generic |= sample_with_generic
      if corr_matrix is not None:
        # Get correlation matrix type
        self._corr_matrix, sample_with_generic = _input_type(
            corr_matrix, dim=dim, dtype=dtype, name='corr_matrix')
        # Update flag to whether to sample with a generic sampler.
        self._sample_with_generic |= sample_with_generic
      else:
        self._corr_matrix = None

    # Volatility function
    def _vol_fn(t, x):
      """Volatility function of correlated Hull-White."""
      # Get parameter values at time `t`
      volatility = _get_parameters(tf.expand_dims(t, -1), self._volatility)[0]
      volatility = tf.transpose(volatility)
      if self._corr_matrix is not None:
        corr_matrix = _get_parameters(tf.expand_dims(t, -1), self._corr_matrix)
        corr_matrix = corr_matrix[0]
        corr_matrix = tf.linalg.cholesky(corr_matrix)
      else:
        corr_matrix = tf.eye(self._dim, dtype=volatility.dtype)

      return volatility * corr_matrix + tf.zeros(
          x.shape.as_list()[:-1] + [self._dim, self._dim],
          dtype=volatility.dtype)

    # Drift function
    def _drift_fn(t, x):
      """Drift function of correlated Hull-White."""
      # Get parameter values at time `t`
      mean_reversion, volatility = _get_parameters(  # pylint: disable=unbalanced-tuple-unpacking
          tf.expand_dims(t, -1), self._mean_reversion, self._volatility)
      fwd_rates = self._instant_forward_rate_fn(t)
      fwd_rates_grad = gradient.fwd_gradient(
          self._instant_forward_rate_fn, t)
      drift = fwd_rates_grad + mean_reversion * fwd_rates
      drift += (volatility**2 / 2 / mean_reversion
                * (1 - tf.math.exp(-2 * mean_reversion * t))
                - mean_reversion * x)
      return drift
    super(VectorHullWhiteModel, self).__init__(dim, _drift_fn, _vol_fn,
                                               dtype, name)

  def sample_paths(self,
                   times,
                   initial_state,
                   num_samples=1,
                   random_type=None,
                   seed=None,
                   skip=0,
                   time_step=None,
                   name=None):
    """Returns a sample of paths from the correlated Hull-White process.

    Uses exact sampling if `self.mean_reversion`, `self.volatility` and
    `self.corr_matrix` are all `Tensor`s or piecewise constant functions, and
    Euler scheme sampling if one of the arguments is a generic callable.

    Args:
      times: Rank 1 `Tensor` of positive real values. The times at which the
        path points are to be evaluated.
      initial_state: A `Tensor` of the same `dtype` as `times` and shape
        broadcastable with `[num_samples, self._dim]`
      num_samples: Positive scalar `int32` `Tensor`. The number of paths to
        draw.
      random_type: Enum value of `RandomType`. The type of (quasi)-random
        number generator to use to generate the paths.
        Default value: `None` which maps to the standard pseudo-random numbers.
      seed: Python `int`. The random seed to use.
        Default value: None, i.e., no seed is set.
      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
        Halton sequence to skip. Used only when `random_type` is 'SOBOL',
        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
        Default value: `0`.
      time_step: Scalar real `Tensor`. Maximal distance between time grid points
        in Euler scheme. Used only when Euler scheme is applied.
      Default value: `None`.
      name: Python string. The name to give this op.
        Default value: `sample_paths`.

    Returns:
      A `Tensor` of shape [num_samples, k, dim] where `k` is the size
      of the `times` and `dim` is the dimension of the process.

    Raises:
      ValueError:
        (a) If `times` has rank different from `1`.
        (b) If Euler scheme is used by times is not supplied.
    """
    # Note: all the notations below are the same as in [1].
    name = name or self._name + '_sample_path'
    with tf.name_scope(name):
      times = tf.convert_to_tensor(times, self._dtype)
      if len(times.shape) != 1:
        raise ValueError('`times` should be a rank 1 Tensor. '
                         'Rank is {} instead.'.format(len(times.shape)))
      if self._sample_with_generic:
        if time_step is None:
          raise ValueError('`time_step` can not be `None` when at least one of '
                           'the parameters is a generic callable.')
        return euler_sampling.sample(dim=self._dim,
                                     drift_fn=self._drift_fn,
                                     volatility_fn=self._volatility_fn,
                                     times=times,
                                     time_step=time_step,
                                     num_samples=num_samples,
                                     initial_state=initial_state,
                                     random_type=random_type,
                                     seed=seed,
                                     skip=skip,
                                     dtype=self._dtype)
      current_rates = tf.broadcast_to(
          tf.convert_to_tensor(initial_state, dtype=self._dtype),
          [num_samples, self._dim])
      current_instant_forward_rates = self._instant_forward_rate_fn(
          tf.constant(0, self._dtype))
      num_requested_times = times.shape[0]
      params = [self._mean_reversion, self._volatility, self._corr_matrix]
      if self._corr_matrix is not None:
        params = params + [self._corr_matrix]
      times, keep_mask = _prepare_grid(
          times, params)
      return self._sample_paths(
          times, num_requested_times,
          current_rates, current_instant_forward_rates,
          num_samples, random_type, skip, keep_mask, seed)

  def _sample_paths(self,
                    times,
                    num_requested_times,
                    current_rates,
                    current_instant_forward_rates,
                    num_samples,
                    random_type,
                    skip,
                    keep_mask,
                    seed):
    """Returns a sample of paths from the process."""
    # Note: all the notations below are the same as in [1].
    # Add zeros as a starting location
    dt = times[1:] - times[:-1]
    if dt.shape.is_fully_defined():
      steps_num = dt.shape.as_list()[-1]
    else:
      steps_num = tf.shape(dt)[-1]
      # TODO(b/148133811): Re-enable Sobol test when TF 2.2 is released.
      if random_type == random.RandomType.SOBOL:
        raise ValueError('Sobol sequence for Euler sampling is temporarily '
                         'unsupported when `time_step` or `times` have a '
                         'non-constant value')
    # In order to use low-discrepancy random_type we need to generate the
    # sequence of independent random normals upfront.
    if random_type in (random.RandomType.SOBOL,
                       random.RandomType.HALTON,
                       random.RandomType.HALTON_RANDOMIZED):
      normal_draws = utils.generate_mc_normal_draws(
          num_normal_draws=self._dim, num_time_steps=steps_num,
          num_sample_paths=num_samples, random_type=random_type,
          seed=seed,
          dtype=self._dtype, skip=skip)
    else:
      normal_draws = None
    mean_reversion, volatility = _get_parameters(  # pylint: disable=unbalanced-tuple-unpacking
        times + tf.math.reduce_min(dt) / 2,
        self._mean_reversion, self._volatility)
    if self._corr_matrix is not None:
      corr_matrix = _get_parameters(
          times + tf.math.reduce_min(dt) / 2, self._corr_matrix)[0]
      corr_matrix_root = tf.linalg.cholesky(corr_matrix)
    else:
      corr_matrix_root = None
    cond_fn = lambda i, *args: i < tf.size(dt)
    def body_fn(i, written_count,
                current_rates,
                current_instant_forward_rates,
                rate_paths):
      """Simulate Heston process to the next time point."""
      current_time = times[i]
      next_time = times[i + 1]
      if normal_draws is None:
        normals = random.mv_normal_sample(
            (num_samples,),
            mean=tf.zeros((self._dim,), dtype=mean_reversion.dtype),
            random_type=random_type, seed=seed)
      else:
        normals = normal_draws[i]
      next_rates, next_instant_forward_rates = _sample_at_next_time(
          i, next_time, current_time,
          mean_reversion[i], volatility[i],
          self._instant_forward_rate_fn,
          current_instant_forward_rates,
          current_rates, corr_matrix_root, normals)
      # Update `rate_paths`
      rate_paths = utils.maybe_update_along_axis(
          tensor=rate_paths,
          do_update=keep_mask[i + 1],
          ind=written_count,
          axis=1,
          new_tensor=tf.expand_dims(next_rates, axis=1))
      written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
      return (i + 1, written_count,
              next_rates,
              next_instant_forward_rates,
              rate_paths)

    rate_paths = tf.zeros((num_samples, num_requested_times, self._dim),
                          dtype=self._dtype)
    _, _, _, _, rate_paths = tf.while_loop(
        cond_fn, body_fn, (0, 0, current_rates,
                           current_instant_forward_rates,
                           rate_paths))
    return rate_paths


def _sample_at_next_time(
    i, next_time, current_time,
    mean_reversion, volatility, instant_forward_rate_fn,
    current_instant_forward_rates,
    current_rates, corr_matrix_root, normals):
  """Generates samples at at `next_time` conditionally on state."""
  if corr_matrix_root is not None:
    normals = tf.linalg.matvec(corr_matrix_root[i], normals)
  next_instant_forward_rates = instant_forward_rate_fn(next_time)
  # Corresponds to alpha(t) from [1]
  alpha_1 = next_instant_forward_rates
  alpha_1 += volatility**2 / mean_reversion**2 / 2 * (
      1 - tf.math.exp(-mean_reversion * next_time))**2
  # Corresponds to alpha(s) from [1]
  alpha_2 = current_instant_forward_rates
  alpha_2 += volatility**2 / mean_reversion**2 / 2 * (
      1 - tf.math.exp(-mean_reversion * current_time))**2
  # Stochastic vol term
  vol = volatility * tf.sqrt(
      0.5 * (1 - tf.exp(2 * mean_reversion * (current_time - next_time)))
      / mean_reversion) * normals
  factor_1 = tf.math.exp(-mean_reversion * (next_time - current_time))
  next_rates = current_rates * factor_1 + alpha_1 - alpha_2 * factor_1 + vol
  return next_rates, next_instant_forward_rates


def _get_parameters(times, *params):
  """Gets parameter values at at specified `times`."""
  res = []
  for param in params:
    if isinstance(param, piecewise.PiecewiseConstantFunc):
      jump_locations = param.jump_locations()
      if len(jump_locations.shape) > 1:
        # If `jump_locations` has batch dimension, transpose the result
        # Shape [num_times, dim]
        res.append(tf.transpose(param(times)))
      else:
        # Shape [num_times, dim]
        res.append(param(times))
    elif callable(param):
      # Used only in drift and colatility computation.
      # Here `times` is of shape [1]
      t = tf.squeeze(times)
      # The result has to have shape [1] + param.shape
      res.append(tf.expand_dims(param(t), 0))
    else:
      res.append(param + tf.zeros(times.shape + param.shape, dtype=times.dtype))
  return res


def _prepare_grid(times, *params):
  """Prepares grid of times for path generation.

  Args:
    times:  Rank 1 `Tensor` of increasing positive real values. The times at
      which the path points are to be evaluated.
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
  additional_times = []
  for param in params:
    if hasattr(param, 'is_piecewise_constant'):
      if param.is_piecewise_constant:
        # Flatten all jump locations
        additional_times.append(tf.reshape(param.jump_locations(), [-1]))
  zeros = tf.constant([0], dtype=times.dtype)
  all_times = tf.concat([zeros] + [times] + additional_times, axis=0)
  additional_times_mask = [
      tf.zeros_like(times, dtype=tf.bool) for times in additional_times]
  mask = tf.concat([
      tf.cast(zeros, dtype=tf.bool),
      tf.ones_like(times, dtype=tf.bool)
  ] + additional_times_mask, axis=0)
  perm = tf.argsort(all_times, stable=True)
  all_times = tf.gather(all_times, perm)
  mask = tf.gather(mask, perm)
  return all_times, mask


def _input_type(param, dim, dtype, name):
  """Checks if the input parameter is a callable or piecewise constant."""
  # If the parameter is callable but not a piecewise constant use
  # generic sampling method (e.g., Euler).
  sample_with_generic = False
  if hasattr(param, 'is_piecewise_constant'):
    if param.is_piecewise_constant:
      jumps_shape = param.jump_locations().shape
      if len(jumps_shape) > 2:
        raise ValueError(
            'Batch rank of `jump_locations` should be `1` for all piecewise '
            'constant arguments but {} instead'.format(len(jumps_shape[:-1])))
      if len(jumps_shape) == 2:
        if dim != jumps_shape[0]:
          raise ValueError(
              'Batch shape of `jump_locations` should be either empty or '
              '`[{0}]` but `[{1}]` instead'.format(dim, jumps_shape[0]))
      return param, sample_with_generic
    else:
      sample_with_generic = True
  elif callable(param):
    sample_with_generic = True
  else:
    # Otherwise, input is a `Tensor`
    param = tf.convert_to_tensor(param, dtype=dtype, name=name)
  return param, sample_with_generic
