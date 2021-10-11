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
"""Cox–Ingersoll–Ross model."""

from typing import Optional

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import generic_ito_process


class CirModel(generic_ito_process.GenericItoProcess):
  """Cox–Ingersoll–Ross model.

  Represents the Ito process:

  ```None
    dX_i(t) = (a - k*X_i(t)) * dt +  sigma * sqrt(X_i(t)) * dW_i(t)
  ```
  where
    a / k: Corresponds to the long term mean.
    k: Corresponds to the speed of reversion.
    sigma: Corresponds to the instantaneous volatility.

  See [1] for details.

  #### References:
    [1]: A. Alfonsi. Affine Diffusions and Related Processes: Simulation,
      Theory and Applications
  """

  # TODO(b/196540888): Add batching support for CIR model
  def __init__(self,
               theta: types.RealTensor,
               mean_reversion: types.RealTensor,
               sigma: types.RealTensor,
               dtype: Optional[tf.DType] = None,
               name: Optional[str] = None):
    """Initializes the CIR Model.

    Args:
      theta: A positive scalar `Tensor`.
      mean_reversion: A positive scalar `Tensor` of the same dtype as `a`. Means
        speed of reversion.
      sigma: A scalar `Tensor` of the same dtype as `a`. Means volatility.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which maps to `tf.float32`.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name `cir_model`.
    """

    dim = 1
    dtype = dtype or tf.float32
    name = name or "cir_model"
    with tf.name_scope(name):
      self._theta = theta
      self._mean_reversion = mean_reversion
      self._sigma = sigma

      def _drift_fn(t, x):
        del t
        return self._theta - self._mean_reversion * x

      def _volatility_fn(t, x):
        del t
        return tf.expand_dims(self._sigma * tf.sqrt(x), axis=-1)

    super(CirModel, self).__init__(dim, _drift_fn, _volatility_fn, dtype, name)

  def sample_paths(self,
                   times: types.RealTensor,
                   initial_state: Optional[types.RealTensor] = None,
                   num_samples: int = 1,
                   random_type: Optional[random.RandomType] = None,
                   seed: Optional[int] = None,
                   name: Optional[str] = None) -> types.RealTensor:
    """Returns a sample of paths from the process.

    Using exact simulation method from [1].

    Args:
      times: Rank 1 `Tensor` of positive real values. The times at which the
        path points are to be evaluated.
      initial_state: A `Tensor` of the same `dtype` as `times` and of shape
        broadcastable with `[num_samples, dim]`. Represents the initial state of
        the Ito process.
        Default value: `None` which maps to a initial state of ones.
      num_samples: Positive scalar `int`. The number of paths to draw.
      random_type: `STATELESS` or `PSEUDO` type from `RandomType` Enum. The type
        of (quasi)-random number generator to use to generate the paths.
      seed: The seed for the random number generator.
        For `PSEUDO` random type: it is an Integer.
        For `STATELESS` random type: it is an integer `Tensor` of shape `[2]`.
          In this case the algorithm samples random numbers with seeds `[seed[0]
          + i, seed[1] + j], i in {0, 1}, j in {0, 1, ..., num_times}`, where
          `num_times` is the size of `times`.
        Default value: `None` which means no seed is set, but it works only with
          `PSEUDO` random type. For `STATELESS` it has to be provided.
      name: Str. The name to give this op.
        Default value: `sample_paths`.

    Returns:
      A `Tensor`s of shape [num_samples, num_times, dim] where `num_times` is
      the size of the `times`.

    Raises:
      ValueError: If `random_type` or `seed` is not supported.

    #### References:
    [1]: A. Alfonsi. Affine Diffusions and Related Processes: Simulation,
      Theory and Applications
    """
    name = name or (self._name + "_sample_path")
    with tf.name_scope(name):
      if initial_state is None:
        initial_state = tf.ones([num_samples, self._dim],
                                dtype=self._dtype,
                                name="initial_state")
      else:
        initial_state = (
            tf.convert_to_tensor(
                initial_state, dtype=self._dtype, name="initial_state") +
            tf.zeros([num_samples, self._dim], dtype=self._dtype))
      times = tf.convert_to_tensor(times, dtype=self._dtype, name="times")
      num_requested_times = tff_utils.get_shape(times)[0]
      if random_type is None:
        random_type = random.RandomType.PSEUDO
      if random_type == random.RandomType.STATELESS and seed is None:
        raise ValueError(
            "`seed` equal to None is not supported with STATELESS random type.")

      return self._sample_paths(
          times=times,
          num_requested_times=num_requested_times,
          initial_state=initial_state,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
      )

  def _sample_paths(
      self,
      times,
      num_requested_times,
      initial_state,
      num_samples,
      random_type,
      seed,
  ):
    """Returns a sample of paths from the process."""
    times = tf.concat([[0], times], -1)
    # Time increments
    # Shape [num_requested_times, 1, 1]
    dts = tf.expand_dims(
        tf.expand_dims(times[1:] - times[:-1], axis=-1), axis=-1)
    (poisson_fn, gamma_fn, poisson_seed_fn,
     gamma_seed_fn) = _get_distributions(random_type)
    distribution_shape = tff_utils.get_shape(initial_state)

    def _sample_at_time(i, update_idx, current_x, samples):
      dt = dts[i]
      zeta = tf.where(self._mean_reversion != 0,
                      (1 - tf.math.exp(-self._mean_reversion * dt)) /
                      self._mean_reversion, dt)
      c = tf.math.divide_no_nan(
          tf.constant(4, dtype=self.dtype()), self._sigma**2 * zeta)
      d = c * tf.math.exp(-self._mean_reversion * dt)

      poisson_rv = poisson_fn(
          shape=distribution_shape,
          lam=d * current_x / 2,
          seed=poisson_seed_fn(seed, i),
          dtype=self._dtype)

      gamma_param_alpha = poisson_rv + 2 * self._theta / (self._sigma**2)
      gamma_param_beta = c / 2

      new_x = gamma_fn(
          shape=distribution_shape,
          alpha=gamma_param_alpha,
          beta=gamma_param_beta,
          seed=gamma_seed_fn(seed, i),
          dtype=self._dtype)
      # `gamma_fn` outputs infinity when `c==0`
      new_x = tf.where(c > 0, new_x, current_x)

      samples = samples.write(i, new_x)
      return (i + 1, update_idx, new_x, samples)

    cond_fn = lambda i, *args: i < num_requested_times
    samples = tf.TensorArray(
        dtype=self._dtype,
        size=num_requested_times,
        element_shape=[num_samples, self._dim],
        clear_after_read=False)
    _, _, _, samples = tf.while_loop(
        cond_fn,
        _sample_at_time, (0, 0, initial_state, samples),
        maximum_iterations=num_requested_times)
    # Shape [num_requested_times, num_samples, dim]
    samples = samples.stack()
    # Shape [num_samples, num_requested_times, dim]
    samples = tf.transpose(samples, perm=[1, 0, 2])
    return samples


def _get_distributions(random_type):
  """Returns the distribution and its parameters depending on the `random_type`."""
  if random_type == random.RandomType.STATELESS:
    poisson_seed_fn = lambda seed, i: tf.stack([seed[0], seed[1] + i])
    gamma_seed_fn = lambda seed, i: tf.stack([seed[0] + 1, seed[1] + i])
    return (tf.random.stateless_poisson, tf.random.stateless_gamma,
            poisson_seed_fn, gamma_seed_fn)
  elif random_type == random.RandomType.PSEUDO:
    seed_fn = lambda seed, _: seed
    return tf.random.poisson, tf.random.gamma, seed_fn, seed_fn
  else:
    raise ValueError("Only STATELESS and PSEUDO random types are supported.")
