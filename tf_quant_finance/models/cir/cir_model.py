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

  def __init__(self,
               theta: types.RealTensor,
               mean_reversion: types.RealTensor,
               sigma: types.RealTensor,
               dtype: Optional[tf.DType] = None,
               name: Optional[str] = None):
    """Initializes the CIR Model.

    Args:
      theta: A positive scalar `Tensor` with shape `batch_shape` + [1].
      mean_reversion: A positive scalar `Tensor` of the same dtype and shape as
        `theta`. Means speed of reversion.
      sigma: A scalar `Tensor` of the same dtype and shape as `theta`.Means
        volatility.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which maps to `tf.float32`.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name `cir_model`.
    """

    dim = 1
    dtype = dtype or tf.float32
    name = name or "cir_model"
    with tf.name_scope(name):

      def _convert_param_to_tensor(param):
        """Converts `param` to `Tesnor`.

        Args:
          param: `Scalar` or `Tensor` with shape `batch_shape` + [1].

        Returns:
          `param` if it `Tensor`, if it is `Scalar` convert it to `Tensor` with
          [1] shape.
        """
        param_t = tf.convert_to_tensor(param, dtype=dtype)
        return param_t * tf.ones(shape=dim, dtype=dtype)

      def _get_batch_shape(param):
        """`param` must has shape `batch_shape + [1]`."""
        param_shape = tff_utils.get_shape(param)
        # Last rank is `1`
        return param_shape[:-1]

      # Converts params to `Tensor` with shape `batch_shape + [1]`
      self._theta = _convert_param_to_tensor(theta)
      self._mean_reversion = _convert_param_to_tensor(mean_reversion)
      self._sigma = _convert_param_to_tensor(sigma)

      self._batch_shape = _get_batch_shape(self._theta)
      self._batch_shape_rank = len(self._batch_shape)

      def _drift_fn(t, x):
        del t

        expand_rank = tff_utils.get_shape(x).rank - self._batch_shape_rank - 1
        # `axis` is -2, because the new dimension needs to be added before `1`
        theta_expand = self._expand_param_on_rank(
            self._theta, expand_rank, axis=-2)
        mean_reversion_expand = self._expand_param_on_rank(
            self._mean_reversion, expand_rank, axis=-2)
        return theta_expand - mean_reversion_expand * x

      def _volatility_fn(t, x):
        del t

        expand_rank = len(tff_utils.get_shape(x)) - self._batch_shape_rank - 1
        # `axis` is -2, because the new dimension needs to be added before `1`
        sigma_expand = self._expand_param_on_rank(
            self._sigma, expand_rank, axis=-2)
        return tf.expand_dims(sigma_expand * tf.sqrt(x), axis=-1)

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
        broadcastable with `batch_shape + [num_samples, 1]`. Represents the
        initial state of the Ito process. `batch_shape` is the shape of the
        independent stochastic processes being modelled and is inferred from the
        initial state `x0`.
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
      A `Tensor`s of shape batch_shape + [num_samples, num_times, 1] where
      `num_times` is
      the size of the `times`.

    Raises:
      ValueError: If `random_type` or `seed` is not supported.

    ## Example

    ```python
    import tensorflow as tf
    import tf_quant_finance as tff

    # In this example `batch_shape` is 2, so parameters has shape [2, 1]
    process = tff.models.CirModel(
        theta=[[0.02], [0.03]],
        mean_reversion=[[0.5], [0.4]],
        sigma=[[0.1], [0.5]],
        dtype=tf.float64)

    num_samples = 5
    # `initial_state` has shape [num_samples, 1]
    initial_state=[[0.1], [0.2], [0.3], [0.4], [0.5]]
    times = [0.1, 0.2, 1.0]
    samples = process.sample_paths(
        times=times,
        num_samples=num_samples,
        initial_state=initial_state)
    # `samples` has shape [2, 5, 3, 1]
    ```

    #### References:
    [1]: A. Alfonsi. Affine Diffusions and Related Processes: Simulation,
      Theory and Applications
    """
    name = name or (self._name + "_sample_path")
    with tf.name_scope(name):
      element_shape = self._batch_shape + [num_samples, self._dim]

      # batch_shape + [1] -> batch_shape + [1 (for num_samples), 1]
      theta = self._expand_param_on_rank(self._theta, 1, axis=-2)
      mean_reversion = self._expand_param_on_rank(
          self._mean_reversion, 1, axis=-2)
      sigma = self._expand_param_on_rank(self._sigma, 1, axis=-2)

      if initial_state is None:
        initial_state = tf.ones(
            element_shape, dtype=self._dtype, name="initial_state")
      else:
        initial_state = (
            tf.convert_to_tensor(
                initial_state, dtype=self._dtype, name="initial_state") +
            tf.zeros(element_shape, dtype=self._dtype))

      times = tf.convert_to_tensor(times, dtype=self._dtype, name="times")
      num_requested_times = tff_utils.get_shape(times)[0]
      if random_type is None:
        random_type = random.RandomType.PSEUDO
      if random_type == random.RandomType.STATELESS and seed is None:
        raise ValueError(
            "`seed` equal to None is not supported with STATELESS random type.")

      return self._sample_paths(
          theta=theta,
          mean_reversion=mean_reversion,
          sigma=sigma,
          element_shape=element_shape,
          times=times,
          num_requested_times=num_requested_times,
          initial_state=initial_state,
          num_samples=num_samples,
          random_type=random_type,
          seed=seed,
      )

  def _sample_paths(
      self,
      theta,
      mean_reversion,
      sigma,
      element_shape,
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
     gamma_seed_fn) = self._get_distributions(random_type)

    def _sample_at_time(i, update_idx, current_x, samples):
      dt = dts[i]
      # Shape batch_shape + [num_samples, dim]
      zeta = tf.where(
          tf.math.equal(mean_reversion, tf.zeros_like(mean_reversion)), dt,
          (1 - tf.math.exp(-mean_reversion * dt)) / mean_reversion)
      c = tf.math.divide_no_nan(
          tf.constant(4, dtype=self._dtype), sigma**2 * zeta)
      d = c * tf.math.exp(-mean_reversion * dt)

      poisson_rv = poisson_fn(
          shape=element_shape,
          lam=d * current_x / 2,
          seed=poisson_seed_fn(seed, i),
          dtype=self._dtype)

      gamma_param_alpha = poisson_rv + 2 * theta / (sigma**2)
      gamma_param_beta = c / 2

      new_x = gamma_fn(
          shape=element_shape,
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
        element_shape=element_shape,
        clear_after_read=False)
    _, _, _, samples = tf.while_loop(
        cond_fn,
        _sample_at_time, (0, 0, initial_state, samples),
        maximum_iterations=num_requested_times)

    # Shape [num_requested_times, batch_shape..., num_samples, 1]
    samples = samples.stack()
    samples_rank = len(tff_utils.get_shape(samples))
    perm = [batch_idx for batch_idx in range(1, samples_rank - 2)
           ] + [samples_rank - 2, 0, samples_rank - 1]
    # Shape batch_shape + [num_samples, num_requested_times, 1]
    return tf.transpose(samples, perm=perm)

  def _expand_param_on_rank(self, param, expand_rank, axis):
    """Adds dimensions to `param`, not inplace.

    Args:
      param: initial element.
      expand_rank: is amount of dimensions that need to be added.
      axis: is axis where to place these dimensions.

    Returns:
      New `Tensor`.
    """
    param_tensor = tf.convert_to_tensor(param, dtype=self._dtype)
    param_expand = param_tensor
    for _ in range(expand_rank):
      param_expand = tf.expand_dims(param_expand, axis)
    return param_expand

  @staticmethod
  def _get_distributions(random_type):
    """Returns the distribution depending on the `random_type`.

    Args:
      random_type: `STATELESS` or `PSEUDO` type from `RandomType` Enum.

    Returns:
     Tuple (Poisson distribution, Gamma distribution, function to generate
     seed for Poisson distribution, function to generate seed for Gamma
     distribution).
    """
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
