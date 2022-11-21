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

"""Uniform distribution with various random types."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.random_ops import halton
from tf_quant_finance.math.random_ops import sobol
from tf_quant_finance.math.random_ops.multivariate_normal import RandomType


def uniform(
    dim,
    sample_shape,
    random_type=None,
    dtype=None,
    seed=None,
    name=None,
    **kwargs):
  """Generates draws from a uniform distribution on [0, 1).

  Allows generating either (pseudo) random or quasi-random draws based on the
  `random_type` parameter. Dimension parameter `dim` is required since for
  quasi-random draws one needs to know the dimensionality of the space as
  opposed to just sample shape.

  #### Example:

  ```python
  sample_shape = [10]  # Generates 10 draws.

  # `Tensor` of shape [10, 1]
  uniform_samples = uniform(1, sample_shape)

  # `Tensor` of shape [10, 5]
  sobol_samples = uniform(5, sample_shape, RandomType.SOBOL)
  ```

  Args:
    dim: A positive Python `int` representing each sample's `event_size.`
    sample_shape: Rank 1 `Tensor` of positive `int32`s. Should specify a valid
      shape for a `Tensor`. The shape of the samples to be drawn.
    random_type: Enum value of `RandomType`. The type of draw to generate.
      Default value: None which is mapped to `RandomType.PSEUDO`.
    dtype: Optional `dtype` (eithier `tf.float32` or `tf.float64`). The dtype of
      the output `Tensor`.
      Default value: `None` which maps to `tf.float32`.
    seed: Seed for the random number generator. The seed is
      only relevant if `random_type` is one of
      `[STATELESS, PSEUDO, HALTON_RANDOMIZED]`. For `PSEUDO`, and
      `HALTON_RANDOMIZED` the seed should be a Python integer. For
      `STATELESS` must be supplied as an integer `Tensor` of shape `[2]`.
      Default value: `None` which means no seed is set.
    name: Python `str` name prefixed to ops created by this class.
      Default value: `None` which is mapped to the default name
        `uniform_distribution`.
    **kwargs: parameters, specific to a random type:
      (1) `skip` is an `int` 0-d `Tensor`. The number of initial points of the
      Sobol or Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      (2) `randomization_params` is an instance of
      `tff.math.random.HaltonParams` that fully describes the randomization
      behavior. Used only when `random_type` is 'HALTON_RANDOMIZED', otherwise
      ignored (see halton.sample args for more details). If this parameter is
      provided when random_type is `HALTON_RANDOMIZED`, the `seed` parameter is
      ignored.
      Default value: `None`. In this case with randomized = True, the necessary
        randomization parameters will be computed from scratch.

  Returns:
    samples: A `Tensor` of shape `sample_shape + [dim]`. The draws
      from the uniform distribution of the requested random type.

  Raises:
    ValueError: if `random_type` is `STATELESS` and the `seed` is `None`.
  """
  random_type = RandomType.PSEUDO if random_type is None else random_type
  dtype = dtype or tf.float32
  with tf.compat.v1.name_scope(name, default_name='uniform_distribution',
                               values=[sample_shape]):

    if random_type == RandomType.PSEUDO:
      return tf.random.uniform(
          shape=sample_shape + [dim], dtype=dtype, seed=seed)
    elif random_type == RandomType.STATELESS:
      if seed is None:
        raise ValueError('`seed` must be supplied if the `random_type` is '
                         'STATELESS.')
      return tf.random.stateless_uniform(
          shape=sample_shape + [dim], dtype=dtype, seed=seed, alg='philox')
    # TODO(b/145104222): Add anthithetic sampling for the uniform distribution.
    elif random_type == RandomType.PSEUDO_ANTITHETIC:
      raise NotImplementedError(
          'At the moment antithetic sampling is not supported for the uniform '
          'distribution.')
    else:
      return _quasi_uniform(dim=dim,
                            sample_shape=sample_shape,
                            random_type=random_type,
                            dtype=dtype,
                            seed=seed,
                            **kwargs)


def _quasi_uniform(
    dim,
    sample_shape,
    random_type,
    dtype,
    seed=None,
    **kwargs):
  """Quasi random draws from a uniform distribution on [0, 1)."""
  # Shape of the output
  output_shape = tf.concat([sample_shape] + [[dim]], -1)
  # Number of quasi random samples
  num_samples = tf.reduce_prod(sample_shape)
  # Number of initial low discrepancy sequence numbers to skip
  if 'skip' in kwargs:
    skip = kwargs['skip']
  else:
    skip = 0
  if random_type == RandomType.SOBOL:
    # Shape [num_samples, dim] of the Sobol samples
    low_discrepancy_seq = sobol.sample(
        dim=dim, num_results=num_samples, skip=skip,
        dtype=dtype)
  else:  # HALTON or HALTON_RANDOMIZED random_dtype
    if 'randomization_params' in kwargs:
      randomization_params = kwargs['randomization_params']
    else:
      randomization_params = None
    randomized = random_type == RandomType.HALTON_RANDOMIZED
    # Shape [num_samples, dim] of the Sobol samples
    low_discrepancy_seq, _ = halton.sample(
        dim=dim,
        sequence_indices=tf.range(skip, skip + num_samples),
        randomized=randomized,
        randomization_params=randomization_params,
        seed=seed,
        dtype=dtype)
  return  tf.reshape(low_discrepancy_seq, output_shape)
