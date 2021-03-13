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

"""Multivariate Normal distribution with various random types."""

import enum
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.random_ops import halton
from tf_quant_finance.math.random_ops import sobol


_SQRT_2 = np.sqrt(2.)


@enum.unique
class RandomType(enum.Enum):
  """Types of random number sequences.

  * `PSEUDO`: The standard Tensorflow random generator.
  * `STATELESS`: The stateless Tensorflow random generator.
  * `HALTON`: The standard Halton sequence.
  * `HALTON_RANDOMIZED`: The randomized Halton sequence.
  * `SOBOL`: The standard Sobol sequence.
  * `PSEUDO_ANTITHETIC`: PSEUDO random numbers along with antithetic variates.
  """
  PSEUDO = 0
  STATELESS = 1
  HALTON = 2
  HALTON_RANDOMIZED = 3
  SOBOL = 4
  PSEUDO_ANTITHETIC = 5
  STATELESS_ANTITHETIC = 6


def multivariate_normal(sample_shape,
                        mean=None,
                        covariance_matrix=None,
                        scale_matrix=None,
                        random_type=None,
                        validate_args=False,
                        seed=None,
                        dtype=None,
                        name=None,
                        **kwargs):
  """Generates draws from a multivariate Normal distribution.

  Draws samples from the multivariate Normal distribution on `R^k` with the
  supplied mean and covariance parameters. Allows generating either
  (pseudo) random or quasi-random draws based on the `random_type` parameter.

  #### Example:

  ```python

  # A single batch example.
  sample_shape = [10]  # Generates 10 draws.
  mean = [0.1, 0.2]  # The mean of the distribution. A single batch.
  covariance = [[1.0, 0.1], [0.1, 0.9]]
  # Produces a Tensor of shape [10, 2] containing 10 samples from the
  # 2 dimensional normal. The draws are generated using the standard random
  # number generator in TF.
  sample = multivariate_normal(sample_shape, mean=mean,
                               covariance_matrix=covariance,
                               random_type=RandomType.PSEUDO)

  # Produces a Tensor of shape [10, 2] containing 10 samples from the
  # 2 dimensional normal. Here the draws are generated using the stateless
  # random number generator. Note that a seed parameter is required and may
  # not be omitted. For the fixed seed, the same numbers will be produced
  # regardless of the rest of the graph or across independent sessions.
  sample_stateless = multivariate_normal(sample_shape, mean=mean,
                                         covariance_matrix=covariance,
                                         random_type=RandomType.STATELESS,
                                         seed=1234)

  # A multi-batch example. We can simultaneously draw from more than one
  # set of parameters similarly to the behaviour in tensorflow distributions
  # library.
  sample_shape = [5, 4]  # Twenty samples arranged as a 5x4 matrix.
  means = [[1.0, -1.0], [0.0, 2.0],[0.3, 1.4]]  # A batch of three mean vectors.
  # This demonstrates the broadcasting of the parameters. While we have
  # a batch of 3 mean vectors, we supply only one covariance matrix. This means
  # that three distributions have different means but the same covariance.
  covariances = [[1.0, 0.1], [0.1, 1.0]]

  # Produces a Tensor of shape [5, 4, 3, 2] containing 20 samples from the
  # batch of 3 bivariate normals.
  sample_batch = multivariate_normal(sample_shape, mean=means,
                                     covariance_matrix=covariance)

  Args:
    sample_shape: Rank 1 `Tensor` of positive `int32`s. Should specify a valid
      shape for a `Tensor`. The shape of the samples to be drawn.
    mean: Real `Tensor` of rank at least 1 or None. The shape of the `Tensor` is
      interpreted as `batch_shape + [k]` where `k` is the dimension of domain.
      The mean value(s) of the distribution(s) to draw from.
      Default value: None which is mapped to a zero mean vector.
    covariance_matrix: Real `Tensor` of rank at least 2 or None. Symmetric
      positive definite `Tensor` of  same `dtype` as `mean`. The strict upper
      triangle of `covariance_matrix` is ignored, so if `covariance_matrix` is
      not symmetric no error will be raised (unless `validate_args is True`).
      `covariance_matrix` has shape `batch_shape + [k, k]` where `b >= 0` and
      `k` is the event size.
      Default value: None which is mapped to the identity covariance.
    scale_matrix: Real `Tensor` of rank at least 2 or None. If supplied, it
      should be positive definite `Tensor` of same `dtype` as `mean`. The
      covariance matrix is related to the scale matrix by `covariance =
      scale_matrix * Transpose(scale_matrix)`.
      Default value: None which corresponds to an identity covariance matrix.
    random_type: Enum value of `RandomType`. The type of draw to generate.
      For `PSEUDO_ANTITHETIC` and `STATELESS_ANTITHETIC` the first dimension of
      `sample_shape` is expected to be an even positive integer. The antithetic
      pairs are then contained in the slices of the `output` tensor as
      `output[:(sample_shape[0] / 2), ...]` and
      `output[(sample_shape[0] / 2):, ...]`.
      Default value: None which is mapped to `RandomType.PSEUDO`.
    validate_args: Python `bool`. When `True`, distribution parameters are
      checked for validity despite possibly degrading runtime performance. When
      `False` invalid inputs may silently render incorrect outputs.
      Default value: False.
    seed: Seed for the random number generator. The seed is
      only relevant if `random_type` is one of
      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
      `HALTON_RANDOMIZED` the seed should be a Python integer. For
      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
      `Tensor` of shape `[2]`.
      Default value: `None` which means no seed is set.
    dtype: Optional `dtype`. The dtype of the input and output tensors.
      Default value: None which maps to the default dtype inferred by
      TensorFlow.
    name: Python `str` name prefixed to Ops created by this class.
      Default value: None which is mapped to the default name
        'multivariate_normal'.
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
    A `Tensor` of shape `sample_shape + batch_shape + [k]`. The draws from the
    multivariate normal distribution.

  Raises:
    ValueError:
      (a) If all of `mean`, `covariance_matrix` and `scale_matrix` are None.
      (b) If both `covariance_matrix` and `scale_matrix` are specified.
      (c) If `random_type` is either `RandomType.STATELESS` or
      `RandomType.STATELESS_ANTITHETIC` and `seed` is not supplied
    NotImplementedError: If `random_type` is neither RandomType.PSEUDO,
      RandomType.STATELESS, RandomType.PSEUDO_ANTITHETIC,
      RandomType.STATELESS_ANTITHETIC, RandomType.SOBOL, RandomType.HALTON, nor
      RandomType.HALTON_RANDOMIZED.
  """
  random_type = RandomType.PSEUDO if random_type is None else random_type
  # Convert sample_shape to a list
  if mean is None and covariance_matrix is None and scale_matrix is None:
    raise ValueError('At least one of mean, covariance_matrix or scale_matrix'
                     ' must be specified.')

  if covariance_matrix is not None and scale_matrix is not None:
    raise ValueError('Only one of covariance matrix or scale matrix'
                     ' must be specified')

  name = name or 'multivariate_normal'
  with tf.name_scope(name):
    if mean is not None:
      mean = tf.convert_to_tensor(mean, dtype=dtype, name='mean')
    if random_type in [RandomType.PSEUDO, RandomType.STATELESS]:
      return _mvnormal_pseudo(
          sample_shape,
          mean,
          covariance_matrix=covariance_matrix,
          scale_matrix=scale_matrix,
          random_type=random_type,
          seed=seed,
          dtype=dtype)
    elif random_type in [RandomType.PSEUDO_ANTITHETIC,
                         RandomType.STATELESS_ANTITHETIC]:
      return _mvnormal_pseudo_antithetic(
          sample_shape,
          mean,
          covariance_matrix=covariance_matrix,
          scale_matrix=scale_matrix,
          random_type=random_type,
          seed=seed,
          dtype=dtype)
    elif random_type == RandomType.SOBOL:
      return _mvnormal_sobol(
          sample_shape,
          mean,
          covariance_matrix=covariance_matrix,
          scale_matrix=scale_matrix,
          validate_args=validate_args,
          dtype=dtype,
          **kwargs)
    elif random_type == RandomType.HALTON:
      return _mvnormal_halton(
          sample_shape,
          mean,
          randomized=False,
          seed=seed,
          covariance_matrix=covariance_matrix,
          scale_matrix=scale_matrix,
          validate_args=validate_args,
          dtype=dtype,
          **kwargs)
    elif random_type == RandomType.HALTON_RANDOMIZED:
      return _mvnormal_halton(
          sample_shape,
          mean,
          randomized=True,
          seed=seed,
          covariance_matrix=covariance_matrix,
          scale_matrix=scale_matrix,
          validate_args=validate_args,
          dtype=dtype,
          **kwargs)
    else:
      raise NotImplementedError(
          'Only STATELESS, PSEUDO, PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC,  '
          'HALTON, HALTON_RANDOMIZED, and SOBOL random types are currently '
          'supported. Supplied: {}'. format(random_type))


def _mvnormal_pseudo(sample_shape,
                     mean,
                     covariance_matrix=None,
                     scale_matrix=None,
                     random_type=RandomType.PSEUDO,
                     seed=None,
                     dtype=None):
  """Returns normal draws using the tfp multivariate normal distribution."""
  (mean, scale_matrix,
   batch_shape, _, dtype) = _process_mean_scale(mean, scale_matrix,
                                                covariance_matrix, dtype)
  output_shape = tf.concat([sample_shape, batch_shape], axis=0)
  if random_type == RandomType.PSEUDO:
    samples = tf.random.normal(shape=output_shape,
                               dtype=dtype,
                               seed=seed)
  else:
    if seed is None:
      raise ValueError('`seed` should be specified if the `random_type` is '
                       '`STATELESS` or `STATELESS_ANTITHETIC`')
    samples = tf.random.stateless_normal(
        shape=output_shape, dtype=dtype, seed=seed)
  if scale_matrix is None:
    return mean + samples
  else:
    return mean + tf.linalg.matvec(scale_matrix, samples)


def _mvnormal_pseudo_antithetic(sample_shape,
                                mean,
                                covariance_matrix=None,
                                scale_matrix=None,
                                random_type=RandomType.PSEUDO_ANTITHETIC,
                                seed=None,
                                dtype=None):
  """Returns normal draws with the antithetic samples."""
  sample_shape = tf.TensorShape(sample_shape).as_list()
  sample_zero_dim = sample_shape[0]
  # For the antithetic sampler `sample_shape` is split evenly between
  # samples and their antithetic counterparts. In order to do the splitting
  # we expect the first dimension of `sample_shape` to be even.
  is_even_dim = tf.compat.v1.debugging.assert_equal(
      sample_zero_dim % 2,
      0,
      message='First dimension of `sample_shape` should be even for '
      'PSEUDO_ANTITHETIC random type')
  with tf.control_dependencies([is_even_dim]):
    antithetic_shape = [sample_zero_dim // 2] + sample_shape[1:]
  if random_type == RandomType.PSEUDO_ANTITHETIC:
    random_type_sample = RandomType.PSEUDO
  else:
    random_type_sample = RandomType.STATELESS
  result = _mvnormal_pseudo(
      antithetic_shape,
      mean,
      covariance_matrix=covariance_matrix,
      scale_matrix=scale_matrix,
      random_type=random_type_sample,
      seed=seed,
      dtype=dtype)
  if mean is None:
    return tf.concat([result, -result], axis=0)
  else:
    return tf.concat([result, 2 * mean - result], axis=0)


def _mvnormal_sobol(sample_shape,
                    mean,
                    covariance_matrix=None,
                    scale_matrix=None,
                    validate_args=False,
                    dtype=None,
                    **kwargs):
  """Returns normal draws using Sobol low-discrepancy sequences."""
  return _mvnormal_quasi(sample_shape,
                         mean,
                         random_type=RandomType.SOBOL,
                         seed=None,
                         covariance_matrix=covariance_matrix,
                         scale_matrix=scale_matrix,
                         validate_args=validate_args,
                         dtype=dtype,
                         **kwargs)


def _mvnormal_halton(sample_shape,
                     mean,
                     randomized,
                     seed=None,
                     covariance_matrix=None,
                     scale_matrix=None,
                     validate_args=False,
                     dtype=None,
                     **kwargs):
  """Returns normal draws using Halton low-discrepancy sequences."""
  random_type = (RandomType.HALTON_RANDOMIZED if randomized
                 else RandomType.HALTON)
  return _mvnormal_quasi(sample_shape,
                         mean,
                         random_type,
                         seed=seed,
                         covariance_matrix=covariance_matrix,
                         scale_matrix=scale_matrix,
                         validate_args=validate_args,
                         dtype=dtype,
                         **kwargs)


def _mvnormal_quasi(sample_shape,
                    mean,
                    random_type,
                    seed,
                    covariance_matrix=None,
                    scale_matrix=None,
                    validate_args=False,
                    dtype=None,
                    **kwargs):
  """Returns normal draws using low-discrepancy sequences."""
  (mean, scale_matrix,
   batch_shape, dim, dtype) = _process_mean_scale(mean, scale_matrix,
                                                  covariance_matrix, dtype)
  # Reverse elements of the batch shape
  batch_shape_reverse = tf.reverse(batch_shape, [0])
  # Transposed shape of the output
  output_shape_t = tf.concat([batch_shape_reverse, sample_shape], -1)
  # Number of quasi random samples
  num_samples = tf.reduce_prod(output_shape_t) // dim
  # Number of initial low discrepancy sequence numbers to skip
  if 'skip' in kwargs:
    skip = kwargs['skip']
  else:
    skip = 0
  if random_type == RandomType.SOBOL:
    # TODO(b/182621549): For Sobol sequences, dimension should be known at graph
    # construction time.
    dim = tf.get_static_value(dim)
    if dim is None:
      raise ValueError('For Sobol sequences, dimension should be known at graph'
                       ' construction time.')
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
        validate_args=validate_args,
        dtype=dtype)

  # Transpose to the shape [dim, num_samples]
  low_discrepancy_seq = tf.transpose(low_discrepancy_seq)
  size_sample = tf.size(sample_shape)
  size_batch = tf.size(batch_shape)
  # Permutation for `output_shape_t` to the output shape
  permutation = tf.concat([tf.range(size_batch, size_batch + size_sample),
                           tf.range(size_batch - 1, -1, -1)], -1)
  # Reshape Sobol samples to the correct output shape
  low_discrepancy_seq = tf.transpose(
      tf.reshape(low_discrepancy_seq, output_shape_t),
      permutation)
  # Apply inverse Normal CDF to Sobol samples to obtain the corresponding
  # Normal samples
  samples = tf.math.erfinv((low_discrepancy_seq - 0.5) * 2)* _SQRT_2
  if scale_matrix is None:
    return mean + samples
  else:
    return mean + tf.linalg.matvec(scale_matrix, samples)


def _process_mean_scale(mean, scale_matrix, covariance_matrix, dtype):
  """Extracts correct mean, scale, batch_shape, dimension, and dtype."""
  if scale_matrix is not None:
    scale_matrix = tf.convert_to_tensor(scale_matrix, dtype=dtype,
                                        name='scale_matrix')
  else:
    if covariance_matrix is not None:
      covariance_matrix = tf.convert_to_tensor(covariance_matrix, dtype=dtype,
                                               name='covariance_matrix')
      scale_matrix = tf.linalg.cholesky(covariance_matrix)
  if mean is None:
    mean = 0.0
    # batch_shape includes the dimension of the samples
    batch_shape = tf.shape(scale_matrix)[:-1]
    # TODO(b/182621549): For Sobol sequences, dimension should be known at graph
    # construction time.
    dim = _get_static_dim(scale_matrix)
    dtype = scale_matrix.dtype
  else:
    batch_shape = tf.shape(mean)
    # TODO(b/182621549): For Sobol sequences, dimension should be known at graph
    # construction time.
    dim = _get_static_dim(mean)
    dtype = mean.dtype
  return mean, scale_matrix, batch_shape, dim, dtype


def _get_static_dim(t):
  """Returns static dimension value if possible."""
  dim = t.shape.as_list()[-1]
  if dim is None:
    return tf.shape(t)[-1]
  else:
    return dim
