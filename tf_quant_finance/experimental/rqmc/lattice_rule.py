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
"""Support for lattice rules."""

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.experimental.rqmc import utils

__all__ = [
    'random_scrambling_vectors',
    'sample_lattice_rule',
]


def random_scrambling_vectors(generating_vectors: types.IntTensor,
                              seed: int,
                              validate_args: bool = False,
                              dtype: tf.DType = None,
                              name: str = None) -> types.RealTensor:
  r"""Returns a `Tensor` drawn from a uniform distribution.

  The returned `Tensor` can be can be added to the specified
  `generating_vectors` in order to randomize it.

  Args:
    generating_vectors: Positive scalar `Tensor` of integers with rank 1.
    seed: Positive scalar `Tensor` with shape [2] and dtype `int32` used as seed
      for the random enerator.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either
      `float32` or `float64`).
      Default value: `None` which maps to `tf.float32`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `random_scrambling_vectors`.

  Returns:
    A `Tensor` of real values between 0 (incl.) and 1 (excl.) with the same
    `shape` as `generating_vectors`.
  """

  with tf.name_scope(name or 'random_scrambling_vectors'):
    control_deps = []
    if validate_args:
      control_deps.append(
          tf.debugging.assert_equal(
              tf.rank(generating_vectors),
              1,
              message='generating_vectors must have rank 1'))

    with tf.control_dependencies(control_deps):
      return tf.random.stateless_uniform(
          utils.get_shape(generating_vectors),
          seed,
          minval=0.,
          maxval=1.,
          dtype=dtype or tf.float32)


def sample_lattice_rule(generating_vectors: types.IntTensor,
                        dim: types.IntTensor,
                        num_results: types.IntTensor,
                        sequence_indices: types.IntTensor = None,
                        additive_shift: types.FloatTensor = None,
                        apply_tent_transform: bool = False,
                        validate_args: bool = False,
                        dtype: tf.DType = None,
                        name: str = None) -> types.RealTensor:
  r"""Constructs a lattice rule from a generating vector.

  Args:
    generating_vectors: Positive scalar `Tensor` of integers with rank 1
      representing the vector from which to sample points.
    dim: Positive scalar `Tensor` of integers with rank 0. The event size of the
      sampled points. Must not exceed the size of `generating_vectors`.
    num_results: Positive scalar `Tensor` of integers with rank 0. The maximum
      number of points to sample.
    sequence_indices: Optional positive scalar `Tensor` of integers with rank 1.
      The elements of the sequence to return specified by their position in the
      sequence.
      Default value: `None` which corresponds to the `[0, num_results)` range.
    additive_shift: Optional scalar `Tensor` of real values with the same
      `shape` as `generating_vectors`. The additive shift to add to all the
      points (modulo 1) before applying the tent transform.
      Default value: `None`.
    apply_tent_transform: Python `bool` indicating whether to apply a tent
      transform to the sampled points.
      Default value: `False`.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either
      `float32` or `float64`).
      Default value: `None` which maps to `float32`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `sample_lattice_rule`.

  Returns:
    A `Tensor` of samples from  the Sobol sequence with `shape`
    `(num_samples,)` where `num_samples = min(num_results,
    size(sequence_indices))`.
  """

  with tf.name_scope(name or 'sample_lattice_rule'):
    # shape: (?,)
    generating_vectors = tf.convert_to_tensor(
        generating_vectors, name='generating_vectors')

    int_dtype = generating_vectors.dtype
    real_dtype = dtype or tf.float32

    dim = tf.convert_to_tensor(dim, dtype=int_dtype, name='dim')
    num_results = tf.convert_to_tensor(
        num_results, dtype=int_dtype, name='num_results')

    control_deps = []
    if validate_args:
      control_deps.append(
          tf.debugging.assert_equal(
              tf.rank(generating_vectors),
              1,
              message='generating_vectors must have rank 1'))
      control_deps.append(
          tf.debugging.assert_less_equal(
              dim,
              tf.size(generating_vectors, out_type=int_dtype),
              message='dim must not exceed the size of generating_vectors'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_results, message='num_results must be positive'))

    with tf.control_dependencies(control_deps):
      # shape: (num_samples,)
      if sequence_indices is None:
        sequence_indices = tf.range(0, num_results)
      sequence_indices = tf.cast(
          sequence_indices, int_dtype, name='sequence_indices')

      unit = tf.ones(shape=(), dtype=real_dtype)

      # shape: (dim,)
      scaled_vector = tf.divide(
          # shape: (dim,)
          tf.cast(generating_vectors[:dim], real_dtype),
          # shape: ()
          tf.cast(num_results, real_dtype))

      # shape: (num_samples, dim)
      points = tf.multiply(
          # shape: (num_samples, 1)
          tf.expand_dims(tf.cast(sequence_indices, real_dtype), axis=1),
          # shape: (1, dim)
          tf.expand_dims(tf.math.floormod(scaled_vector, unit), axis=0))

      if additive_shift is not None:
        # shape: (num_results,)
        additive_shift = tf.cast(
            additive_shift, real_dtype, name='additive_shift')
        # shape: (num_samples, dim)
        points += additive_shift[:dim]

        # shape: (num_samples, dim)
      points = tf.math.floormod(points, unit)

      # shape: (num_samples, dim)
      return utils.tent_transform(points) if apply_tent_transform else points
