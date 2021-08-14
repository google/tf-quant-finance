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
"""Support for digital nets.

This module enables sampling of a digital net from a set of generating matrices.
A generating matrix is a binary matrix with shape `(m, n)` implemented as a
scalar `Tensor` of integers with shape `(1, n)` by multiplexing the bits of each
column using MSB 0 bit numbering.

In mathematical terms, the `Tensor` `T` representing a generating matrix `M`
with shape `(m, n)` is obtained by applying the following formula:
`T[0,j] = sum(M[i,j] * 2^(n - 1 - i); 0 <= i < n)`

For instance, the representation of the identity matrix of size 5 is:
`[[16, 8, 4, 2, 1]]`.

With this convention, a set of `N` generating matrices with shape `(m, n)` is
implemented as a single scalar `Tensor` of integers with shape `(N, n)`.
"""

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.math.qmc import utils

__all__ = [
    'digital_net_sample',
    'random_digital_shift',
    'random_scrambling_matrices',
    'scramble_generating_matrices',
]


def random_digital_shift(dim: types.IntTensor,
                         num_digits: types.IntTensor,
                         seed: int,
                         validate_args: bool = False,
                         dtype: tf.DType = None,
                         name: str = None) -> types.IntTensor:
  """Returns a `Tensor` drawn from a uniform distribution.

  The result can be can be passed to the `sample_digital_net` function to shift
  sampled points through a bitwise xor.

  #### Examples

  ```python
  import tf_quant_finance as tff

  # Example: Creating a Digital shift which can randomize sampled 2D points.

  dim = 2
  num_digits = 10
  seed = (2, 3)

  tff.math.qmc.random_digital_shift(dim, num_digits, seed=seed)
  # ==> tf.Tensor([586, 1011], shape=(2,), dtype=int32)
  ```

  Args:
    dim: Positive scalar `Tensor` of integers with rank 0. The event size of the
      sampled points to shift.
    num_digits: Positive scalar `Tensor` of integers with rank 0. the base-2
      precision of the sampled points to shift.
    seed: Positive scalar `Tensor` with shape [2] and dtype `int32` used as seed
      for the random generator.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either
      `tf.int32` or `tf.int64`).
      Default value: `None` which maps to `tf.int32`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `random_digital_shift`.

  Returns:
    A `Tensor` with `shape` `(dim,)`.
  """

  return _random_stateless_uniform((dim,),
                                   num_digits,
                                   seed,
                                   validate_args=validate_args,
                                   dtype=dtype,
                                   name=name or 'random_digital_shift')


def random_scrambling_matrices(dim: types.IntTensor,
                               num_digits: types.IntTensor,
                               seed: int,
                               validate_args: bool = False,
                               dtype: tf.DType = None,
                               name: str = None) -> types.IntTensor:
  """Returns a `Tensor` drawn from a uniform distribution.

  The result can be can be passed to the `scramble_generating_matrices` function
  to randomize a given `generating_matrices`.

  #### Examples

  ```python
  import tf_quant_finance as tff

  # Example: Creating random matrices which can scramble 2D generating matrices.

  dim = 2
  num_digits = 10
  seed = (2, 3)

  tff.math.qmc.random_scrambling_matrices(dim, num_digits, seed=seed)
  # ==> tf.Tensor([
  #             [586, 1011, 896,  818, 550, 1009, 880, 855,  686, 758],
  #             [872,  958, 870, 1000, 963,  919, 994, 583, 1007, 739],
  #         ], shape=(2, 10), dtype=int32)
  ```

  Args:
    dim: Positive scalar `Tensor` of integers with rank 0. The event size of
      points which can be sampled from the generating matrices to scramble.
    num_digits: Positive scalar `Tensor` of integers with rank 0. the base-2
      precision of the points which can be sampled from the generating matrices
      to scramble.
    seed: Positive scalar `Tensor` with shape [2] and dtype `int32` used as seed
      for the random generator.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either
      `tf.int32` or `tf.int64`).
      Default value: `None` which maps to `tf.int32`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `random_scrambling_matrices`.

  Returns:
    A `Tensor` with `shape` `(dim, num_digits)`.
  """

  return _random_stateless_uniform((dim, num_digits),
                                   num_digits,
                                   seed,
                                   validate_args=validate_args,
                                   dtype=dtype,
                                   name=name or 'random_scrambling_matrices')


def _random_stateless_uniform(shape: types.IntTensor,
                              num_digits: types.IntTensor,
                              seed: int,
                              validate_args: bool = False,
                              dtype: tf.DType = None,
                              name: str = None) -> types.IntTensor:
  """Returns a `Tensor` drawn from a uniform distribution with a given `shape`.

  Args:
    shape: Positive scalar `Tensor` of integers with rank 1. The shape of the
      returned `Tensor`.
    num_digits: Positive scalar `Tensor` of integers with rank 0. the base-2
      precision of the points which can be sampled from `generating_matrices`.
    seed: Positive scalar `Tensor` with shape [2] and dtype `int32` used as seed
      for the random generator.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either
      `tf.int32` or `tf.int64`).
      Default value: `None` which maps to `tf.int32`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `random_stateless_uniform`.

  Returns:
    A `Tensor` with the requested `shape`.
  """

  with tf.name_scope(name or 'random_stateless_uniform'):
    dtype = dtype or tf.int32

    shape = tf.convert_to_tensor(shape, dtype=dtype, name='dim')
    num_digits = tf.convert_to_tensor(
        num_digits, dtype=dtype, name='num_digits')

    control_deps = []
    if validate_args:
      control_deps.append(
          tf.debugging.assert_positive(shape, message='shape must be positive'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_digits, message='num_digits must be positive'))

    with tf.control_dependencies(control_deps):
      minval = tf.cast(utils.exp2(num_digits - 1), dtype=dtype)
      maxval = tf.cast(utils.exp2(num_digits), dtype=dtype)

      return tf.random.stateless_uniform(
          shape, seed, minval=minval, maxval=maxval, dtype=dtype)


def digital_net_sample(generating_matrices: types.IntTensor,
                       num_results: types.IntTensor,
                       num_digits: types.IntTensor,
                       sequence_indices: types.IntTensor = None,
                       scrambling_matrices: types.IntTensor = None,
                       digital_shift: types.IntTensor = None,
                       apply_tent_transform: bool = False,
                       validate_args: bool = False,
                       dtype: tf.DType = None,
                       name: str = None) -> types.IntTensor:
  r"""Constructs a digital net from a generating matrix.

  #### Examples

  ```python
  import tf_quant_finance as tff

  # Example: Sampling 1,000 points from 2D Sobol generating matrices.

  dim = 2
  num_results = 1000
  num_digits = 10

  tff.math.qmc.digital_net_sample(
      tff.math.qmc.sobol_generating_matrices(dim, num_results, num_digits),
      num_results,
      num_digits)
  # ==> tf.Tensor([
  #             [0.,         0.        ],
  #             [0.5,        0.5       ],
  #             [0.25,       0.75      ],
  #             ...
  #             [0.65527344, 0.9736328 ],
  #             [0.40527344, 0.7236328 ],
  #             [0.90527344, 0.22363281],
  #         ], shape=(1000, 2), dtype=float32)
  ```

  Args:
    generating_matrices: Positive scalar `Tensor` of integers with rank 2. The
      matrix from which to sample points.
    num_results: Positive scalar `Tensor` of integers with rank 0. The maximum
      number of points to sample from `generating_matrices`.
    num_digits: Positive scalar `Tensor` of integers with rank 0. the base-2
      precision of the points sampled from `generating_matrices`.
    sequence_indices: Optional positive scalar `Tensor` of integers with rank 1.
      The elements of the sequence to return specified by their position in the
      sequence.
      Default value: `None` which corresponds to the `[0, num_results)` range.
    scrambling_matrices: Optional positive scalar `Tensor` of integers with the
      same shape as `generating_matrices`. The left matrix scramble to apply to
      the generating matrices.
      Default value: `None`.
    digital_shift: Optional positive scalar `Tensor` of integers with shape
      (`dim`) where `dim = tf.shape(generating_matrices)[0]`. The digital shift
      to apply to all the sampled points via a bitwise xor.
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
      Default value: `None` which maps to `sample_digital_net`.

  Returns:
    A `Tensor` of samples from  the Sobol sequence with `shape`
    `(num_samples, dim)` where `num_samples = min(num_results,
    size(sequence_indices))` and `dim = tf.shape(generating_matrices)[0]`.
  """

  with tf.name_scope(name or 'sample_digital_net'):
    # shape: (dim, log_num_results)
    generating_matrices = tf.convert_to_tensor(
        generating_matrices, name='generating_matrices')
    if scrambling_matrices is not None:
      scrambling_matrices = tf.convert_to_tensor(
          scrambling_matrices, name='scrambling_matrices')

    dim = utils.get_shape(generating_matrices)[0]
    int_dtype = generating_matrices.dtype
    real_dtype = dtype or tf.float32

    num_results = tf.convert_to_tensor(
        num_results, dtype=int_dtype, name='num_results')
    num_digits = tf.convert_to_tensor(
        num_digits, dtype=int_dtype, name='num_digits')

    log_num_results = tf.cast(
        tf.math.ceil(utils.log2(tf.cast(num_results, tf.float32))), int_dtype,
        'log_num_results')

    # shape: (num_samples,)
    if sequence_indices is not None:
      sequence_indices = tf.cast(
          sequence_indices, int_dtype, name='sequence_indices')

    control_deps = []
    if validate_args:
      control_deps.append(
          tf.debugging.assert_equal(
              tf.rank(generating_matrices),
              2,
              message='generating_matrices must have rank 2'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_results, message='num_results must be positive'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_digits, message='num_digits must be positive'))
      control_deps.append(
          tf.debugging.assert_less(
              log_num_results,
              tf.cast(32, int_dtype),
              message='log2(num_results) must be less than 32'))
      if sequence_indices is not None:
        control_deps.append(
            tf.debugging.assert_equal(
                tf.rank(sequence_indices),
                1,
                message='sequence_indices must have rank 1'))
        control_deps.append(
            tf.debugging.assert_less(
                sequence_indices,
                num_results,
                message='values in sequence_indices must be less than num_results'
            ))
      if scrambling_matrices is not None:
        control_deps.append(
            tf.debugging.assert_equal(
                utils.get_shape(scrambling_matrices),
                utils.get_shape(generating_matrices),
                message='scrambling_matrices must have the same shape as ' +
                'generating_matrices'))
      if digital_shift is not None:
        control_deps.append(
            tf.debugging.assert_equal(
                tf.rank(digital_shift),
                1,
                message='digital_shift must have rank 1'))
        control_deps.append(
            tf.debugging.assert_equal(
                tf.size(digital_shift),
                dim,
                message='digital_shift must have size ' +
                'tf.shape(generating_matrices)[0]'))

    # shape: (num_samples,)
    if sequence_indices is None:
      sequence_indices = tf.range(
          0, num_results, dtype=int_dtype, name='sequence_indices')

    with tf.control_dependencies(control_deps):
      # shape: (dim)
      if digital_shift is None:
        digital_shift = tf.zeros(
            shape=(dim), dtype=int_dtype, name='digital_shift')
      else:
        digital_shift = tf.cast(digital_shift, int_dtype, name='digital_shift')

      if scrambling_matrices is not None:
        generating_matrices = scramble_generating_matrices(
            generating_matrices,
            scrambling_matrices,
            num_digits,
            validate_args=validate_args,
            dtype=int_dtype)

      # shape: (1, dim, log_num_results)
      generating_matrices = tf.expand_dims(generating_matrices, axis=0)

      def loop_predicate_fn(binary_points, log_index):
        del binary_points
        return log_index < log_num_results

      def loop_body_fn(binary_points, log_index):
        # shape: (num_samples, dim)
        updated_binary_points = tf.bitwise.bitwise_xor(
            binary_points,
            utils.filter_tensor(
                # shape: (1, dim)
                tf.gather(generating_matrices, log_index, axis=2),
                # shape: (num_samples, 1)
                tf.cast(tf.expand_dims(sequence_indices, axis=1), int_dtype),
                # shape: ()
                log_index))

        return (updated_binary_points, log_index + 1)

      binary_points, _ = tf.while_loop(
          loop_predicate_fn,
          loop_body_fn,
          loop_vars=(
              # shape: (num_samples, dim)
              tf.repeat(
                  tf.expand_dims(digital_shift, 0), tf.size(sequence_indices),
                  0),
              # shape: ()
              tf.constant(0, dtype=int_dtype)),
          maximum_iterations=tf.cast(log_num_results, tf.int32))

      # shape: ()
      max_binary_point = tf.bitwise.left_shift(
          tf.constant(1, dtype=int_dtype), num_digits)

      # shape: (num_samples, dim)
      points = tf.divide(
          tf.cast(binary_points, real_dtype),
          tf.cast(max_binary_point, real_dtype))

      # shape: (num_samples, dim)
      return utils.tent_transform(points) if apply_tent_transform else points


def scramble_generating_matrices(generating_matrices: types.IntTensor,
                                 scrambling_matrices: types.IntTensor,
                                 num_digits: types.IntTensor,
                                 validate_args: bool = False,
                                 dtype: tf.DType = None,
                                 name: str = None):
  r"""Scrambles a generating matrix.

  #### Examples

  ```python
  import tf_quant_finance as tff

  # Example: Scrambling the 2D Sobol generating matrices.

  dim = 2
  num_results = 1000
  num_digits = 10
  seed = (2, 3)

  tff.math.qmc.scramble_generating_matrices(
      tff.math.qmc.sobol_generating_matrices(dim, num_results, num_digits),
      tff.math.qmc.random_scrambling_matrices(dim, num_digits, seed=seed),
      num_digits)
  # ==> tf.Tensor([
  #             [586, 505, 224, 102,  34,  31,  13,   6,   2,   1],
  #             [872, 695, 945, 531, 852, 663, 898, 568, 875, 693],
  #         ], shape=(2, 10), dtype=int32)
  ```

  Args:
    generating_matrices: Positive scalar `Tensor` of integers.
    scrambling_matrices: Positive Scalar `Tensor` of integers with the same
      `shape` as `generating_matrices`.
    num_digits: Positive scalar `Tensor` of integers with rank 0. The base-2
      precision of the points which can be sampled from `generating_matrices`.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either `int32`
      or `int64`).
      Default value: `None` which maps to `generating_matrices.dtype`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `scramble_generating_matrices`.

  Returns:
    A `Tensor` with the same `shape` and `dtype` as `generating_matrices`.
  """

  with tf.name_scope(name or 'scramble_generating_matrices'):
    if dtype is None:
      generating_matrices = tf.convert_to_tensor(generating_matrices)

    dtype = dtype or generating_matrices.dtype

    num_digits = tf.convert_to_tensor(
        num_digits, dtype=dtype, name='num_digits')

    # shape: (dim, log_num_results)
    generating_matrices = tf.cast(
        generating_matrices, dtype=dtype, name='generating_matrices')

    # shape: (dim, log_num_results)
    scrambling_matrices = tf.cast(
        scrambling_matrices, dtype=dtype, name='scrambling_matrices')

    control_deps = []
    if validate_args:
      control_deps.append(
          tf.debugging.assert_equal(
              tf.rank(generating_matrices),
              tf.rank(scrambling_matrices),
              message='input matrices must have the same rank'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_digits, message='num_digits must be positive'))

    with tf.control_dependencies(control_deps):

      def loop_predicate_fn(matrix, shift):
        del matrix
        return shift < num_digits

      def loop_body_fn(matrix, shift):
        # shape: (dim, log_num_results)
        shifted_scrambling_matrices = tf.bitwise.right_shift(
            tf.gather(scrambling_matrices, [shift], axis=1), shift)

        # shape: (dim, log_num_results)
        updated_matrix = tf.bitwise.bitwise_xor(
            matrix,
            utils.filter_tensor(shifted_scrambling_matrices,
                                generating_matrices, num_digits - 1 - shift))

        return (updated_matrix, shift + 1)

      matrix, _ = tf.while_loop(
          loop_predicate_fn,
          loop_body_fn,
          loop_vars=(
              tf.zeros_like(generating_matrices),
              tf.constant(0, dtype=dtype),
          ),
          maximum_iterations=tf.cast(num_digits, tf.int32))

      # shape: (dim, log_num_results)
      return matrix
