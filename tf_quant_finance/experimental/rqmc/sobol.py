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
"""Support for Sobol sequence generation."""

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.experimental.rqmc import digital_net
from tf_quant_finance.experimental.rqmc import utils
from tf_quant_finance.math.random_ops import sobol

__all__ = [
    'sample_sobol',
    'sobol_generating_matrices',
]

(_PRIMITIVE_POLYNOMIAL_COEFFICIENTS,
 _INITIAL_DIRECTION_NUMBERS) = sobol.load_data()


def sample_sobol(dim: types.IntTensor,
                 num_results: types.IntTensor,
                 sequence_indices: types.IntTensor = None,
                 digital_shift: types.IntTensor = None,
                 scrambling_matrices: types.IntTensor = None,
                 apply_tent_transform: bool = False,
                 validate_args: bool = False,
                 dtype: tf.DType = None,
                 name: str = None) -> types.RealTensor:
  r"""Samples points from the Sobol sequence.

  Args:
    dim: Positive scalar `Tensor` of integers with rank 0. The event size of the
      sampled points.
    num_results: Positive scalar `Tensor` of integers with rank 0. The number of
      points to sample.
    sequence_indices: Optional positive scalar `Tensor` of integers with rank 1.
      The elements of the sequence to return specified by their position in the
      sequence.
      Default value: `None` which corresponds to the `[0, num_results)` range.
    digital_shift: Optional digital shift to be applied to all the points via a
      bitwise xor.
      Default value: `None`.
    scrambling_matrices: Positive scalar `Tensor` with the same `shape` and
      `dtype` as `generating_matrices`. Used to randomize `generating_matrices`.
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
      Default value: `None` which maps to `sample_sobol`.

  Returns:
    A `Tensor` of samples from  the Sobol sequence with `shape`
    `(num_samples, dim)` where `num_samples = min(num_results,
    size(sequence_indices))` and `dim = tf.shape(generating_matrices)[0]`.
  """

  with tf.name_scope(name or 'sample_sobol'):
    dtype = dtype or tf.float32

    num_digits = tf.cast(
        tf.math.ceil(utils.log2(tf.cast(num_results, dtype=tf.float32))),
        tf.int32)

    # shape: (dim, log_num_results)
    generating_matrices = sobol_generating_matrices(
        dim,
        num_results,
        num_digits,
        validate_args=validate_args,
        dtype=tf.int32)

    if scrambling_matrices is not None:
      # shape: (dim, log_num_results)
      generating_matrices = digital_net.scramble_generating_matrices(
          generating_matrices,
          scrambling_matrices,
          num_digits,
          validate_args=validate_args)

    # shape: (num_results, dim)
    return digital_net.sample_digital_net(
        generating_matrices,
        num_results,
        num_digits,
        sequence_indices=sequence_indices,
        digital_shift=digital_shift,
        apply_tent_transform=apply_tent_transform,
        validate_args=validate_args,
        dtype=dtype)


def sobol_generating_matrices(dim: types.IntTensor,
                              num_results: types.IntTensor,
                              num_digits: types.IntTensor,
                              validate_args: bool = False,
                              dtype: tf.DType = None,
                              name: str = None) -> types.IntTensor:
  r"""Returns Sobol generating matrices.

  Args:
    dim: Positive scalar `Tensor` of integers with rank 0. The event size of
      points which can be sampled from the resulting generating matrices.
    num_results: Positive scalar `Tensor` of integers with rank 0. The maximum
      number of points which can be sampled from the resulting generating
      matrices.
    num_digits: Positive scalar `Tensor` of integers with rank 0. The base-2
      precision of points which can be sampled from the resulting generating
      matrices.
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either a signed
      or unsigned integer `dtype`).
      Default value: `None` which maps to `int32`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` which maps to `sobol_generating_matrices`.

  Returns:
    A scalar `Tensor` with shape `(dim, log_num_results)` where
    `log_num_results = ceil(log2(num_results))`.
  """

  with tf.name_scope(name or 'sobol_generating_matrices'):
    dtype = dtype or tf.int32

    dim = tf.convert_to_tensor(dim, dtype=dtype, name='dim')
    num_results = tf.convert_to_tensor(
        num_results, dtype=dtype, name='num_results')
    num_digits = tf.convert_to_tensor(
        num_digits, dtype=dtype, name='num_digits')

    log_num_results = tf.cast(
        tf.math.ceil(utils.log2(tf.cast(num_results, dtype=tf.float32))),
        dtype=dtype)

    control_deps = []
    if validate_args:
      control_deps.append(
          tf.debugging.assert_positive(dim, message='dim must be positive'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_results, message='num_results must be positive'))
      control_deps.append(
          tf.debugging.assert_positive(
              num_digits, message='num_digits must be positive'))
      control_deps.append(
          tf.debugging.assert_less(
              log_num_results,
              tf.constant(32, dtype=dtype),
              message='log2(num_results) must be less than 32'))

    with tf.control_dependencies(control_deps):
      # shape: (1, log_num_results)
      identity = _identity_matrix(log_num_results, num_digits, dtype=dtype)
      # shape: (dim - 1, log_num_results)
      matrices = _sobol_generating_matrices(
          dim - 1, log_num_results, num_digits, dtype=dtype)
      # shape: (dim, log_num_results)
      return tf.concat((identity, matrices), axis=0)


def _identity_matrix(num_columns: types.IntTensor,
                     num_digits: types.IntTensor,
                     dtype: tf.DType = None) -> types.IntTensor:
  r"""Returns the identity matrix.

  Args:
    num_columns: Positive scalar `Tensor` with rank 0 representing the number of
      columns of the returned matrix.
    num_digits: Positive scalar `Tensor` with rank 0 representing the base-2
      precision of the samples.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either a signed
      or unsigned integer `dtype`).
      Default value: `None` which maps to `int32`.

  Returns:
    A scalar `Tensor` with shape `(1, num_columns)`.
  """

  dtype = dtype or tf.int32
  shifts = tf.range(num_digits - 1, num_digits - 1 - num_columns, delta=-1)
  # shape: (1, num_columns)
  return tf.bitwise.left_shift(
      tf.ones(shape=(1, num_columns), dtype=dtype), tf.cast(shifts, dtype))


def _sobol_generating_matrices(dim: types.IntTensor,
                               log_num_results: types.IntTensor,
                               num_digits: types.IntTensor,
                               dtype=None) -> types.IntTensor:
  r"""Returns all Sobol generating matrices.

  Args:
    dim: Positive scalar `Tensor` with rank 0 representing the event size of
      points which can be sampled from the resulting generating matrix.
    log_num_results: Positive scalar `Tensor` with rank 0 representing the
      base-2 logarithm of the maximum number of points which can be sampled from
      the resulting generating matrix.
    num_digits: Positive scalar `Tensor` with rank 0 representing the base-2
      precision of points which can be sampled from the resulting generating
      matrix.
    dtype: Optional `dtype`. The `dtype` of the output `Tensor` (either a signed
      or unsigned integer `dtype`).
      Default value: `None` which maps to `int32`.

  Returns:
    A scalar `Tensor` with shape `(dim, ceil(log2(num_results)))`.
  """
  global _INITIAL_DIRECTION_NUMBERS
  global _PRIMITIVE_POLYNOMIAL_COEFFICIENTS

  dtype = dtype or tf.int32

  indices = tf.cast(tf.range(0, log_num_results), dtype)
  dimensions = tf.range(0, dim)

  # shape: (?, ?)
  directions = tf.convert_to_tensor(
      _INITIAL_DIRECTION_NUMBERS, dtype=dtype, name='direction_numbers')
  # shape: (log_num_results, ?)
  directions = directions[:log_num_results]
  # shape: (log_num_results, dim)
  directions = tf.gather(directions, dimensions, axis=1)
  # shape: (dim, log_num_results)
  directions = tf.cast(tf.transpose(directions), dtype)

  # shape: (?,)
  polynomial = tf.convert_to_tensor(
      _PRIMITIVE_POLYNOMIAL_COEFFICIENTS,
      dtype=dtype,
      name='polynomial_coefficients')
  # shape: (1, dim)
  polynomial = tf.cast(
      tf.gather(polynomial, tf.expand_dims(dimensions, axis=1)), dtype)

  # shape: (1, dim)
  degree = tf.cast(
      tf.math.floor(utils.log2(tf.cast(polynomial, dtype=tf.float32))),
      dtype=dtype)

  def loop_predicate_fn(matrix_values, column):
    del matrix_values
    return column < log_num_results - 1

  def loop_body_fn(matrices, column):
    # Loop invariant: At the end of the iteration, all values from column 0 to
    # `column` have been calculated.

    # shape: (dim, log_num_results)
    column_values = tf.gather(matrices, [column], axis=1)

    # Columns whose index is smaller than the degree of the primitive polynomial
    # are obtained from direction numbers and thus should not be updated.
    # During a given iteration, only the next n columns (where n is the degree
    # of the primitive polynomial) should be updated.
    # shape: (dim, log_num_results)
    should_be_updated = tf.logical_and(
        tf.less_equal(tf.math.maximum(degree, column + 1), indices),
        tf.less_equal(indices, column + degree))

    # shape: (dim, log_num_results)
    updated_matrices = tf.bitwise.bitwise_xor(
        tf.where(
            tf.equal(indices, column + degree),
            tf.bitwise.right_shift(column_values, degree), matrices),
        utils.filter_tensor(column_values, polynomial,
                            column + degree - indices))

    # shape: (dim, log_num_results)
    returned_matrices = tf.where(should_be_updated, updated_matrices, matrices)

    return (returned_matrices, column + 1)

  # shape: (dim, log_num_results)
  initial_matrices = tf.bitwise.left_shift(
      directions,
      tf.cast(tf.expand_dims(num_digits - 1 - indices, axis=0), dtype))

  matrices, _ = tf.while_loop(
      loop_predicate_fn,
      loop_body_fn,
      loop_vars=(initial_matrices, tf.constant(0, dtype)),
      maximum_iterations=tf.cast(log_num_results, tf.int32) - 1)

  # shape: (dim, log_num_results)
  return matrices
