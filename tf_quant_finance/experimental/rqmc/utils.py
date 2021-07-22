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
"""Utility functions."""

from typing import Union

import tensorflow.compat.v2 as tf

from tf_quant_finance import types


def exp2(value: types.IntTensor) -> types.IntTensor:
  r"""Returns the point-wise base-2 exponentiation of a given `Tensor`.

  Args:
    value: Positive scalar `Tensor` of integers.

  Returns:
    `Tensor` with the same `shape` and `dtype` as `value` equal to `1 << value`.
  """
  shape = get_shape(value)
  dtype = value.dtype

  max_allowed_value = tf.constant(8 * dtype.size, shape=shape, dtype=dtype)
  if not dtype.is_unsigned:
    max_allowed_value -= 1

  return tf.where(
      tf.greater_equal(value, max_allowed_value),
      tf.constant(dtype.max, shape=shape, dtype=dtype),
      tf.bitwise.left_shift(tf.constant(1, shape=shape, dtype=dtype), value))


def log2(value: types.FloatTensor) -> types.FloatTensor:
  r"""Returns the point-wise base-2 logarithm a given `Tensor`.

  Args:
    value: Positive scalar `Tensor` of real values.

  Returns:
    `Tensor` with the same `shape` and `dtype` as `value` equal to `ln(value) /
    ln(2)`.
  """
  return tf.math.log(value) / tf.math.log(tf.constant(2, dtype=value.dtype))


def get_shape(
    value: Union[types.FloatTensor, types.IntTensor]) -> types.IntTensor:
  r"""Returns the `shape` of a given `Tensor`.

  Args:
    value: Scalar `Tensor of integers or real values.

  Returns:
    `Tensor` of integers with rank 1.
  """
  result = value.shape
  return tf.shape(value) if None in result.as_list() else result


def tent_transform(value: types.FloatTensor) -> types.FloatTensor:
  r"""Returns the tent transform of a given `Tensor`.

  Args:
    value: Scalar `Tensor` of real values in the `[0, 1)` range.

  Returns:
    `Tensor` with the same `shape` as `value` equal to `2 ** value` if `value`
    is less than `0.5` or `2 * (1 - value)` otherwise.
  """

  return tf.where(value < 0.5, 2 * value, 2 * (1 - value))


def filter_tensor(value: types.IntTensor, bit_mask: types.IntTensor,
                  bit_index: types.IntTensor) -> types.IntTensor:
  r"""Filters an input `Tensor` based on bit sets in a mask `Tensor`.

  Args:
    value: Scalar `Tensor` of integers.
    bit_mask: Positive scalar `Tensor` of integers with the same `shape` and
      `dtype` as `value`.
    bit_index: Positive scalar `Tensor` of integers with the same `shape` and
      `dtype` as `value`.

  Returns:
    `Tensor` with the same `shape` as `value` equal to `value` if the
    `bit_index`th bit in `LSB 0` order is set in `bit_mask`, or `zero`
    otherwise i.e.: `value * (1 & (bit_mask >>> bit_index))`
  """

  # Whether the `bit_index`th bit in LSB 0 order is set in `bit_filter`.
  is_bit_set = tf.equal(
      tf.cast(1, value.dtype),
      tf.bitwise.bitwise_and(
          tf.bitwise.right_shift(bit_mask, bit_index), tf.cast(1, value.dtype)))

  return tf.where(is_bit_set, value, 0)
