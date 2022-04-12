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
"""Helper functions for padding multiple tensors."""

import tensorflow.compat.v2 as tf
from tf_quant_finance.datetime import date_tensor


def pad_tensors(tensors, pad_values=None, dtype=None, name=None):
  """Pads the innermost dimension of `Tensor`s to a common shape.

  Given a list of `Tensor`s of the same `dtype` and with shapes
  `batch_shape_i + [n_i]`, pads the innermost dimension of each tensor to
  `batch_shape_i + [max(n_i)]`. For each tensor `t`, the padding is done with
  values `t[..., -1]`.

  ### Example. Pad with the terminal value
  ```python
  x = [[1, 2, 3, 9], [2, 3, 5, 2]]
  y = [4, 5, 8]
  pad_tensors([x, y])
  # Expected: [array([[1, 2, 3, 9], [2, 3, 5, 2]], array([4, 5, 8, 8])]
  ```

  ### Example. Pad with user-supplied values
  ```python
  x = [[1, 2, 3, 9], [2, 3, 5, 2]]
  y = [4, 5, 8]
  pad_tensors([x, y], pad_values=10)
  # Expected: [array([[1, 2, 3, 9], [2, 3, 5, 2]], array([4, 5, 8, 10])]
  ```

  Args:
    tensors: A list of tensors of the same `dtype` and shapes
      `batch_shape_i + [n_i]`.
    pad_values: An optional scalar `Tensor` or a list of `Tensor`s of shapes
      broadcastable with the corresponding `batch_shape_i` and the same
      `dtype` as `pad_values`. Corresponds to the padded values used for
      `tensors`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this class.
      Default value: `None` which maps to the default name `pad_tensors`.
  Returns:
    A list of `Tensor`s of shape `batch_shape_i + [max(n_i)]`.

  Raises:
    ValueError: If input is not an instance of a list or a tuple.
  """
  if not isinstance(tensors, (tuple, list)):
    raise ValueError(
        f"`tensors` should be a list or a tuple but have type {type(tensors)}")
  if not tensors:
    return []
  name = name or "pad_tensors"
  with tf.name_scope(name):
    t0 = tf.convert_to_tensor(tensors[0], dtype=dtype)
    dtype = dtype or t0.dtype
    tensors = [t0] + [tf.convert_to_tensor(t, dtype=dtype) for t in tensors[1:]]
    max_size = tf.reduce_max([tf.shape(t)[-1] for t in tensors])
    padded_tensors = []
    pad_values = _prepare_pad_values(pad_values, tensors, dtype)
    for pad_value, t in zip(pad_values, tensors):
      paddings = (
          (t.shape.rank - 1) * [[0, 0]] + [[0, max_size - tf.shape(t)[-1]]])
      # Padded value has to be a constant
      constant_values = tf.reduce_min(t) - 1
      pad_t = tf.pad(t, paddings, mode="CONSTANT",
                     constant_values=constant_values)
      pad_t = tf.where(pad_t > constant_values,
                       pad_t, pad_value)
      padded_tensors.append(pad_t)
  return padded_tensors


def pad_date_tensors(date_tensors, name=None):
  """Pads the innermost dimension of `DateTensor`s to a common shape.

  Given a list of `DateTensor`s of shapes `batch_shape_i + [n_i]`, pads the
  innermost dimension of each corresponding ordinal tensor to
  `batch_shape_i + [max(n_i)]`. For each ordinal tensor `t`, the padding is done
  with values `t[..., -1]`.

  ### Example
  ```python
  x = [(2020, 1, 1), (2021, 2, 2)]
  y = [(2019, 5, 5), (2028, 10, 21), (2028, 11, 10)]
  pad_date_tensors([x, y])
  # Expected: [DateTensor: [(2020, 1, 1), (2021, 2, 2), (2021, 2, 2)],
  #            DateTensor: [(2019, 5, 5), (2028, 10, 21), (2028, 11, 10)]]
  ```

  Args:
    date_tensors: a list of tensors of shapes `batch_shape_i + [n_i]`.
    name: Python string. The name to give to the ops created by this class.
      Default value: `None` which maps to the default name `pad_date_tensors`.
  Returns:
    A list of `DateTensor`s of shape `batch_shape_i + [max(n_i)]`.
  """
  name = name or "pad_date_tensors"
  with tf.name_scope(name):
    ordinals = [date_tensor.convert_to_date_tensor(t).ordinal()
                for t in date_tensors]
    padded_tensors = pad_tensors(ordinals)
    return [date_tensor.from_ordinals(p) for p in padded_tensors]


def _prepare_pad_values(pad_values, tensors, dtype):
  """Prepares a list of paddings."""
  if pad_values is not None:
    if isinstance(pad_values, (list, tuple)):
      # Shapes compatible with [batch_shape_i, 1]
      pad_values = [tf.expand_dims(
          tf.convert_to_tensor(pad_value, dtype=dtype, name="pad_values"),
          axis=-1) for pad_value in pad_values]
    else:
      pad_values = len(tensors) * [tf.constant(pad_values, dtype=dtype,
                                               name="pad_values")]
  else:
    # Use final value for padding if `pad_values` are not supplied
    pad_values = [t[..., -1:] for t in tensors]
  return pad_values

__all__ = [
    "pad_tensors",
    "pad_date_tensors"
]
