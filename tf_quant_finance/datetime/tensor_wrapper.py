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
"""Base class for Tensor wrappers."""

import abc
import tensorflow.compat.v2 as tf


class TensorWrapper(metaclass=abc.ABCMeta):
  """Base class for Tensor wrappers.

  Implements ops that manipulate the backing tensors of Tensor wrappers
  (e.g. DateTensor, PeriodTensor). These ops are mostly about reshaping the
  backing tensors, such as tf.reshape, tf.expand_dims, tf.stack, etc. Also
  includes indexing and slicing.

  Inheritors must implement _apply_op(self, op_fn) and provide a static method
  _apply_sequence_to_tensor_op(op_fn, tensors). For example:

  ```python
  class MyWrapper(TensorWrapper):
    def __init__(self, backing_tensor):
       self._backing_tensor = backing_tensor

    def _apply_op(self, op_fn):
      new_backing_tensor = op_fn(self._backing_tensor)
      return MyWrapper(new_backing_tensor)

    @staticmethod
    def _apply_sequence_to_tensor_op(op_fn, tensors):
      new_backing_tensor = op_fn([t._backing_tensor for t in tensors])
      return MyWrapper(new_backing_tensor)
  ```

  Then 'MyWrapper` can be used as follows:

  ```python
  m1 = MyWrapper(tf.constant([[1, 2, 3], [4, 5, 6]]))
  m2 = MyWrapper(...)
  m3 = m1[0, 1:-1]
  m4 = m1.expand_dims(axis=-1)
  m5 = MyWrapper.concat((m1, m2), axis=-1)
  # etc.
  ```
  """

  @classmethod
  def concat(cls, tensor_wrappers, axis):
    """See tf.concat."""
    cls._validate_tensor_types(tensor_wrappers, "concat")
    return cls._apply_sequence_to_tensor_op(
        lambda ts: tf.concat(ts, axis), tensor_wrappers)

  @classmethod
  def stack(cls, tensor_wrappers, axis=0):
    """See tf.stack."""
    cls._validate_tensor_types(tensor_wrappers, "stack")
    return cls._apply_sequence_to_tensor_op(
        lambda ts: tf.stack(ts, axis), tensor_wrappers)

  @classmethod
  def where(cls, condition, tensor_wrapper_1, tensor_wrapper_2):
    """See tf.where. Only three-argument version is supported here."""
    tensor_wrappers = [tensor_wrapper_1, tensor_wrapper_2]
    cls._validate_tensor_types(tensor_wrappers, "where")
    return cls._apply_sequence_to_tensor_op(
        lambda ts: tf.compat.v2.where(condition, ts[0], ts[1]), tensor_wrappers)

  @classmethod
  def _validate_tensor_types(cls, tensor_wrappers, function_name):
    for tensor in tensor_wrappers:
      if not isinstance(tensor, cls):
        raise ValueError("{}.{} cannot be applied to {}".format(
            cls.__name__, function_name,
            type(tensor).__name__))

  def expand_dims(self, axis):
    """See tf.expand_dims."""
    return self._apply_op(lambda t: tf.expand_dims(t, axis))

  def reshape(self, shape):
    """See tf.reshape."""
    return self._apply_op(lambda t: tf.reshape(t, shape))

  def identity(self):
    """See tf.identity."""
    return self._apply_op(tf.identity)

  def broadcast_to(self, shape):
    """See tf.broadcast_to."""
    return self._apply_op(lambda t: tf.broadcast_to(t, shape))

  def transpose(self, perm=None):
    """See tf.transpose."""
    return self._apply_op(lambda t: tf.transpose(t, perm))

  def squeeze(self, axis=None):
    """See tf.squeeze."""
    return self._apply_op(lambda t: tf.squeeze(t, axis))

  def boolean_mask(self, mask, axis=None):
    """See tf.boolean_mask."""
    return self._apply_op(lambda t: tf.boolean_mask(t, mask, axis=axis))

  def __getitem__(self, key):
    """Implements indexing."""
    return self._apply_op(lambda t: t.__getitem__(key))

  def __getslice__(self, *args):
    """Implements slicing."""
    return self._apply_op(lambda t: t.__getslice__(*args))

  @classmethod
  @abc.abstractmethod
  def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
    """Applies given sequence-to-tensor op.

    This method is used for implementing ops that take a sequence of tensors and
    return a new tensor, such as tf.concat and tf.stack. Implementing wrappers
    should apply `op_fn` to the backing tensor(s) and return an new wrapper
    instance with the combined backing tensor.

    Args:
     op_fn: Callable that applies sequence-to-tensor op to the given sequence
       of Tensors. E.g. applies tf.concat.
     tensor_wrappers: a sequence of tensor wrappers to be transformed. All
       elements have the type of the implementing TensorWrapper class.

    Returns:
      A TensorWrapper instance with combined backing tensor(s).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _apply_op(self, op_fn):
    """Applies given tensor-to-tensor op.

    This method is used for implementing ops that take a tensor and return a new
    tensor, such as tf.expand_dims or tf.transpose. Implementing wrappers
    should apply `op_fn` to the backing tensor(s) and return an new wrapper
    instance with the updated backing tensor.

    Args:
       op_fn: Callable that applies tensor-to-tensor op to the given Tensor.
        E.g. applies tf.expand_dims.

    Returns:
      A TensorWrapper instance with updated backing tensor(s).
    """
    raise NotImplementedError()
