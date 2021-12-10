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
"""Utilities for shape manipulation."""

from typing import Tuple, Sequence, Union, Optional

import tensorflow.compat.v2 as tf

from tf_quant_finance import types

__all__ = [
    'get_shape',
    'broadcast_common_batch_shape',
    'broadcast_tensors',
    'common_shape'
]


def get_shape(
    x: tf.Tensor,
    name: Optional[str] = None) -> Union[tf.TensorShape, types.IntTensor]:
  """Returns static shape of `x` if it is fully defined, or dynamic, otherwise.

  ####Example
  ```python
  import tensorflow as tf
  import tf_quant_finance as tff

  x = tf.zeros([5, 2])
  prefer_static_shape(x)
  # Expected: [5, 2]

  Args:
    x: A tensor of any shape and `dtype`
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `get_shape`.

  Returns:
    A shape of `x` which a list, if the shape is fully defined, or a `Tensor`
    for dynamically shaped `x`.
  """
  name = 'get_shape' if name is None else name
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x)
    is_fully_defined = x.shape.is_fully_defined()
    if is_fully_defined:
      return x.shape
    return tf.shape(x)


def common_shape(
    *args: Sequence[tf.Tensor],
    name: Optional[str] = None) -> Union[tf.TensorShape, tf.Tensor]:
  """Returns common shape for a sequence of Tensors.

  The common shape is the smallest-rank shape to which all tensors are
  broadcastable.

  #### Example
  ```python
  import tensorflow as tf
  import tf_quant_finance as tff

  args = [tf.ones([1, 2], dtype=tf.float64), tf.constant([[True], [False]])]
  tff.utils.common_shape(*args)
  # Expected: [2, 2]
  ```

  Args:
    *args: A sequence of `Tensor`s of compatible shapes and any `dtype`s.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `broadcast_tensor_shapes`.

  Returns:
    A common shape for the input `Tensor`s, which an instance of TensorShape,
    if the input shapes are fully defined, or a `Tensor` for dynamically shaped
    inputs.

  Raises:
    ValueError: If inputs are of incompatible shapes.
  """
  name = 'common_shape' if name is None else name
  with tf.name_scope(name):
    # Flag to decide whether input Tensors have fully defined shapes
    is_fully_defined = True
    if args:
      for arg in args:
        arg = tf.convert_to_tensor(arg)
        is_fully_defined &= arg.shape.is_fully_defined()
      if is_fully_defined:
        output_shape = args[0].shape
        for arg in args[1:]:
          try:
            output_shape = tf.broadcast_static_shape(output_shape, arg.shape)
          except ValueError:
            raise ValueError(f'Shapes of {args} are incompatible')
        return output_shape
      output_shape = tf.shape(args[0])
      for arg in args[1:]:
        output_shape = tf.broadcast_dynamic_shape(output_shape, tf.shape(arg))
      return output_shape


def broadcast_tensors(
    *args: Sequence[tf.Tensor],
    name: Optional[str] = None) -> Tuple[tf.Tensor]:
  """Broadcasts arguments to the common shape.

  #### Example
  ```python
  import tensorflow as tf
  import tf_quant_finance as tff

  args = [tf.ones([1, 2], dtype=tf.float64), tf.constant([[True], [False]])]
  tff.utils.broadcast_tensor_shapes(*args)
  # Expected: (array([[1., 1.], [1., 1.]]),
  #            array([[True, True], [False, False]])
  ```

  Args:
    *args: A sequence of `Tensor`s of compatible shapes and any `dtype`s.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `broadcast_tensor_shapes`.

  Returns:
    A tuple of broadcasted `Tensor`s. Each `Tensor` has the same `dtype` as the
    corresponding input `Tensor`.

  Raises:
    ValueError: If inputs are of incompatible shapes.
  """
  name = 'broadcast_tensors' if name is None else name
  with tf.name_scope(name):
    output_shape = common_shape(*args)
    return tuple(tf.broadcast_to(arg, output_shape)
                 for arg in args)


def broadcast_common_batch_shape(
    *args: Sequence[tf.Tensor],
    event_ranks: Optional[int] = None,
    name: Optional[str] = None) -> Tuple[tf.Tensor]:
  """Broadcasts argument batch shapes to the common shape.

  Each input `Tensor` is assumed to be of shape `batch_shape_i + event_shape_i`.
  The function finds a common `batch_shape` and broadcasts each `Tensor` to
  `batch_shape + event_shape_i`. The common batch shape is the minimal shape
  such that all `batch_shape_i` can broadcast to it.

  #### Example 1. Batch shape is all dimensions but the last one
  ```python
  import tensorflow as tf
  import tf_quant_finance as tff

  # Two Tensors of shapes [2, 3] and [2]. The batch shape of the 1st Tensor is
  # [2] and for the second is []. The common batch shape is [2]
  args = [tf.ones([2, 3], dtype=tf.float64), tf.constant([True, False])]
  tff.utils.broadcast_common_batch_shape(*args)
  # Expected: (array([[1., 1., 1.], [1., 1., 1.]]),
  #            array([[True, True], [False, False]])
  ```

  #### Example 2. Specify ranks of event shapes
  ```python
  import tensorflow as tf
  import tf_quant_finance as tff

  args = [tf.ones([2, 3], dtype=tf.float64), tf.constant([True, False])]
  tff.utils.broadcast_common_batch_shape(*args,
                                         event_ranks)
  # Expected: (array([[1., 1., 1.], [1., 1., 1.]]),
  #            array([[True, True], [False, False]])
  ```

  Args:
    *args: A sequence of `Tensor`s of compatible shapes and any `dtype`s.
    event_ranks: A sequence of integers of the same length as `args` specifying
      ranks of `event_shape` for each input `Tensor`.
      Default value: `None` which means that all dimensions but the last one
      are treated as batch dimension.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `broadcast_tensor_shapes`.

  Returns:
    A tuple of broadcasted `Tensor`s. Each `Tensor` has the same `dtype` as the
    corresponding input `Tensor`.

  Raises:
    ValueError:
      (a) If `event_ranks` is supplied and is of different from `args` length.
      (b) If inputs are of incompatible shapes.
  """
  name = 'broadcast_common_batch_shape' if name is None else name
  with tf.name_scope(name):
    if event_ranks is None:
      event_ranks = [1] * len(args)
    if len(event_ranks) != len(args):
      raise ValueError(
          '`args` and `event_dims` should be of the same length but are {0} '
          'and {1} elements, respectively'.format(len(event_ranks), len(args)))
    dummies = [tf.zeros(get_shape(arg)[:-d])
               for arg, d in zip(args, event_ranks)]
    common_batch_shape = common_shape(*dummies)
    return tuple(tf.broadcast_to(x, tf.concat(
        [common_batch_shape, get_shape(x)[-d:]], axis=0))
                 for x, d in zip(args, event_ranks))
