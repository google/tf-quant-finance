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
"""Array difference ops."""


import tensorflow.compat.v2 as tf


# TODO(b/136354274): Move this function to a math library and provide a more
# efficient C++ kernel.
def diff(x, order=1, exclusive=False, dtype=None, name=None):
  """Computes the difference between elements of an array at a regular interval.

  If exclusive is True, then computes

  ```
    result[..., i] = x[..., i+order] - x[..., i] for i < size(x) - order

  ```

  This is the same as doing `x[..., order:] - x[..., :-order]`. Note that in
  this case the result `Tensor` is smaller in size than the input `Tensor`.

  If exclusive is False, then computes:

  ```
    result[..., i] = x[..., i] - x[..., i-order] for i >= order
    result[..., i] = x[..., i]  for 0 <= i < order

  ```

  #### Example

  ```python
    x = tf.constant([1, 2, 3, 4, 5])
    dx = diff(x, order=1, exclusive=False)  # Returns [1, 1, 1, 1, 1]
    dx1 = diff(x, order=1, exclusive=True)  # Returns [1, 1, 1, 1]
    dx2 = diff(x, order=2, exclusive=False)  # Returns [1, 2, 2, 2, 2]
  ```

  Args:
    x: A `Tensor` of shape `batch_shape + [n]` and of any dtype for which
      arithmetic operations are permitted.
    order: Positive Python int. The order of the difference to compute. `order =
      1` corresponds to the difference between successive elements.
      Default value: 1
    exclusive: Python bool. See description above.
      Default value: False
    dtype: Optional `tf.DType`. If supplied, the dtype for `x` to use when
      converting to `Tensor`.
      Default value: None which maps to the default dtype inferred by TF.
    name: Python `str` name prefixed to Ops created by this class.
      Default value: None which is mapped to the default name 'diff'.

  Returns:
    diffs: A `Tensor` of the same dtype as `x`. If `exclusive` is True,
      then the shape is `batch_shape + [n-min(order, n)]`, otherwise it is
      `batch_shape + [n]`. The final dimension of which contains the differences
      of the requested order.
  """
  with tf.name_scope(name or 'diff'):
    x = tf.convert_to_tensor(x, dtype=dtype)
    exclusive_diff = x[..., order:] - x[..., :-order]
    if exclusive:
      return exclusive_diff

    return tf.concat([x[..., :order], exclusive_diff], axis=0)
