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
# Lint as: python2, python3
"""Piecewise utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def find_interval_index(query_xs,
                        interval_lower_xs,
                        last_interval_is_closed=False,
                        dtype=None,
                        name=None):
  """Function to find the index of the interval where query points lies.

  Given a list of adjacent half-open intervals [x_0, x_1), [x_1, x_2), ...,
  [x_{n-1}, x_n), [x_n, inf), described by a list [x_0, x_1, ..., x_{n-1}, x_n].
  Return the index where the input query points lie. If x >= x_n, n is returned,
  and if x < x_0, -1 is returned. If `last_interval_is_closed` is set to `True`,
  the last interval [x_{n-1}, x_n] is interpreted as closed (including x_n).

  ### Example

  ```python
  interval_lower_xs = [0.25, 0.5, 1.0, 2.0, 3.0]
  query_xs = [0.25, 3.0, 5.0, 0.0, 0.5, 0.8]
  result = find_interval_index(query_xs, interval_lower_xs)
  # result == [0, 4, 4, -1, 1, 1]
  ```

  Args:
    query_xs: Rank 1 real `Tensor` of any size, the list of x coordinates for
      which the interval index is to be found. The values must be strictly
      increasing.
    interval_lower_xs: Rank 1 `Tensor` of the same shape and dtype as
      `query_xs`. The values x_0, ..., x_n that define the interval starts.
    last_interval_is_closed: If set to `True`, the last interval is interpreted
      as closed.
    dtype: Optional `tf.Dtype`. If supplied, the dtype for `query_xs` and
      `interval_lower_xs`.
      Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: Optional name of the operation.

  Returns:
    A tensor that matches the shape of `query_xs` with dtype=int32 containing
    the indices of the intervals containing query points. `-1` means the query
    point lies before all intervals and `n-1` means that the point lies in the
    last half-open interval (if `last_interval_is_closed` is `False`) or that
    the point lies to the right of all intervals (if `last_interval_is_closed`
    is `True`).
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='find_interval_index',
      values=[query_xs, interval_lower_xs, last_interval_is_closed]):
    # TODO(b/138988951): add ability to validate that intervals are increasing.
    # TODO(b/138988951): validate that if last_interval_is_closed, input size
    # must be > 1.
    query_xs = tf.convert_to_tensor(query_xs, dtype=dtype)
    interval_lower_xs = tf.convert_to_tensor(interval_lower_xs, dtype=dtype)

    # Result assuming that last interval is half-open.
    indices = tf.searchsorted(interval_lower_xs, query_xs, side='right') - 1

    # Handling the branch if the last interval is closed.
    last_index = tf.shape(interval_lower_xs)[-1] - 1
    last_x = tf.gather(interval_lower_xs, [last_index], axis=-1)
    # should_cap is a tensor true where a cell is true iff indices is the last
    # index at that cell and the query x <= the right boundary of the last
    # interval.
    should_cap = tf.logical_and(
        tf.equal(indices, last_index), tf.less_equal(query_xs, last_x))

    # cap to last_index if the query x is not in the last interval, otherwise,
    # cap to last_index - 1.
    caps = last_index - tf.cast(should_cap, dtype=tf.dtypes.int32)

    return tf.where(last_interval_is_closed, tf.minimum(indices, caps), indices)
