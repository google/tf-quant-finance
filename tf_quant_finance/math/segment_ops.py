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
"""Element wise ops acting on segments of arrays."""


import tensorflow.compat.v2 as tf

from tf_quant_finance.math import diff_ops


def segment_diff(x,
                 segment_ids,
                 order=1,
                 exclusive=False,
                 dtype=None,
                 name=None):
  """Computes difference of successive elements in a segment.

  For a complete description of segment_* ops see documentation of
  `tf.segment_max`. This op extends the `diff` functionality to segmented
  inputs.

  The behaviour of this op is the same as that of the op `diff` within each
  segment. The result is effectively a concatenation of the results of `diff`
  applied to each segment.

  #### Example

  ```python
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    # First order diff. Expected result: [3, -4, 6, 2, -22, 2, -9, 4, -3]
    dx1 = segment_diff(
        x, segment_ids=segments, order=1, exclusive=True)
    # Non-exclusive, second order diff.
    # Expected result: [2, 5, -1, 2, 8, 32, 10, -20, -7, 4, 8, 1]
    dx2 = segment_diff(
        x, segment_ids=segments, order=2, exclusive=False)
  ```

  Args:
    x: A rank 1 `Tensor` of any dtype for which arithmetic operations are
      permitted.
    segment_ids: A `Tensor`. Must be one of the following types: int32, int64. A
      1-D tensor whose size is equal to the size of `x`. Values should be sorted
      and can be repeated.
    order: Positive Python int. The order of the difference to compute. `order =
      1` corresponds to the difference between successive elements.
      Default value: 1
    exclusive: Python bool. See description above.
      Default value: False
    dtype: Optional `tf.Dtype`. If supplied, the dtype for `x` to use when
      converting to `Tensor`.
      Default value: None which maps to the default dtype inferred by TF.
    name: Python `str` name prefixed to Ops created by this class.
      Default value: None which is mapped to the default name 'segment_diff'.

  Returns:
    diffs: A `Tensor` of the same dtype as `x`. Assuming that each segment is
      of length greater than or equal to order, if `exclusive` is True,
      then the size is `n-order*k` where `n` is the size of x,
      `k` is the number of different segment ids supplied if `segment_ids` is
      not None or 1 if `segment_ids` is None. If any of the segments is of
      length less than the order, then the size is:
      `n-sum(min(order, length(segment_j)), j)` where the sum is over segments.
      If `exclusive` is False, then the size is `n`.
  """
  with tf.compat.v1.name_scope(name, default_name='segment_diff', values=[x]):
    x = tf.convert_to_tensor(x, dtype=dtype)
    raw_diffs = diff_ops.diff(x, order=order, exclusive=exclusive)
    if segment_ids is None:
      return raw_diffs
    # If segment ids are supplied, raw_diffs are incorrect at locations:
    # p, p+1, ... min(p+order-1, m_p-1) where p is the index of the first
    # element of a segment other than the very first segment (which is
    # already correct). m_p is the segment length.
    # Find positions where the segments begin.
    has_segment_changed = tf.concat(
        [[False], tf.not_equal(segment_ids[1:] - segment_ids[:-1], 0)], axis=0)
    # Shape [k, 1]
    segment_start_index = tf.cast(tf.where(has_segment_changed), dtype=tf.int32)
    segment_end_index = tf.concat(
        [tf.reshape(segment_start_index, [-1])[1:], [tf.size(segment_ids)]],
        axis=0)
    segment_end_index = tf.reshape(segment_end_index, [-1, 1])
    # The indices of locations that need to be adjusted. This needs to be
    # constructed in steps. First we generate p, p+1, ... p+order-1.
    # Shape [num_segments-1, order]
    fix_indices = (
        segment_start_index + tf.range(order, dtype=segment_start_index.dtype))
    in_bounds = tf.where(fix_indices < segment_end_index)
    # Keep only the ones in bounds.
    fix_indices = tf.reshape(tf.gather_nd(fix_indices, in_bounds), [-1, 1])

    needs_fix = tf.scatter_nd(
        fix_indices,
        # Unfortunately, scatter_nd doesn't support bool on GPUs so we need to
        # do ints here and then convert to bool.
        tf.reshape(tf.ones_like(fix_indices, dtype=tf.int32), [-1]),
        shape=tf.shape(x))
    # If exclusive is False, then needs_fix means we need to replace the values
    # in raw_diffs at those locations with the values in x.
    needs_fix = tf.cast(needs_fix, dtype=tf.bool)
    if not exclusive:
      return tf.where(needs_fix, x, raw_diffs)

    # If exclusive is True, we have to be more careful. The raw_diffs
    # computation has removed the first 'order' elements. After removing the
    # corresponding elements from needs_fix, we use it to remove the elements
    # from raw_diffs.
    return tf.boolean_mask(raw_diffs, tf.logical_not(needs_fix[order:]))


def segment_cumsum(x, segment_ids, exclusive=False, dtype=None, name=None):
  """Computes cumulative sum of elements in a segment.

  For a complete description of segment_* ops see documentation of
  `tf.segment_sum`. This op extends the `tf.math.cumsum` functionality to
  segmented inputs.

  The behaviour of this op is the same as that of the op `tf.math.cumsum` within
  each segment. The result is effectively a concatenation of the results of
  `tf.math.cumsum` applied to each segment with the same interpretation for the
  argument `exclusive`.

  #### Example

  ```python
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    # Inclusive cumulative sum.
    # Expected result: [2, 7, 8, 15, 24, 32, 42, 54, 57, 4, 12, 17]
    cumsum1 = segment_cumsum(
        x, segment_ids=segments, exclusive=False)
    # Exclusive cumsum.
    # Expected result: [0, 2, 7, 8, 15, 0, 32, 42, 54, 0, 4, 12]
    cumsum2 = segment_cumsum(
        x, segment_ids=segments, exclusive=True)
  ```

  Args:
    x: A rank 1 `Tensor` of any dtype for which arithmetic operations are
      permitted.
    segment_ids: A `Tensor`. Must be one of the following types: int32, int64. A
      1-D tensor whose size is equal to the size of `x`. Values should be sorted
      and can be repeated. Values must range from `0` to `num segments - 1`.
    exclusive: Python bool. See description above.
      Default value: False
    dtype: Optional `tf.Dtype`. If supplied, the dtype for `x` to use when
      converting to `Tensor`.
      Default value: None which maps to the default dtype inferred by TF.
    name: Python `str` name prefixed to Ops created by this class.
      Default value: None which is mapped to the default name 'segment_cumsum'.

  Returns:
    cumsums: A `Tensor` of the same dtype as `x`. Assuming that each segment is
      of length greater than or equal to order, if `exclusive` is True,
      then the size is `n-order*k` where `n` is the size of x,
      `k` is the number of different segment ids supplied if `segment_ids` is
      not None or 1 if `segment_ids` is None. If any of the segments is of
      length less than the order, then the size is:
      `n-sum(min(order, length(segment_j)), j)` where the sum is over segments.
      If `exclusive` is False, then the size is `n`.
  """
  with tf.compat.v1.name_scope(name, default_name='segment_cumsum', values=[x]):
    x = tf.convert_to_tensor(x, dtype=dtype)
    raw_cumsum = tf.math.cumsum(x, exclusive=exclusive)
    if segment_ids is None:
      return raw_cumsum
    # It is quite tedious to do a vectorized version without a while loop so
    # we skip that for now.
    # TODO(b/137940928): Replace these ops with more efficient C++ kernels.
    def scanner(accumulators, args):
      cumsum, prev_segment, prev_value = accumulators
      value, segment = args
      if exclusive:
        initial_value, inc_value = tf.zeros_like(value), cumsum + prev_value
      else:
        initial_value, inc_value = value, cumsum + value
      next_cumsum = tf.where(
          tf.equal(prev_segment, segment), inc_value, initial_value)
      return next_cumsum, segment, value

    return tf.scan(
        scanner, (x, segment_ids),
        initializer=(tf.zeros_like(x[0]), tf.zeros_like(segment_ids[0]) - 1,
                     tf.zeros_like(x[0])))[0]


__all__ = ['segment_cumsum', 'segment_diff']
