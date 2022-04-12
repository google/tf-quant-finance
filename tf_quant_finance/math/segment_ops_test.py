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
"""Tests for math.segment_ops.py."""


import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math import segment_ops


class SegmentOpsTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_segment_diffs_no_segment_exclusive(self):
    x = tf.constant([11, 13, 17, 19, 23])
    dx1 = self.evaluate(
        segment_ops.segment_diff(x, segment_ids=None, order=1, exclusive=True))
    np.testing.assert_array_equal(dx1, [2, 4, 2, 4])
    dx2 = self.evaluate(
        segment_ops.segment_diff(x, segment_ids=None, order=2, exclusive=True))
    np.testing.assert_array_equal(dx2, [6, 6, 6])

  @test_util.run_in_graph_and_eager_modes
  def test_segment_diffs_no_segment_inclusive(self):
    x = tf.constant([11, 13, 17, 19, 23])
    dx1 = self.evaluate(
        segment_ops.segment_diff(x, segment_ids=None, order=1, exclusive=False))
    np.testing.assert_array_equal(dx1, [11, 2, 4, 2, 4])

    dx2 = self.evaluate(
        segment_ops.segment_diff(x, segment_ids=None, order=2, exclusive=False))
    np.testing.assert_array_equal(dx2, [11, 13, 6, 6, 6])

  @test_util.run_in_graph_and_eager_modes
  def test_segment_diffs_segment_exclusive(self):
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    dx1 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=1, exclusive=True))
    np.testing.assert_array_equal(dx1, ([3, -4, 6, 2] + [-22, 2, -9] + [4, -3]))

    dx2 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=2, exclusive=True))
    np.testing.assert_array_equal(dx2, ([-1, 2, 8] + [-20, -7] + [1]))

  @test_util.run_in_graph_and_eager_modes
  def test_segment_diffs_segment_inclusive(self):
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    dx1 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=1, exclusive=False))
    np.testing.assert_array_equal(
        dx1, ([2, 3, -4, 6, 2] + [32, -22, 2, -9] + [4, 4, -3]))

    dx2 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=2, exclusive=False))
    np.testing.assert_array_equal(
        dx2, ([2, 5, -1, 2, 8] + [32, 10, -20, -7] + [4, 8, 1]))

  @test_util.run_in_graph_and_eager_modes
  def test_segment_diffs_large_order(self):
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    dx1 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=4, exclusive=False))
    np.testing.assert_array_equal(
        dx1, ([2, 5, 1, 7, 7] + [32, 10, 12, 3] + [4, 8, 5]))

    dx2 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=4, exclusive=True))
    np.testing.assert_array_equal(
        dx2, ([7] + [] + []))  # The empty arrays are for the segments 1 and 2.

  @test_util.run_in_graph_and_eager_modes
  def test_segment_diffs_small_segment(self):
    x = tf.constant([2, 5, 1, 7] + [9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0] + [1] + [2, 2, 2, 2] + [3, 3, 3])
    dx1 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=2, exclusive=False))
    np.testing.assert_array_equal(
        dx1, ([2, 5, -1, 2] + [9] + [32, 10, -20, -7] + [4, 8, 1]))

    dx2 = self.evaluate(
        segment_ops.segment_diff(
            x, segment_ids=segments, order=2, exclusive=True))
    np.testing.assert_array_equal(dx2, ([-1, 2] + [] + [-20, -7] + [1]))

  @test_util.run_in_graph_and_eager_modes
  def test_segment_cumsum_no_segment_exclusive(self):
    x = tf.constant([-11, 13, 17, 19, 23])
    cx = self.evaluate(
        segment_ops.segment_cumsum(x, segment_ids=None, exclusive=True))
    np.testing.assert_array_equal(cx, [0, -11, 2, 19, 38])

  @test_util.run_in_graph_and_eager_modes
  def test_segment_cumsum_no_segment_inclusive(self):
    x = tf.constant([-11, 13, 17, 19, 23])
    cx = self.evaluate(
        segment_ops.segment_cumsum(x, segment_ids=None, exclusive=False))
    np.testing.assert_array_equal(cx, [-11, 2, 19, 38, 61])

  @test_util.run_in_graph_and_eager_modes
  def test_segment_cumsum_segment_exclusive(self):
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    cx = self.evaluate(
        segment_ops.segment_cumsum(x, segment_ids=segments, exclusive=True))
    np.testing.assert_array_equal(cx, [0, 2, 7, 8, 15, 0, 32, 42, 54, 0, 4, 12])

  @test_util.run_in_graph_and_eager_modes
  def test_segment_cumsum_segment_inclusive(self):
    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])
    dx1 = self.evaluate(
        segment_ops.segment_cumsum(x, segment_ids=segments, exclusive=False))
    np.testing.assert_array_equal(dx1,
                                  [2, 7, 8, 15, 24, 32, 42, 54, 57, 4, 12, 17])

  @test_util.run_in_graph_and_eager_modes
  def test_segment_cumsum_small_segment(self):
    x = tf.constant([2, 5, 1, 7] + [9] + [32, 10, 12, 3] + [4, 8, 5])
    segments = tf.constant([0, 0, 0, 0] + [1] + [2, 2, 2, 2] + [3, 3, 3])
    cx1 = self.evaluate(
        segment_ops.segment_cumsum(x, segment_ids=segments, exclusive=False))
    np.testing.assert_array_equal(
        cx1, ([2, 7, 8, 15] + [9] + [32, 42, 54, 57] + [4, 12, 17]))

    cx2 = self.evaluate(
        segment_ops.segment_cumsum(x, segment_ids=segments, exclusive=True))
    np.testing.assert_array_equal(
        cx2, ([0, 2, 7, 8] + [0] + [0, 32, 42, 54] + [0, 4, 12]))


if __name__ == '__main__':
  tf.test.main()
