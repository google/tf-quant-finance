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
"""Tests for Quasi Monte-Carlo utils."""

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

utils = tff.math.qmc.utils


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase):

  def test_exp2_without_overflow(self):
    test_cases = [
        tf.uint8, tf.int8, tf.uint16, tf.int16, tf.uint32, tf.int32, tf.uint64,
        tf.int64
    ]

    for dtype in test_cases:
      expected = tf.constant([2, 4, 8, 16, 32, 64], dtype=dtype)
      actual = utils.exp2(tf.constant([1, 2, 3, 4, 5, 6], dtype=dtype))

      with self.subTest('Values'):
        self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_exp2_with_overflow(self):
    test_cases = [(tf.uint8, 9), (tf.int8, 8), (tf.uint16, 17), (tf.int16, 16),
                  (tf.uint32, 33), (tf.int32, 32), (tf.uint64, 65),
                  (tf.int64, 64)]

    for dtype, value in test_cases:
      expected = tf.constant([dtype.max, dtype.max], dtype=dtype)
      actual = utils.exp2(tf.constant([value, value + 1], dtype=dtype))

      with self.subTest('Values'):
        self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_log2(self):
    test_cases = [(tf.float16, 1e-3), (tf.float32, 1e-6), (tf.float64, 1e-12)]

    for dtype, tolerance in test_cases:
      expected = tf.constant([1, 2, 3, 4, 5, 6], dtype=dtype)
      actual = utils.log2(tf.constant([2, 4, 8, 16, 32, 64], dtype=dtype))

      with self.subTest('Values'):
        self.assertAllClose(
            self.evaluate(actual), self.evaluate(expected), rtol=tolerance)
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_tent_transform(self):
    test_cases = [(tf.float16, 5e-3), (tf.float32, 1e-6), (tf.float64, 1e-12)]

    for dtype, tolerance in test_cases:
      expected = tf.constant([0, .2, .4, .6, .8, 1., .8, .6, .4, .2, 0.],
                             dtype=dtype)
      actual = utils.tent_transform(
          tf.constant([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], dtype=dtype))

      with self.subTest('Values'):
        self.assertAllClose(
            self.evaluate(actual), self.evaluate(expected), rtol=tolerance)
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)

  def test_filter_tensor(self):
    test_cases = [
        tf.uint8, tf.int8, tf.uint16, tf.int16, tf.uint32, tf.int32, tf.uint64,
        tf.int64
    ]

    for dtype in test_cases:
      # Should only retain values for which given bits in the bit mask are set.
      expected = tf.constant([[6, 5, 0], [3, 0, 1]], dtype=dtype)
      actual = utils.filter_tensor(
          # Tensor to filter
          tf.constant([[6, 5, 4], [3, 2, 1]], dtype=dtype),
          # Bit mask
          tf.constant([[1, 2, 11], [3, 5, 4]], dtype=dtype),
          # Indices (in LSB 0 order) of bits in the mask used for filtering
          tf.constant([[0, 1, 2], [0, 1, 2]], dtype=dtype))

      with self.subTest('Values'):
        self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
      with self.subTest('DType'):
        self.assertEqual(actual.dtype, expected.dtype)


if __name__ == '__main__':
  tf.test.main()
