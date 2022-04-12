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
"""Tests for math.piecewise."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class PadTest(parameterized.TestCase, tf.test.TestCase):

  def test_empty_tensor_pad(self):
    padded_tensors = tff.math.pad.pad_tensors([])
    self.assertEqual(padded_tensors, [])

  @parameterized.named_parameters(
      {
          "testcase_name": "DifferentBatchSize",
          "x": [[1, 2, 3, 9], [2, 3, 5, 2]],
          "y": [4, 5, 8],
          "pad_values": None,
          "expected_results": [[[1, 2, 3, 9], [2, 3, 5, 2]],
                               [4, 5, 8, 8]],
          "dtype": tf.int32
      }, {
          "testcase_name": "SameBatchSize",
          "x": [[1, 2, 3, 9], [2, 3, 5, 2]],
          "y": [[4, 5, 8], [4, 5, 1]],
          "pad_values": None,
          "expected_results": [[[1, 2, 3, 9], [2, 3, 5, 2]],
                               [[4, 5, 8, 8], [4, 5, 1, 1]]],
          "dtype": tf.float64
      }, {
          "testcase_name": "SameBatchSizeScalarPad",
          "x": [[1, 2, 3, 9], [2, 3, 5, 2]],
          "y": [[4, 5, 8], [4, 5, 1]],
          "pad_values": 10,
          "expected_results": [[[1, 2, 3, 9], [2, 3, 5, 2]],
                               [[4, 5, 8, 10], [4, 5, 1, 10]]],
          "dtype": tf.float64
      }, {
          "testcase_name": "SameBatchSizeWithPad",
          "x": [[[1, 2], [3, 9]], [[2, 3], [5, 2]]],
          "y": [[4, 5, 8], [4, 5, 1]],
          "pad_values": [[20, 30], 10],
          "expected_results": [[[[1, 2, 20], [3, 9, 30]],
                                [[2, 3, 20], [5, 2, 30]]],
                               [[4, 5, 8], [4, 5, 1]]],
          "dtype": tf.float64
      },
  )
  def test_tensor_pad(self, x, y, pad_values, expected_results, dtype):
    padded_tensors = tff.math.pad.pad_tensors(
        [x, y], pad_values=pad_values, dtype=dtype)
    for t, res in zip(padded_tensors, expected_results):
      self.assertAllEqual(self.evaluate(t), res)

  # Keep dates in ordinals for brevity
  @parameterized.named_parameters(
      {
          "testcase_name": "DifferentBatchSize",
          "x": [[737425, 737823], [737486, 737486]],
          "y": [737184, 740641, 740661],
          "expected_results": [
              [[737425, 737823, 737823], [737486, 737486, 737486]],
              [737184, 740641, 740661]]
      }, {
          "testcase_name": "SameBatchSize",
          "x": [[737425, 737823], [737486, 737486]],
          "y": [[737184, 740641, 740661], [737184, 740641, 740661]],
          "expected_results": [
              [[737425, 737823, 737823], [737486, 737486, 737486]],
              [[737184, 740641, 740661], [737184, 740641, 740661]]]
      },
  )
  def test_date_tensor_pad(self, x, y, expected_results):
    x = tff.datetime.dates_from_ordinals(x)
    y = tff.datetime.dates_from_ordinals(y)
    padded_tensors = tff.math.pad.pad_date_tensors([x, y])
    for t, res in zip(padded_tensors, expected_results):
      self.assertAllEqual(self.evaluate(t.ordinal()), res)

if __name__ == "__main__":
  tf.test.main()
