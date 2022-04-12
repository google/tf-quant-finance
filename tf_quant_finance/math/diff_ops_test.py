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
"""Tests for diff.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import math
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class DiffOpsTest(parameterized.TestCase, tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_diffs(self):
    x = tf.constant([1, 2, 3, 4, 5])
    dx = self.evaluate(math.diff(x, order=1, exclusive=False))
    np.testing.assert_array_equal(dx, [1, 1, 1, 1, 1])

    dx1 = self.evaluate(math.diff(x, order=1, exclusive=True))
    np.testing.assert_array_equal(dx1, [1, 1, 1, 1])

    dx2 = self.evaluate(math.diff(x, order=2, exclusive=False))
    np.testing.assert_array_equal(dx2, [1, 2, 2, 2, 2])

  @test_util.deprecated_graph_mode_only
  def test_diffs_differentiable(self):
    """Tests that the diffs op is differentiable."""
    x = tf.constant(2.0)
    xv = tf.stack([x, x * x, x * x * x], axis=0)

    # Produces [x, x^2 - x, x^3 - x^2]
    dxv = self.evaluate(math.diff(xv))
    np.testing.assert_array_equal(dxv, [2., 2., 4.])

    grad = self.evaluate(tf.gradients(math.diff(xv), x)[0])
    # Note that TF gradients adds up the components of the jacobian.
    # The sum of [1, 2x-1, 3x^2-2x] at x = 2 is 12.
    self.assertEqual(grad, 12.0)

  @parameterized.named_parameters(
      {
          'testcase_name': 'exclusive_0',
          'exclusive': True,
          'axis': 0,
          'dx_true': np.array([[9, 18, 27, 36]])
      }, {
          'testcase_name': 'exclusive_1',
          'exclusive': True,
          'axis': 1,
          'dx_true': np.array([[1, 1, 1], [10, 10, 10]])
      }, {
          'testcase_name': 'nonexclusive_0',
          'exclusive': False,
          'axis': 0,
          'dx_true': np.array([[1, 2, 3, 4], [9, 18, 27, 36]]),
      }, {
          'testcase_name': 'nonexclusive_1',
          'exclusive': False,
          'axis': 1,
          'dx_true': np.array([[1, 1, 1, 1], [10, 10, 10, 10]]),
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_batched_axis(self, exclusive, axis, dx_true):
    """Tests batch diff works with axis argument use of exclusivity."""
    x = tf.constant([[1, 2, 3, 4], [10, 20, 30, 40]])
    dx = self.evaluate(math.diff(x, order=1, exclusive=exclusive, axis=axis))
    self.assertAllEqual(dx, dx_true)


if __name__ == '__main__':
  tf.test.main()
