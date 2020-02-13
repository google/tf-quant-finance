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
"""Tests for diff.py."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import math
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

# from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class DiffOpsTest(tf.test.TestCase):

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


if __name__ == '__main__':
  tf.test.main()
