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
"""Tests for math.gradient.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance import math
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class GradientTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_forward_gradient(self):
    t = tf.range(1, 3, dtype=tf.float32)  # Shape [2]
    func = lambda t: tf.stack([t, t ** 2, t ** 3], axis=0)  # Shape [3, 2]
    fwd_grad = self.evaluate(math.fwd_gradient(func, t))
    self.assertEqual(fwd_grad.shape, (3, 2))
    np.testing.assert_allclose(fwd_grad, [[1., 1.], [2., 4.], [3., 12.]])

  @test_util.run_in_graph_and_eager_modes
  def test_make_val_and_grad_fn(self):
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @math.make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(input_tensor=scales * (x - minimum)**2)

    point = tf.constant([2.0, 2.0], dtype='float64')
    val, grad = self.evaluate(quadratic(point))
    self.assertNear(val, 5.0, 1e-5)
    self.assertArrayNear(grad, [4.0, 6.0], 1e-5)


if __name__ == '__main__':
  tf.test.main()
