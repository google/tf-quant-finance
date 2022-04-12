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
"""Tests for math.gradient.py."""


import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class GradientTest(tf.test.TestCase):

  def test_forward_gradient(self):
    t = tf.range(1, 3, dtype=tf.float32)  # Shape [2]
    func = lambda t: tf.stack([t, t ** 2, t ** 3], axis=0)  # Shape [3, 2]
    with self.subTest("EagerExecution"):
      fwd_grad = self.evaluate(tff.math.fwd_gradient(func, t))
      self.assertEqual(fwd_grad.shape, (3, 2))
      np.testing.assert_allclose(fwd_grad, [[1., 1.], [2., 4.], [3., 12.]])
    with self.subTest("GraphExecution"):
      @tf.function
      def grad_computation():
        y = func(t)
        return tff.math.fwd_gradient(y, t)
      fwd_grad = self.evaluate(grad_computation())
      self.assertEqual(fwd_grad.shape, (3, 2))
      np.testing.assert_allclose(fwd_grad, [[1., 1.], [2., 4.], [3., 12.]])

  def test_forward_unconnected_gradient(self):
    t = tf.range(1, 3, dtype=tf.float32)  # Shape [2]
    zeros = tf.zeros([2], dtype=t.dtype)
    func = lambda t: tf.stack([zeros, zeros, zeros], axis=0)  # Shape [3, 2]
    expected_result = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    with self.subTest("EagerExecution"):
      fwd_grad = self.evaluate(tff.math.fwd_gradient(
          func, t, unconnected_gradients=tf.UnconnectedGradients.ZERO))
      self.assertEqual(fwd_grad.shape, (3, 2))
      np.testing.assert_allclose(fwd_grad, expected_result)
    with self.subTest("GraphExecution"):
      @tf.function
      def grad_computation():
        y = func(t)
        return tff.math.fwd_gradient(
            y, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
      fwd_grad = self.evaluate(grad_computation())
      self.assertEqual(fwd_grad.shape, (3, 2))
      np.testing.assert_allclose(fwd_grad, expected_result)

  def test_backward_gradient(self):
    t = tf.range(1, 3, dtype=tf.float32)  # Shape [2]
    func = lambda t: tf.stack([t, t ** 2, t ** 3], axis=0)  # Shape [3, 2]
    with self.subTest("EagerExecution"):
      backward_grad = self.evaluate(tff.math.gradients(func, t))
      self.assertEqual(backward_grad.shape, (2,))
      np.testing.assert_allclose(backward_grad, [6., 17.])
    with self.subTest("GraphExecution"):
      @tf.function
      def grad_computation():
        y = func(t)
        return tff.math.gradients(y, t)
      backward_grad = self.evaluate(grad_computation())
      self.assertEqual(backward_grad.shape, (2,))
      np.testing.assert_allclose(backward_grad, [6., 17.])

  def test_backward_unconnected_gradient(self):
    t = tf.range(1, 3, dtype=tf.float32)  # Shape [2]
    zeros = tf.zeros([2], dtype=t.dtype)
    expected_result = [0.0, 0.0]
    func = lambda t: tf.stack([zeros, zeros, zeros], axis=0)  # Shape [3, 2]
    with self.subTest("EagerExecution"):
      backward_grad = self.evaluate(tff.math.gradients(
          func, t, unconnected_gradients=tf.UnconnectedGradients.ZERO))
      self.assertEqual(backward_grad.shape, (2,))
      np.testing.assert_allclose(backward_grad, expected_result)
    with self.subTest("GraphExecution"):
      @tf.function
      def grad_computation():
        y = func(t)
        return tff.math.gradients(
            y, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
      backward_grad = self.evaluate(grad_computation())
      self.assertEqual(backward_grad.shape, (2,))
      np.testing.assert_allclose(backward_grad, expected_result)

  def test_make_val_and_grad_fn(self):
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @tff.math.make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(input_tensor=scales * (x - minimum)**2)

    point = tf.constant([2.0, 2.0], dtype=tf.float64)
    val, grad = self.evaluate(quadratic(point))
    self.assertNear(val, 5.0, 1e-5)
    self.assertArrayNear(grad, [4.0, 6.0], 1e-5)


if __name__ == "__main__":
  tf.test.main()
