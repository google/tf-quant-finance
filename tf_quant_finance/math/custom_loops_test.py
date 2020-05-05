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
"""Tests for custom_loops.py."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math import custom_loops

for_loop = custom_loops.for_loop


@test_util.run_all_in_graph_and_eager_modes
class ForLoopWithCustomGradientTest(parameterized.TestCase, tf.test.TestCase):

  def test_simple_grad_wrt_parameter(self):
    x = tf.constant([3.0])
    sigma = tf.constant(2.0)

    with tf.GradientTape() as tape:
      tape.watch(sigma)
      def body(i, state):
        del i
        x = state[0]
        return [x * sigma]
      out = for_loop(body, [x], [sigma], 3)[0]

    grad = tape.gradient(out, sigma)
    self.assertAllEqual(36, grad)

  def test_simple_grad_wrt_initial_state(self):
    x = tf.constant([3.0])
    sigma = tf.constant(2.0)

    with tf.GradientTape() as tape:
      tape.watch(x)
      def body(i, state):
        del i
        x = state[0]
        return [x * sigma]
      out = for_loop(body, [x], [sigma], 3)[0]

    grad = tape.gradient(out, x)
    self.assertAllEqual([8], grad)

  def test_multiple_state_vars(self):
    x = tf.constant([3.0, 4.0])
    y = tf.constant([5.0, 6.0])
    z = tf.constant([7.0, 8.0])
    alpha = tf.constant(2.0)
    beta = tf.constant(1.0)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([alpha, beta])
      def body(i, state):
        x, y, z = state
        k = tf.cast(i + 1, tf.float32)
        return [x * alpha - beta, y * k * alpha * beta, z * beta + x]
      out = for_loop(body, [x, y, z], [alpha, beta], 3)

    with self.subTest("independent_vars"):
      grad = tape.gradient(out[1], alpha)
      self.assertAllEqual(792, grad)
    with self.subTest("dependent_vars"):
      grad = tape.gradient(out[2], beta)
      self.assertAllEqual(63, grad)

  def test_batching(self):
    x = tf.constant([[3.0, 4.0], [30.0, 40.0]])
    y = tf.constant([[5.0, 6.0], [50.0, 60.0]])
    z = tf.constant([[7.0, 8.0], [70.0, 80.0]])
    alpha = tf.constant(2.0)
    beta = tf.constant(1.0)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([alpha, beta])
      def body(i, state):
        x, y, z = state
        k = tf.cast(i + 1, tf.float32)
        return [x * alpha - beta, y * k * alpha * beta, z * beta + x]
      out = for_loop(body, [x, y, z], [alpha, beta], 3)
    with self.subTest("independent_vars"):
      grad = tape.gradient(out[1], alpha)
      self.assertAllEqual([792, 7920], grad)
    with self.subTest("dependent_vars"):
      grad = tape.gradient(out[2], beta)
      self.assertAllEqual([63, 720], grad)

  def test_with_xla(self):
    @tf.function
    def fn():
      x = tf.constant([[3.0, 4.0], [30.0, 40.0]])
      y = tf.constant([[7.0, 8.0], [70.0, 80.0]])
      alpha = tf.constant(2.0)
      beta = tf.constant(1.0)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch([alpha, beta])
        def body(i, state):
          del i
          x, y = state
          return [x * alpha - beta, y * beta + x]
        out = for_loop(body, [x, y], [alpha, beta], 3)
      return tape.gradient(out[1], beta)

    grad = self.evaluate(tf.xla.experimental.compile(fn))[0]
    self.assertAllEqual([63, 720], grad)

  def test_state_independent_of_param(self):
    x = tf.constant([3.0])
    sigma = tf.constant(2.0)

    with tf.GradientTape() as tape:
      tape.watch(sigma)
      def body(i, state):
        del i
        return [state[0] * 2]
      out = for_loop(body, [x], [sigma], 3)[0]

    grad = tape.gradient(out, sigma)
    self.assertAllEqual(0, grad)

  @parameterized.named_parameters(
      {
          "testcase_name": "1_state_1_param",
          "state_dims": (1,),
          "num_params": 1,
      },
      {
          "testcase_name": "1_state_3_params",
          "state_dims": (1,),
          "num_params": 3,
      },
      {
          "testcase_name": "1_state_0_params",
          "state_dims": (1,),
          "num_params": 0,
      },
      {
          "testcase_name": "3_states_1_param",
          "state_dims": (1, 1, 1),
          "num_params": 1,
      },
      {
          "testcase_name": "3_states_3_param",
          "state_dims": (1, 1, 1),
          "num_params": 3,
      },
      {
          "testcase_name": "states_with_same_dims",
          "state_dims": (3, 3, 3),
          "num_params": 2,
      },
      {
          "testcase_name": "states_with_different_dims",
          "state_dims": (2, 3, 1),
          "num_params": 3,
      },
  )
  def test_shapes(self, state_dims, num_params):
    # Checks that the loop can handle various shapes and outputs correct shapes.
    def test_with_batch_shape(batch_shape):
      initial_state = [tf.ones(shape=batch_shape + (d,)) for d in state_dims]
      params = [tf.constant(1.0) for _ in range(num_params)]
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(initial_state)
        tape.watch(params)
        def body(i, state):
          del i
          if not params:
            return state
          sum_params = tf.add_n(params)
          state = [s * sum_params for s in state]
          return state
        final_state = for_loop(body, initial_state, params, 3)

      for s_in in initial_state:
        for s_out in final_state:
          grad = tape.gradient(s_out, s_in)
          self.assertAllEqual(s_in.shape, grad.shape)

      for p in params:
        for s_out in final_state:
          grad = tape.gradient(s_out, p)
          self.assertAllEqual(batch_shape, grad.shape)

    with self.subTest("no_batch"):
      test_with_batch_shape(batch_shape=())
    with self.subTest("simple_batch"):
      test_with_batch_shape(batch_shape=(5,))
    with self.subTest("complex_batch"):
      test_with_batch_shape(batch_shape=(2, 8, 3))


if __name__ == "__main__":
  tf.test.main()
