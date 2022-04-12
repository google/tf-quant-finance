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
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

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
      self.assertAllEqual(8712, grad)
    with self.subTest("dependent_vars"):
      grad = tape.gradient(out[2], beta)
      self.assertAllEqual(783, grad)

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
    self.assertAllEqual(783, grad)

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
          "times": 3,
      },
      {
          "testcase_name": "1_state_3_params",
          "state_dims": (1,),
          "num_params": 3,
          "times": 3,
      },
      {
          "testcase_name": "1_state_0_params",
          "state_dims": (1,),
          "num_params": 0,
          "times": 3,
      },
      {
          "testcase_name": "3_states_1_param",
          "state_dims": (1, 1, 1),
          "num_params": 1,
          "times": 3,
      },
      {
          "testcase_name": "3_states_3_param",
          "state_dims": (1, 1, 1),
          "num_params": 3,
          "times": 3,
      },
      {
          "testcase_name": "states_with_same_dims",
          "state_dims": (3, 3, 3),
          "num_params": 2,
          "times": 3,
      },
      {
          "testcase_name": "states_with_different_dims",
          "state_dims": (2, 3, 1),
          "num_params": 3,
          "times": [3],
      },
      {
          "testcase_name": "states_with_different_dims_multiple_times",
          "state_dims": (2, 3, 1),
          "num_params": 3,
          "times": [2, 3],
      },
  )
  def test_shapes(self, state_dims, num_params, times):
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
        final_state = for_loop(body, initial_state, params, times)

      for s_in in initial_state:
        for s_out in final_state:
          grad = tape.gradient(s_out, s_in)
          self.assertAllEqual(s_in.shape, grad.shape)

      for p in params:
        for s_out in final_state:
          grad = tape.gradient(s_out, p)
          self.assertAllEqual([], grad.shape)

    with self.subTest("no_batch"):
      test_with_batch_shape(batch_shape=())
    with self.subTest("simple_batch"):
      test_with_batch_shape(batch_shape=(5,))
    with self.subTest("complex_batch"):
      test_with_batch_shape(batch_shape=(2, 8, 3))

  @parameterized.named_parameters(
      {
          "testcase_name": "params_test",
          "params_test": True
      },
      {
          "testcase_name": "initial_state_test",
          "params_test": False
      },
  )
  def test_accumulating_for_loop_grap_param(self, params_test):
    # Multiple  number of iterations produce correct gradients for params
    x = np.arange(24)
    x = np.reshape(x, [4, 3, 2])
    sigma_np = 2.0
    initial_state = tf.convert_to_tensor(x, dtype=tf.float64)
    sigma = tf.convert_to_tensor(sigma_np, dtype=tf.float64)

    def fn(initial_state, sigma):
      def body(i, state):
        del i
        x = state[0]
        return [x * sigma]
      return for_loop(body, [initial_state], [sigma], [3, 5])[0]

    # Tests for parameters gradient
    if params_test:
      def fwd_grad_fn(sigma):
        g = lambda sigma: fn(initial_state, sigma)
        return  tff.math.fwd_gradient(g, sigma, use_gradient_tape=True)

      def value_and_gradient(sigma):
        g = lambda sigma: fn(initial_state, sigma)
        return  tff.math.value_and_gradient(g, sigma, use_gradient_tape=True)
      expected_val = np.stack([sigma_np**3 * x, sigma_np**5 * x], axis=0)
      expected_fwd_grad = np.stack([3 * sigma_np**2 * x,
                                    5 * sigma_np**4 * x], axis=0)
      expected_grad = np.sum(expected_fwd_grad)
      with self.subTest("ParamsForwardGradXLA"):
        fwd_grad = tf.function(fwd_grad_fn, jit_compile=True)(sigma)
        self.assertAllClose(fwd_grad, expected_fwd_grad)
      with self.subTest("ParamsValueAndGradXLA"):
        val, grad = tf.function(value_and_gradient, jit_compile=True)(sigma)
        self.assertAllClose(expected_val, val)
        self.assertAllClose(grad, expected_grad)
      with self.subTest("ParamsForwardGrad"):
        fwd_grad = fwd_grad_fn(sigma)
        self.assertAllClose(fwd_grad, expected_fwd_grad)
      with self.subTest("ParamsValueAndGrad"):
        val, grad = value_and_gradient(sigma)
        self.assertAllClose(expected_val, val)
        self.assertAllClose(grad, expected_grad)
    # Tests for initial state gradient
    if not params_test:
      def state_fwd_grad_fn(initial_state):
        g = lambda initial_state: fn(initial_state, sigma)
        return  tff.math.fwd_gradient(g, initial_state, use_gradient_tape=True)

      def state_value_and_gradient(initial_state):
        g = lambda initial_state: fn(initial_state, sigma)
        return  tff.math.value_and_gradient(g, initial_state,
                                            use_gradient_tape=True)
      expected_val = np.stack([sigma_np**3 * x, sigma_np**5 * x], axis=0)
      expected_fwd_grad = np.stack([sigma_np**3 * np.ones_like(x),
                                    sigma_np**5 * np.ones_like(x)], axis=0)
      expected_grad = np.sum(expected_fwd_grad, axis=0)
      with self.subTest("StateForwardGradXLA"):
        fwd_grad = tf.function(state_fwd_grad_fn,
                               jit_compile=True)(initial_state)
        self.assertAllClose(fwd_grad, expected_fwd_grad)
      with self.subTest("StateValueAndGradXLA"):
        val, grad = tf.function(state_value_and_gradient,
                                jit_compile=True)(initial_state)
        self.assertAllClose(expected_val, val)
        self.assertAllClose(grad, expected_grad)
      with self.subTest("StateForwardGrad"):
        fwd_grad = state_fwd_grad_fn(initial_state)
        self.assertAllClose(fwd_grad, expected_fwd_grad)
      with self.subTest("StateValueAndGrad"):
        val, grad = state_value_and_gradient(initial_state)
        self.assertAllClose(expected_val, val)
        self.assertAllClose(grad, expected_grad)


if __name__ == "__main__":
  tf.test.main()
