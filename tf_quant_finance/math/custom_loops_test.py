# Copyright 2019 Google LLC [Apache-2.0]
"""Tests for `custom_loops.for_loop` (JAX native: gradients via jax.grad)."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from tf_quant_finance import _tf as tf
from tf_quant_finance import math as tff_math
from tf_quant_finance.math.custom_loops import for_loop


class ForLoopWithCustomGradientTest(parameterized.TestCase, tf.test.TestCase):

  def test_simple_grad_wrt_parameter(self):
    x = jnp.array([3.0])

    def fn(sigma):
      def body(i, state):
        del i
        return [state[0] * sigma]
      return for_loop(body, [x], [sigma], 3)[0]

    grad = jax.grad(lambda sigma: fn(sigma).sum())(2.0)
    self.assertAllEqual(36, grad)

  def test_simple_grad_wrt_initial_state(self):
    sigma = 2.0

    def fn(x):
      def body(i, state):
        del i
        return [state[0] * sigma]
      return for_loop(body, [x], [sigma], 3)[0]

    grad = jax.grad(lambda x: fn(x).sum())(jnp.array([3.0]))
    self.assertAllEqual([8], grad)

  def test_multiple_state_vars(self):
    x = jnp.array([3.0, 4.0])
    y = jnp.array([5.0, 6.0])
    z = jnp.array([7.0, 8.0])

    def fn(alpha, beta):
      def body(i, state):
        sx, sy, sz = state
        k = tf.cast(i + 1, tf.float32)
        return [sx * alpha - beta, sy * k * alpha * beta, sz * beta + sx]
      return for_loop(body, [x, y, z], [alpha, beta], 3)

    out_a2_b1 = fn(2.0, 1.0)
    with self.subTest("independent_vars"):
      grad = jax.grad(lambda a: fn(a, 1.0)[1].sum())(2.0)
      self.assertAllEqual(792, grad)
    with self.subTest("dependent_vars"):
      grad = jax.grad(lambda b: fn(2.0, b)[2].sum())(1.0)
      self.assertAllEqual(63, grad)

  def test_batching(self):
    x = jnp.array([[3.0, 4.0], [30.0, 40.0]])
    y = jnp.array([[5.0, 6.0], [50.0, 60.0]])
    z = jnp.array([[7.0, 8.0], [70.0, 80.0]])

    def fn(alpha, beta):
      def body(i, state):
        sx, sy, sz = state
        k = tf.cast(i + 1, tf.float32)
        return [sx * alpha - beta, sy * k * alpha * beta, sz * beta + sx]
      return for_loop(body, [x, y, z], [alpha, beta], 3)

    out = fn(2.0, 1.0)
    self.assertEqual(out[0].shape, (2, 2))
    with self.subTest("independent_vars"):
      grad = jax.grad(lambda a: fn(a, 1.0)[1].sum())(2.0)
      self.assertAllEqual(8712, grad)
    with self.subTest("dependent_vars"):
      grad = jax.grad(lambda b: fn(2.0, b)[2].sum())(1.0)
      self.assertAllEqual(783, grad)

  def test_state_independent_of_param(self):
    x = jnp.array([3.0])

    def fn(sigma):
      def body(i, state):
        del i
        return [state[0] * 2]
      return for_loop(body, [x], [sigma], 3)[0]

    grad = jax.grad(lambda sigma: fn(sigma).sum())(2.0)
    self.assertAllEqual(0, grad)

  @parameterized.named_parameters(
      {"testcase_name": "1_state_1_param", "state_dims": (1,), "num_params": 1, "times": 3},
      {"testcase_name": "3_states_1_param", "state_dims": (1, 1, 1), "num_params": 1, "times": 3},
      {"testcase_name": "states_with_same_dims", "state_dims": (3, 3, 3), "num_params": 2, "times": 3},
      {"testcase_name": "states_with_different_dims_multiple_times",
       "state_dims": (2, 3, 1), "num_params": 3, "times": [2, 3]},
  )
  def test_shapes(self, state_dims, num_params, times):
    def check(batch_shape):
      init = [jnp.ones(batch_shape + (d,)) for d in state_dims]
      params = [jnp.asarray(1.0) for _ in range(num_params)]

      def fn(initial_state, params):
        def body(i, state):
          del i
          if not params:
            return state
          sp = sum(params)
          return [s * sp for s in state]
        return for_loop(body, initial_state, params, times)

      # grad wrt each initial_state element must match that element's shape
      for k in range(len(init)):
        g = jax.grad(lambda s: fn([s if j == k else init[j] for j in range(len(init))],
                                  params)[0].sum())(init[k])
        self.assertEqual(init[k].shape, g.shape)

    check(())
    check((5,))
    check((2, 8, 3))

  @parameterized.named_parameters(
      {"testcase_name": "params_test", "params_test": True},
      {"testcase_name": "initial_state_test", "params_test": False},
  )
  def test_accumulating_for_loop_grap_param(self, params_test):
    x = np.arange(24).reshape(4, 3, 2)
    sigma_np = 2.0
    initial_state = jnp.asarray(x, dtype=jnp.float64)

    def fn(initial_state, sigma):
      def body(i, state):
        del i
        return [state[0] * sigma]
      return for_loop(body, [initial_state], [sigma], [3, 5])[0]

    expected_val = np.stack([sigma_np**3 * x, sigma_np**5 * x], axis=0)
    expected_fwd_grad = np.stack([3 * sigma_np**2 * x, 5 * sigma_np**4 * x], axis=0)

    if params_test:
      g = lambda sigma: fn(initial_state, sigma)
      fwd_grad = tff_math.fwd_gradient(g, jnp.asarray(sigma_np))
      self.assertAllClose(fwd_grad, expected_fwd_grad)
      val, grad = tff_math.value_and_gradient(g, jnp.asarray(sigma_np))
      self.assertAllClose(expected_val, val)
      self.assertAllClose(grad, np.sum(expected_fwd_grad))
    else:
      g = lambda initial_state: fn(initial_state, jnp.asarray(sigma_np))
      expected_fwd_grad_state = np.stack(
          [sigma_np**3 * np.ones_like(x), sigma_np**5 * np.ones_like(x)], axis=0)
      fwd_grad = tff_math.fwd_gradient(g, initial_state)
      self.assertAllClose(fwd_grad, expected_fwd_grad_state)
      val, grad = tff_math.value_and_gradient(g, initial_state)
      self.assertAllClose(expected_val, val)
      self.assertAllClose(grad, np.sum(expected_fwd_grad_state, axis=0))


if __name__ == "__main__":
  import unittest
  unittest.main()
