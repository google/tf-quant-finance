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

"""Tests for Fletcher-Reeves algorithm."""


import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def _norm(x):
  return np.linalg.norm(x, np.inf)


# Test functions.
def _rosenbrock(x):
  """See https://en.wikipedia.org/wiki/Rosenbrock_function."""
  term1 = 100 * tf.reduce_sum(tf.square(x[1:] - tf.square(x[:-1])))
  term2 = tf.reduce_sum(tf.square(1 - x[:-1]))
  return term1 + term2


def _himmelblau(coord):
  """See https://en.wikipedia.org/wiki/Himmelblau%27s_function."""
  x, y = coord[..., 0], coord[..., 1]
  return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def _mc_cormick(coord):
  """See https://www.sfu.ca/~ssurjano/mccorm.html."""
  x = coord[0]
  y = coord[1]
  return tf.sin(x + y) + tf.square(x - y) - 1.5 * x + 2.5 * y + 1


def _beale(coord):
  """See https://www.sfu.ca/~ssurjano/beale.html."""
  x = coord[0]
  y = coord[1]
  term1 = (1.5 - x + x * y)**2
  term2 = (2.25 - x + x * y**2)**2
  term3 = (2.625 - x + x * y**3)**2
  return term1 + term2 + term3


@test_util.run_all_in_graph_and_eager_modes
class ConjugateGradientTest(tf.test.TestCase):

  def _check_algorithm(self,
                       func=None,
                       start_point=None,
                       gtol=1e-4,
                       expected_argmin=None):
    """Runs algorithm on given test case and verifies result."""
    val_grad_func = lambda x: tff.math.value_and_gradient(func, x)
    start_point = tf.constant(start_point, dtype=tf.float64)
    expected_argmin = np.array(expected_argmin, dtype=np.float64)

    f_call_ctr = tf.Variable(0, dtype=tf.int32)

    def val_grad_func_with_counter(x):
      with tf.compat.v1.control_dependencies(
          [tf.compat.v1.assign_add(f_call_ctr, 1)]):
        return val_grad_func(x)

    result = tff.math.optimizer.conjugate_gradient_minimize(
        val_grad_func_with_counter,
        start_point,
        tolerance=gtol,
        max_iterations=200)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    result = self.evaluate(result)
    f_call_ctr = self.evaluate(f_call_ctr)

    # Check that minimum is found.
    with self.subTest(name="Position"):
      self.assertAllClose(result.position, expected_argmin, rtol=1e-3,
                          atol=1e-3)
    # Check that gradient norm is below tolerance.
    grad_norm = np.max(result.objective_gradient)
    with self.subTest(name="GradientNorm"):
      self.assertLessEqual(grad_norm, 100 * gtol)
    # Check that number of function calls, declared by algorithm, is correct.
    with self.subTest(name="NumberOfEvals"):
      self.assertEqual(result.num_objective_evaluations, f_call_ctr)
    # Check returned function and gradient values.
    pos = tf.constant(result.position, dtype=tf.float64)
    f_at_pos, grad_at_pos = self.evaluate(val_grad_func(pos))
    with self.subTest(name="ObjectiveValue"):
      self.assertAllClose(result.objective_value, f_at_pos)
    with self.subTest(name="ObjectiveGradient"):
      self.assertAllClose(result.objective_gradient, grad_at_pos)
    # Check that all converged and none failed.
    with self.subTest(name="AllConverged"):
      self.assertTrue(np.all(result.converged))
    with self.subTest("NoneFailed"):
      self.assertFalse(np.any(result.failed))

  def test_univariate(self):
    self._check_algorithm(
        func=lambda x: (x[0] - 20)**2,
        start_point=[100.0],
        expected_argmin=[20.0])

  def test_quadratics(self):

    def test_random_quadratic(dim, seed):
      """Generates random test case for function x^T A x + b x."""
      np.random.seed(seed)
      a = np.random.uniform(size=(dim, dim))
      a = np.array(
          np.dot(a, a.T), dtype=np.float64)  # Must be positive semidefinite.
      b = np.array(np.random.uniform(size=(dim,)), dtype=np.float64)
      argmin = -np.dot(np.linalg.inv(a), b)
      a = tf.constant(a)
      b = tf.constant(b)

      def paraboloid(x):
        return 0.5 * tf.einsum("i,ij,j->", x, a, x) + tf.einsum("i,i->", b, x)

      self._check_algorithm(
          start_point=np.random.uniform(size=(dim,)),
          func=paraboloid,
          expected_argmin=argmin)

    test_random_quadratic(2, 43)
    test_random_quadratic(3, 43)
    test_random_quadratic(4, 43)
    test_random_quadratic(5, 43)
    test_random_quadratic(10, 43)
    test_random_quadratic(15, 43)

  def test_paraboloid_4th_order(self):
    self._check_algorithm(
        func=lambda x: tf.reduce_sum(x**4),
        start_point=[1, 2, 3, 4, 5],
        expected_argmin=[0, 0, 0, 0, 0],
        gtol=1e-10)

  def test_logistic_regression(self):
    dim = 5
    n_objs = 10000
    np.random.seed(1)
    betas = np.random.randn(dim)  # The true beta
    intercept = np.random.randn()  # The true intercept
    features = np.random.randn(n_objs, dim)  # The feature matrix
    probs = 1 / (1 + np.exp(
        -np.matmul(features, np.expand_dims(betas, -1)) - intercept))
    labels = np.random.binomial(1, probs)  # The true labels
    regularization = 0.8
    feat = tf.constant(features, dtype=tf.float64)
    lab = tf.constant(labels, dtype=feat.dtype)

    def f_negative_log_likelihood(params):
      intercept, beta = params[0], params[1:]
      logit = tf.matmul(feat, tf.expand_dims(beta, -1)) + intercept
      log_likelihood = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=lab, logits=logit))
      l2_penalty = regularization * tf.reduce_sum(beta**2)
      total_loss = log_likelihood + l2_penalty
      return total_loss
    start_point = np.ones(dim + 1)
    argmin = [
        -2.38636155, 1.61778325, -0.60694238, -0.51523609, -1.09832275,
        0.88892742
    ]

    self._check_algorithm(
        func=f_negative_log_likelihood,
        start_point=start_point,
        expected_argmin=argmin,
        gtol=1e-5)

  def test_data_fitting(self):
    """Tests MLE estimation for a simple geometric GLM."""
    n, dim = 100, 3
    dtype = tf.float64
    np.random.seed(234095)
    x = np.random.choice([0, 1], size=[dim, n])
    s = 0.01 * np.sum(x, 0)
    p = 1. / (1 + np.exp(-s))
    y = np.random.geometric(p)
    x_data = tf.convert_to_tensor(value=x, dtype=dtype)
    y_data = tf.expand_dims(tf.convert_to_tensor(value=y, dtype=dtype), -1)

    def neg_log_likelihood(state):
      state_ext = tf.expand_dims(state, 0)
      linear_part = tf.matmul(state_ext, x_data)
      linear_part_ex = tf.stack([tf.zeros_like(linear_part), linear_part],
                                axis=0)
      term1 = tf.squeeze(
          tf.matmul(tf.reduce_logsumexp(linear_part_ex, axis=0), y_data), -1)
      term2 = (0.5 * tf.reduce_sum(state_ext * state_ext, axis=-1) -
               tf.reduce_sum(linear_part, axis=-1))
      return tf.squeeze(term1 + term2)

    self._check_algorithm(
        func=neg_log_likelihood,
        start_point=np.ones(shape=[dim]),
        expected_argmin=[-0.020460034354, 0.171708568111, 0.021200423717])

  def test_rosenbrock_2d_v1(self):
    self._check_algorithm(
        func=_rosenbrock,
        start_point=[-1.2, 2],
        expected_argmin=[1.0, 1.0])

  def test_rosenbrock_2d_v2(self):
    self._check_algorithm(
        func=_rosenbrock,
        start_point=[7, -12],
        expected_argmin=[1.0, 1.0])

  def test_rosenbock_7d(self):
    self._check_algorithm(
        func=_rosenbrock,
        start_point=np.zeros(7),
        expected_argmin=np.ones(7))

  def test_himmelblau_v1(self):
    self._check_algorithm(
        func=_himmelblau,
        start_point=[4, 3],
        expected_argmin=[3.0, 2.0],
        gtol=1e-8)

  def test_himmelblau_v2(self):
    self._check_algorithm(
        func=_himmelblau,
        start_point=[-2, 3],
        expected_argmin=[-2.805118, 3.131312],
        gtol=1e-8)

  def test_himmelblau_v3(self):
    self._check_algorithm(
        func=_himmelblau,
        start_point=[-3, -3],
        expected_argmin=[-3.779310, -3.283186],
        gtol=1e-8)

  def test_himmelblau_v4(self):
    self._check_algorithm(
        func=_himmelblau,
        start_point=[3, -1],
        expected_argmin=[3.584428, -1.848126],
        gtol=1e-8)

  def test_mc_cormick(self):
    self._check_algorithm(
        func=_mc_cormick,
        start_point=[0, 0],
        expected_argmin=[-0.54719, -1.54719])

  def test_beale(self):
    self._check_algorithm(
        func=_beale,
        start_point=[-1.0, -1.0],
        expected_argmin=[3.0, 0.5],
        gtol=1e-8)

  def test_himmelblau_batch_all(self):
    self._check_algorithm(
        func=_himmelblau,
        start_point=[[1, 1], [-2, 2], [-1, -1], [1, -2]],
        expected_argmin=[[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186],
                         [3.584428, -1.848126]],
        gtol=1e-8)

  def test_himmelblau_batch_any(self):
    val_grad_func = tff.math.make_val_and_grad_fn(_himmelblau)
    starts = tf.constant([[1, 1], [-2, 2], [-1, -1], [1, -2]], dtype=tf.float64)
    expected_minima = np.array([[3, 2], [-2.805118, 3.131312],
                                [-3.779310, -3.283186], [3.584428, -1.848126]],
                               dtype=np.float64)

    # Run with `converged_any` stopping condition, to stop as soon as any of
    # the batch members have converged.
    batch_results = tff.math.optimizer.conjugate_gradient_minimize(
        val_grad_func,
        initial_position=starts,
        stopping_condition=tff.math.optimizer.converged_any,
        tolerance=1e-8)
    batch_results = self.evaluate(batch_results)

    self.assertFalse(np.any(batch_results.failed))  # None have failed.
    self.assertTrue(np.any(batch_results.converged))  # At least one converged.
    self.assertFalse(np.all(batch_results.converged))  # But not all did.

    # Converged points are near expected minima.
    for actual, expected in zip(batch_results.position[batch_results.converged],
                                expected_minima[batch_results.converged]):
      self.assertArrayNear(actual, expected, 1e-5)
    self.assertEqual(batch_results.num_iterations, 7)
    self.assertEqual(batch_results.num_objective_evaluations, 27)

  def test_dynamic_shapes(self):
    """Can build op with dynamic shapes in graph mode."""
    if tf.executing_eagerly():
      return
    minimum = np.array([1.0, 1.0])
    scales = np.array([2.0, 3.0])

    @tff.math.make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(input_tensor=scales * (x - minimum)**2)

    # Test with a vector of unknown dimension.
    start = tf.compat.v1.placeholder(tf.float32, shape=[None])
    op = tff.math.optimizer.conjugate_gradient_minimize(
        quadratic, initial_position=start, tolerance=1e-8)
    self.assertFalse(op.position.shape.is_fully_defined())

    with self.cached_session() as session:
      results = session.run(op, feed_dict={start: [0.6, 0.8]})
    self.assertTrue(results.converged)
    self.assertLessEqual(_norm(results.objective_gradient), 1e-8)
    self.assertArrayNear(results.position, minimum, 1e-5)

  def test_multiple_functions(self):
    # Define 3 independednt quadratic functions, each with its own minimum.
    minima = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    func = lambda x: tf.reduce_sum(tf.square(x - minima), axis=1)
    self._check_algorithm(
        func=func, start_point=np.zeros_like(minima), expected_argmin=minima)

  def test_float32(self):
    minimum = np.array([1.0, 1.0], dtype=np.float32)
    scales = np.array([2.0, 3.0], dtype=np.float32)
    start = np.zeros_like(minimum)

    @tff.math.make_val_and_grad_fn
    def quadratic(x):
      return tf.reduce_sum(input_tensor=scales * (x - minimum)**2)

    result = tff.math.optimizer.conjugate_gradient_minimize(
        quadratic, initial_position=start)
    self.assertEqual(result.position.dtype, tf.float32)
    self.assertArrayNear(self.evaluate(result.position), minimum, 1e-5)


if __name__ == "__main__":
  tf.test.main()
