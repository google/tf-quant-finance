from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


boundary_numerator = tff.experimental.american_option_pricing.exercise_boundary.boundary_numerator
boundary_denominator = tff.experimental.american_option_pricing.exercise_boundary.boundary_denominator
exercise_boundary = tff.experimental.american_option_pricing.exercise_boundary.exercise_boundary


@test_util.run_all_in_graph_and_eager_modes
class ExerciseBoundaryTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestMultiple',
          'k': [100, 100],
          'tau_grid': [[0., 0.5, 1.], [0., 1., 2.]],
          'r': [0.01, 0.02],
          'q': [0.01, 0.02],
          'sigma': [0.1, 0.15],
          'expected_shape': (2, 3)
      },)
  def test_numerator(self, k, tau_grid, r, q, sigma, expected_shape):
    k = tf.constant(k, dtype=tf.float64)
    tau_grid = tf.constant(tau_grid, dtype=tf.float64)
    r = tf.constant(r, dtype=tf.float64)
    q = tf.constant(q, dtype=tf.float64)
    sigma = tf.constant(sigma, dtype=tf.float64)
    # TODO(viktoriac): test against numpy results
    k_exp = k[:, tf.newaxis, tf.newaxis]
    r_exp = r[:, tf.newaxis, tf.newaxis]
    q_exp = q[:, tf.newaxis, tf.newaxis]
    def _b_0(tau_grid_exp):
      return tf.ones_like(tau_grid_exp) * k_exp * tf.math.minimum(
          tf.constant(1.0, dtype=tf.float64), r_exp / q_exp)
    actual_numerator = boundary_numerator(tau_grid, _b_0, k, r, q, sigma)
    actual_numerator = np.array(self.evaluate(actual_numerator))
    self.assertEqual(actual_numerator.shape, expected_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestMultiple',
          'k': [100, 100],
          'tau_grid': [[0., 0.5, 1.], [0., 1., 2.]],
          'r': [0.01, 0.02],
          'q': [0.01, 0.02],
          'sigma': [0.1, 0.15],
          'expected_shape': (2, 3),
      },)
  def test_denominator(self, k, tau_grid, r, q, sigma, expected_shape):
    k = tf.constant(k, dtype=tf.float64)
    tau_grid = tf.constant(tau_grid, dtype=tf.float64)
    r = tf.constant(r, dtype=tf.float64)
    q = tf.constant(q, dtype=tf.float64)
    sigma = tf.constant(sigma, dtype=tf.float64)
    # TODO(viktoriac): test against numpy results
    k_exp = k[:, tf.newaxis, tf.newaxis]
    r_exp = r[:, tf.newaxis, tf.newaxis]
    q_exp = q[:, tf.newaxis, tf.newaxis]
    def _b_0(tau_grid_exp):
      return tf.ones_like(tau_grid_exp) * k_exp * tf.math.minimum(
          tf.constant(1.0, dtype=tf.float64), r_exp / q_exp)
    actual_denominator = boundary_denominator(tau_grid, _b_0, k, r, q, sigma)
    actual_denominator = np.array(self.evaluate(actual_denominator))
    self.assertEqual(actual_denominator.shape, expected_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestMultiple',
          'k': [100, 100],
          'tau': [1, 2],
          'r': [0.01, 0.02],
          'q': [0.01, 0.02],
          'sigma': [0.1, 0.15],
          'grid_num_points': 3,
          'max_depth': 30,
          'tolerance': 1e-8,
          'integration_num_points': 32,
          'expected_shape': (2, 3),
      },)
  def test_exercise_boundary(self, k, tau, r, q, sigma, grid_num_points,
                             max_depth, tolerance, integration_num_points,
                             expected_shape):
    k = tf.constant(k, dtype=tf.float64)
    tau = tf.constant(tau, dtype=tf.float64)
    tau_grid = tf.linspace(
        tf.constant(0., dtype=tf.float64), tau, grid_num_points, axis=-1)
    r = tf.constant(r, dtype=tf.float64)
    q = tf.constant(q, dtype=tf.float64)
    sigma = tf.constant(sigma, dtype=tf.float64)
    k_exp = tf.expand_dims(k, axis=-1)
    r_exp = tf.expand_dims(r, axis=-1)
    q_exp = tf.expand_dims(q, axis=-1)
    tau_grid_exp = tf.expand_dims(tau_grid, axis=-1)
    # TODO(viktoriac): test against numpy results
    actual_boundary_function = exercise_boundary(tau_grid, k, r, q, sigma,
                                                 max_depth,
                                                 tolerance,
                                                 integration_num_points,
                                                 dtype=tf.float64)
    actual_boundary_exp = actual_boundary_function(tau_grid_exp)
    actual_boundary = tf.squeeze(actual_boundary_exp)
    # Test if next iteration of the function gives same results
    numerator = boundary_numerator(tau_grid, actual_boundary_function, k, r, q,
                                   sigma, integration_num_points)
    denominator = boundary_denominator(tau_grid, actual_boundary_function, k, r,
                                       q, sigma, integration_num_points)
    next_boundary_points = tf.math.divide_no_nan(
        k_exp * tf.math.exp(-(r_exp - q_exp) * tau_grid) * numerator,
        denominator)
    actual_boundary = np.array(self.evaluate(actual_boundary))
    next_boundary_points = np.array(self.evaluate(next_boundary_points))
    # Test results against next iteration of exercise boundary function
    np.testing.assert_allclose(actual_boundary, next_boundary_points, rtol=1e-8)
    # Test shape of result
    self.assertEqual(actual_boundary.shape, expected_shape)


if __name__ == '__main__':
  tf.test.main()
