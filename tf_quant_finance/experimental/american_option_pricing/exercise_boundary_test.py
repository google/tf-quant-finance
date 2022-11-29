from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import utils
from tf_quant_finance.experimental.american_option_pricing import common
from tf_quant_finance.experimental.american_option_pricing import exercise_boundary
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


boundary_numerator = exercise_boundary.boundary_numerator
boundary_denominator = exercise_boundary.boundary_denominator
exercise_boundary = exercise_boundary.exercise_boundary
divide_with_positive_denominator = common.divide_with_positive_denominator
machine_eps = common.machine_eps


@test_util.run_all_in_graph_and_eager_modes
class ExerciseBoundaryTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestMultiple',
          'k': [1, 2, 2],
          'tau_grid': [[0., 0.5, 1.], [0., 1., 2.], [0., 3., 6.]],
          'r': [0.01, 0.02, 0.04],
          'q': [0.01, 0.02, 0.0],
          'sigma': [0.1, 0.15, 0.05],
          'dtype': tf.float64,
          'expected_shape': (3, 3),
      },)
  def test_numerator(self, k, tau_grid, r, q, sigma, dtype, expected_shape):
    k = tf.constant(k, dtype=dtype)
    tau_grid = tf.constant(tau_grid, dtype=dtype)
    r = tf.constant(r, dtype=dtype)
    q = tf.constant(q, dtype=dtype)
    sigma = tf.constant(sigma, dtype=dtype)
    # TODO(viktoriac): test against numpy results
    k_exp = k[:, tf.newaxis, tf.newaxis]
    r_exp = r[:, tf.newaxis, tf.newaxis]
    q_exp = q[:, tf.newaxis, tf.newaxis]
    epsilon = machine_eps(dtype)
    def _b_0(tau_grid_exp):
      one = tf.constant(1.0, dtype=dtype)
      return tf.ones_like(tau_grid_exp) * k_exp * tf.where(
          tf.math.abs(q_exp) < epsilon, one,
          tf.math.minimum(one, r_exp / q_exp))
    actual_numerator = boundary_numerator(tau_grid, _b_0, k, r, q, sigma)
    actual_numerator = np.array(self.evaluate(actual_numerator))
    self.assertEqual(actual_numerator.shape, expected_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestMultiple',
          'k': [1, 2, 2],
          'tau_grid': [[0., 0.5, 1.], [0., 1., 2.], [0., 3., 6.]],
          'r': [0.01, 0.02, 0.04],
          'q': [0.01, 0.02, 0.0],
          'sigma': [0.1, 0.15, 0.05],
          'dtype': tf.float64,
          'expected_shape': (3, 3),
      },)
  def test_denominator(self, k, tau_grid, r, q, sigma, dtype, expected_shape):
    k = tf.constant(k, dtype=dtype)
    tau_grid = tf.constant(tau_grid, dtype=dtype)
    r = tf.constant(r, dtype=dtype)
    q = tf.constant(q, dtype=dtype)
    sigma = tf.constant(sigma, dtype=dtype)
    # TODO(viktoriac): test against numpy results
    k_exp = k[:, tf.newaxis, tf.newaxis]
    r_exp = r[:, tf.newaxis, tf.newaxis]
    q_exp = q[:, tf.newaxis, tf.newaxis]
    epsilon = machine_eps(dtype)
    def _b_0(tau_grid_exp):
      one = tf.constant(1.0, dtype=dtype)
      return tf.ones_like(tau_grid_exp) * k_exp * tf.where(
          tf.math.abs(q_exp) < epsilon, one,
          tf.math.minimum(one, r_exp / q_exp))
    actual_denominator = boundary_denominator(tau_grid, _b_0, k, r, q, sigma)
    actual_denominator = np.array(self.evaluate(actual_denominator))
    self.assertEqual(actual_denominator.shape, expected_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestMultiple',
          'k': [1, 5],
          'tau': [0.01, 1],
          'r': [0.01, 0.035],
          'q': [0.01, 0.07],
          'sigma': [0.1, 0.32],
          'grid_num_points': 40,
          'max_iterations': 100,
          'tolerance': 1e-8,
          'integration_num_points': 32,
          'convergence_atol': 1e-8,
          'dtype': tf.float64,
          'expected_shape': (2, 40),
      },
      {
          'testcase_name': 'TestRZero',
          'k': [1, 2],
          'tau': [0.01, 0.02],
          'r': [0.0, 0.0],
          'q': [0.01, 0.02],
          'sigma': [0.15, 0.32],
          'grid_num_points': 40,
          'max_iterations': 200,
          'tolerance': 1e-8,
          'integration_num_points': 32,
          'convergence_atol': 1e-8,
          'dtype': tf.float64,
          'expected_shape': (2, 40),
      },
      {
          'testcase_name': 'TestQZero',
          'k': [1, 2],
          'tau': [0.01, 0.02],
          'r': [0.01, 0.02],
          'q': [0.0, 0.0],
          'sigma': [0.15, 0.32],
          'grid_num_points': 40,
          'max_iterations': 200,
          'tolerance': 1e-8,
          'integration_num_points': 32,
          'convergence_atol': 1e-8,
          'dtype': tf.float64,
          'expected_shape': (2, 40),
      },
      )
  def test_exercise_boundary(self, k, tau, r, q, sigma, grid_num_points,
                             max_iterations, tolerance, integration_num_points,
                             convergence_atol, dtype, expected_shape):
    k = tf.constant(k, dtype=dtype)
    tau = tf.constant(tau, dtype=dtype)
    tau_grid = tf.linspace(tau / grid_num_points, tau, grid_num_points, axis=-1)
    r = tf.constant(r, dtype=dtype)
    q = tf.constant(q, dtype=dtype)
    sigma = tf.constant(sigma, dtype=dtype)
    k_exp = tf.expand_dims(k, axis=-1)
    r_exp = tf.expand_dims(r, axis=-1)
    q_exp = tf.expand_dims(q, axis=-1)
    # TODO(viktoriac): test against numpy results
    actual_boundary_function = exercise_boundary(tau_grid, k, r, q, sigma,
                                                 max_iterations,
                                                 tolerance,
                                                 integration_num_points,
                                                 dtype=dtype)
    actual_boundary = actual_boundary_function(tau_grid)
    # Test if next iteration of the function gives same results
    def actual_boundary_fn_3d(tau_grid_exp):
      shape_1 = utils.get_shape(tau_grid_exp)[1]
      shape_2 = utils.get_shape(tau_grid_exp)[2]
      tau_grid_exp_reshape = tf.reshape(tau_grid_exp, [-1, shape_1 * shape_2])
      interpolation = actual_boundary_function(tau_grid_exp_reshape)
      return tf.reshape(interpolation, [-1, shape_1, shape_2])
    numerator = boundary_numerator(tau_grid, actual_boundary_fn_3d, k, r, q,
                                   sigma, integration_num_points)
    denominator = boundary_denominator(tau_grid, actual_boundary_fn_3d, k, r,
                                       q, sigma, integration_num_points)
    next_boundary_points = divide_with_positive_denominator(
        k_exp * tf.math.exp(-(r_exp - q_exp) * tau_grid) * numerator,
        denominator)
    actual_boundary = np.array(self.evaluate(actual_boundary))
    next_boundary_points = np.array(self.evaluate(next_boundary_points))
    # Test results against next iteration of exercise boundary function
    np.testing.assert_allclose(actual_boundary, next_boundary_points,
                               atol=convergence_atol)
    # Test shape of result
    self.assertEqual(actual_boundary.shape, expected_shape)


if __name__ == '__main__':
  tf.test.main()
