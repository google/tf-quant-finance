from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


boundary_numerator = tff.experimental.american_option_pricing.exercise_boundary.boundary_numerator


class ExerciseBoundaryTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'TestEmpty',
          'k': [],
          'tau': [],
          'r': [],
          'q': [],
          'sigma': [],
          'expected_boundary': []
      },
      {
          'testcase_name': 'TestSingle',
          'k': 45,
          'tau': 3,
          'r': 0.15,
          'q': 0.05,
          'sigma': 0.02,
          'expected_boundary': 1.56363801,
      },
      {
          'testcase_name': 'TestMultiple',
          'k': [100, 100],
          'tau': [1, 2],
          'r': [0.01, 0.02],
          'q': [0.01, 0.02],
          'sigma': [0.1, 0.15],
          'expected_boundary': [0.4849528, 0.47702501],
      },
  )
  def test_numerator(self, k, tau, r, q, sigma, expected_boundary):
    k = tf.constant(k, dtype=tf.float64)
    tau = tf.constant(tau, dtype=tf.float64)
    r = tf.constant(r, dtype=tf.float64)
    q = tf.constant(q, dtype=tf.float64)
    sigma = tf.constant(sigma, dtype=tf.float64)
    expected_boundary = tf.constant(expected_boundary, dtype=tf.float64)
    max_rel_error = 1e-8
    def _b_0(_, k, r, q):
      return k * tf.math.minimum(1, r/q)
    actual_boundary = boundary_numerator(tau, _b_0, k, r, q, sigma)
    assert np.all(
        np.abs(actual_boundary -
               expected_boundary) <= np.abs(expected_boundary) * max_rel_error)


if __name__ == '__main__':
  tf.test.main()
