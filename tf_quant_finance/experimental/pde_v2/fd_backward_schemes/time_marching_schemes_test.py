# Lint as: python2, python3
"""Tests for parabolic PDE time marching schemes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.crank_nicolson import crank_nicolson_scheme
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.explicit import explicit_scheme
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.extrapolation import extrapolation_scheme
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.implicit import implicit_scheme
from tf_quant_finance.experimental.pde_v2.fd_backward_schemes.weighted_implicit_explicit import weighted_implicit_explicit_scheme
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class TimeMarchingSchemeTest(tf.test.TestCase, parameterized.TestCase):

  parameters = {
      'testcase_name': 'Implicit',
      'scheme': implicit_scheme,
      'accuracy_order': 1
  }, {
      'testcase_name': 'Explicit',
      'scheme': explicit_scheme,
      'accuracy_order': 1
  }, {
      'testcase_name': 'Weighted',
      'scheme': weighted_implicit_explicit_scheme(theta=0.3),
      'accuracy_order': 1
  }, {
      'testcase_name': 'CrankNicolson',
      'scheme': crank_nicolson_scheme,
      'accuracy_order': 2
  }, {
      'testcase_name': 'Extrapolation',
      'scheme': extrapolation_scheme,
      'accuracy_order': 2
  }
  # Not including CompositeTimeMarchingScheme, its correctness is tested
  # elsewhere.

  @parameterized.named_parameters(*parameters)
  def testHomogeneous(self, scheme, accuracy_order):
    # Tests solving du/dt = At for a time step.
    # Compares with exact solution u(t) = exp(At) u(0).

    # Time step should be small enough to "resolve" different orders of accuracy
    time_step = 0.0001
    u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
    matrix = tf.constant(
        [[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]],
        dtype=tf.float64)

    tridiag_form = self._convert_to_tridiagonal_format(matrix)
    actual = self.evaluate(
        scheme(u, 0, time_step, lambda t: (tridiag_form, None),
               backwards=False))
    expected = self.evaluate(
        tf.squeeze(
            tf.matmul(tf.linalg.expm(matrix * time_step), tf.expand_dims(u,
                                                                         1))))

    error_tolerance = 30 * time_step**(accuracy_order + 1)
    self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

  @parameterized.named_parameters(*parameters)
  def testHomogeneousBackwards(self, scheme, accuracy_order):
    # Tests solving du/dt = At for a backward time step.
    # Compares with exact solution u(0) = exp(-At) u(t).
    time_step = 0.0001
    u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
    matrix = tf.constant(
        [[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]],
        dtype=tf.float64)

    tridiag_form = self._convert_to_tridiagonal_format(matrix)
    actual = self.evaluate(
        scheme(
            u,
            0,
            time_step,
            lambda t: (tridiag_form, None),
            backwards=True))

    expected = self.evaluate(
        tf.squeeze(
            tf.matmul(
                tf.linalg.expm(-matrix * time_step), tf.expand_dims(u, 1))))

    error_tolerance = 30 * time_step**(accuracy_order + 1)
    self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

  @parameterized.named_parameters(*parameters)
  def testInhomogeneous(self, scheme, accuracy_order):
    # Tests solving du/dt = At + b for a time step.
    # Compares with exact solution u(t) = exp(At) u(0) + (exp(At) - 1) A^(-1) b.
    time_step = 0.0001
    u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
    matrix = tf.constant(
        [[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]],
        dtype=tf.float64)
    b = tf.constant([1, -1, -2, 2], dtype=tf.float64)

    tridiag_form = self._convert_to_tridiagonal_format(matrix)
    actual = self.evaluate(
        scheme(
            u,
            0,
            time_step,
            lambda t: (tridiag_form, b),
            backwards=False))

    exponent = tf.linalg.expm(matrix * time_step)
    eye = tf.eye(4, 4, dtype=tf.float64)
    u = tf.expand_dims(u, 1)
    b = tf.expand_dims(b, 1)
    expected = (
        tf.matmul(exponent, u) +
        tf.matmul(exponent - eye, tf.matmul(tf.linalg.inv(matrix), b)))
    expected = self.evaluate(tf.squeeze(expected))

    error_tolerance = 30 * time_step**(accuracy_order + 1)
    self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

  @parameterized.named_parameters(*parameters)
  def testInhomogeneousBackwards(self, scheme, accuracy_order):
    # Tests solving du/dt = At + b for a backward time step.
    # Compares with exact solution u(0) = exp(-At) u(t)
    # + (exp(-At) - 1) A^(-1) b.
    time_step = 0.0001
    u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
    matrix = tf.constant(
        [[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]],
        dtype=tf.float64)
    b = tf.constant([1, -1, -2, 2], dtype=tf.float64)

    tridiag_form = self._convert_to_tridiagonal_format(matrix)
    actual = self.evaluate(
        scheme(
            u,
            0,
            time_step,
            lambda t: (tridiag_form, b),
            backwards=True))

    exponent = tf.linalg.expm(-matrix * time_step)
    eye = tf.eye(4, 4, dtype=tf.float64)
    u = tf.expand_dims(u, 1)
    b = tf.expand_dims(b, 1)
    expected = (
        tf.matmul(exponent, u) +
        tf.matmul(exponent - eye, tf.matmul(tf.linalg.inv(matrix), b)))
    expected = self.evaluate(tf.squeeze(expected))

    error_tolerance = 30 * time_step**(accuracy_order + 1)
    self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

  def _convert_to_tridiagonal_format(self, matrix):
    matrix_np = self.evaluate(matrix)
    n = matrix_np.shape[0]
    superdiag = [matrix_np[i, i + 1] for i in range(n - 1)] + [0]
    diag = [matrix_np[i, i] for i in range(n)]
    subdiag = [0] + [matrix_np[i + 1, i] for i in range(n - 1)]
    return tuple(
        tf.constant(v, dtype=matrix.dtype) for v in (diag, superdiag, subdiag))


if __name__ == '__main__':
  tf.test.main()
