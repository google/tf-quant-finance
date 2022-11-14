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
"""Tests for numeric integration methods."""

import collections
import math
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.math.integration import adaptive_update

tff_int = tff.math.integration

IntegrationTestCase = collections.namedtuple('IntegrationTestCase', [
    'func', 'lower', 'upper', 'tolerance', 'antiderivative',
    'expected_gauss_n32_result'
])

AdaptiveUpdateTestCase = collections.namedtuple('AdaptiveUpdateTestCase', [
    'lower', 'upper', 'estimate', 'error', 'tolerance', 'new_lower',
    'new_upper', 'sum_goods'
])

# pylint:disable=g-long-lambda
BASIC_TEST_CASES = [
    IntegrationTestCase(
        func=lambda x: tf.exp(2 * x + 1),
        lower=1.0,
        upper=3.0,
        tolerance=1e-5,
        antiderivative=lambda x: np.exp(2 * x + 1) / 2,
        # calculated using scipy
        expected_gauss_n32_result=538.2738107526367,
    ),
    IntegrationTestCase(
        func=lambda x: x**5,
        lower=-10.0,
        upper=100.0,
        tolerance=1e-5,
        antiderivative=lambda x: x**6 / 6,
        expected_gauss_n32_result=166666500000.0006,
    ),
    IntegrationTestCase(
        func=lambda x: (x**3 + x**2 - 4 * x + 1) / (x**2 + 1)**2,
        lower=0.0,
        upper=10.0,
        tolerance=1e-5,
        antiderivative=lambda x: sum([
            2.5 / (x**2 + 1),
            0.5 * np.log(x**2 + 1),
            np.arctan(x),
        ]),
        expected_gauss_n32_result=1.303440407930151,
    ),
    IntegrationTestCase(
        func=lambda x: (tf.sinh(2 * x) + 3 * tf.sinh(x)) /
        (tf.cosh(x)**2 + 2 * tf.cosh(0.5 * x)**2),
        lower=2.0,
        upper=4.0,
        tolerance=1e-5,
        antiderivative=lambda x: sum([
            np.log(np.cosh(x)**2 + np.cosh(x) + 1),
            (4 / np.sqrt(3)) * np.arctan((1 + 2 * np.cosh(x)) / np.sqrt(3.0)),
        ]),
        expected_gauss_n32_result=4.102650634197022,
    ),
    IntegrationTestCase(
        func=lambda x: tf.exp(2 * x) * tf.math.sqrt(tf.exp(x) + tf.exp(2 * x)),
        lower=2.0,
        upper=4.0,
        tolerance=1e-5,
        antiderivative=lambda x: sum([
            np.sqrt((np.exp(x) + np.exp(2 * x))**3) / 3,
            -(1 + 2 * np.exp(x)) * np.sqrt(np.exp(x) + np.exp(2 * x)) / 8,
            np.log(np.sqrt(1 + np.exp(x)) + np.exp(0.5 * x)) / 8,
        ]),
        expected_gauss_n32_result=54842.93035676345,
    ),
    IntegrationTestCase(
        func=lambda x: tf.exp(-x**2),
        lower=0.0,
        upper=1.0,
        tolerance=1e-5,
        antiderivative=lambda x: 0.5 * np.sqrt(np.pi) * np.array([math.erf(x)]),
        expected_gauss_n32_result=0.746824132812427,
    ),
]

TEST_CASE_RAPID_CHANGE = IntegrationTestCase(
    func=lambda x: 1.0 / tf.sqrt(x + 1e-6),
    lower=0.0,
    upper=1.0,
    tolerance=1e-5,
    antiderivative=lambda x: 2.0 * np.sqrt(x + 1e-6),
    expected_gauss_n32_result=1.9731597165275736,
)

TEST_CASE_MULTIPLE_INTERVALS = [
    IntegrationTestCase(
        func=lambda x: 1.0 / tf.sqrt(x + 1e-6),
        lower=[0.0, 2.5],
        upper=[1.0, 4.0],
        tolerance=1e-8,
        antiderivative=lambda x: [2.0 * np.sqrt(y + 1e-6) for y in x],
        expected_gauss_n32_result=1.9731597165275736,
    ),
]

ADAPTIVE_UPDATE_TEST_CASES = [
    AdaptiveUpdateTestCase(
        lower=[[1.0, 2.0], [3.5, 4.0]],
        upper=[[2.0, 3.0], [4.0, 4.5]],
        estimate=[[1.0, 2.0], [3.0, 4.0]],
        error=[[0.02, 0.04], [0.1, 0.8]],
        tolerance=0.1,
        new_lower=[[1.0, 1.5], [4.0, 4.25]],
        new_upper=[[1.5, 2.0], [4.25, 4.5]],
        sum_goods=[2.0, 3.0],
    ),
    AdaptiveUpdateTestCase(
        lower=[[1.0, 2.0, 4.0, 5.0], [3.5, 4.0, 7.5, 8.0]],
        upper=[[2.0, 3.0, 5.0, 6.0], [4.0, 4.5, 8.0, 8.5]],
        estimate=[[1.0, 2.0, 3.0, 2.0], [3.0, 4.0, 1.5, 0.5]],
        error=[[0.002, 0.003, 0.01, 0.001], [0.01, 0.008, 0.004, 0.001]],
        tolerance=0.005,
        new_lower=[[], []],
        new_upper=[[], []],
        sum_goods=[8.0, 9.0],
    ),
    AdaptiveUpdateTestCase(
        lower=[[4.0, 5.0], [7.5, 8.0]],
        upper=[[5.0, 6.0], [8.0, 8.5]],
        estimate=[[3.0, 2.0], [1.5, 0.5]],
        error=[[0.01, 0.001], [0.004, 0.001]],
        tolerance=1e-4,
        new_lower=[[4.0, 5.0, 4.5, 5.5], [7.5, 8.0, 7.75, 8.25]],
        new_upper=[[4.5, 5.5, 5.0, 6.0], [7.75, 8.25, 8.0, 8.5]],
        sum_goods=[0.0, 0.0],
    ),
]


@test_util.run_all_in_graph_and_eager_modes
class IntegrationTest(tf.test.TestCase):

  def _test_batches_and_types(self, integrate_function, args):
    """Checks handling batches and dtypes."""
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    a = [[0.0, 0.0], [0.0, 0.0]]
    b = [[np.pi / 2, np.pi], [1.5 * np.pi, 2 * np.pi]]
    a = [a, a]
    b = [b, b]
    k = tf.constant([[[[1.0]]], [[[2.0]]]])
    func = lambda x: tf.cast(k, dtype=x.dtype) * tf.sin(x)
    ans = [[[1.0, 2.0], [1.0, 0.0]], [[2.0, 4.0], [2.0, 0.0]]]

    results = []
    for dtype in dtypes:
      lower = tf.constant(a, dtype=dtype)
      upper = tf.constant(b, dtype=dtype)
      results.append(integrate_function(func, lower, upper, **args))

    results = self.evaluate(results)

    for i in range(len(results)):
      assert results[i].dtype == dtypes[i]
      assert np.allclose(results[i], ans, atol=1e-3)

  def _test_accuracy(self, integrate_function, args, test_case, max_rel_error):
    func = test_case.func
    lower = tf.constant(test_case.lower, dtype=tf.float64)
    upper = tf.constant(test_case.upper, dtype=tf.float64)
    exact = test_case.antiderivative(
        test_case.upper) - test_case.antiderivative(test_case.lower)
    approx = integrate_function(func, lower, upper, **args)
    approx = self.evaluate(approx)
    assert np.abs(approx - exact) <= np.abs(exact) * max_rel_error

  def _test_against_scipy_results(self, integrate_function, args, test_case,
                                  scipy_exact, max_rel_error):
    func = test_case.func
    lower = tf.constant(test_case.lower, dtype=tf.float64)
    upper = tf.constant(test_case.upper, dtype=tf.float64)
    approx = integrate_function(func, lower, upper, **args)
    approx = self.evaluate(approx)
    assert np.abs(approx - scipy_exact) <= np.abs(scipy_exact) * max_rel_error

  def _test_kronrod_accuracy(self, integrate_function, args, test_case,
                             max_rel_error):
    func = test_case.func
    lower = tf.constant(test_case.lower, dtype=tf.float64)
    upper = tf.constant(test_case.upper, dtype=tf.float64)
    tolerance = test_case.tolerance
    exact = np.array(test_case.antiderivative(np.array(
        test_case.upper))) - np.array(
            test_case.antiderivative(np.array(test_case.lower)))
    approx = integrate_function(func, lower, upper, tolerance, **args)
    approx = np.array(self.evaluate(approx))
    assert np.all(np.abs(approx - exact) <= np.abs(exact) * max_rel_error)

  def _test_adaptive_update(self, update_function, args, test_case):
    lower = tf.constant(test_case.lower, dtype=tf.float64)
    upper = tf.constant(test_case.upper, dtype=tf.float64)
    estimate = tf.constant(test_case.estimate, dtype=tf.float64)
    error = tf.constant(test_case.error, dtype=tf.float64)
    tolerance = test_case.tolerance
    expected_new_lower = tf.constant(test_case.new_lower, dtype=tf.float64)
    expected_new_upper = tf.constant(test_case.new_upper, dtype=tf.float64)
    expected_sum_goods = tf.constant(test_case.sum_goods, dtype=tf.float64)
    calc = update_function(lower, upper, estimate, error, tolerance, **args)
    actual_new_lower, actual_new_upper, actual_sum_goods = self.evaluate(calc)
    self.assertAllClose(expected_new_lower, actual_new_lower)
    self.assertAllClose(expected_new_upper, actual_new_upper)
    self.assertAllClose(expected_sum_goods, actual_sum_goods)

  def _test_gradient(self, integrate_function, args):
    """Checks that integration result can be differentiated."""

    # We consider I(a) = int_0^1 cos(ax) dx.
    # Then dI/da = (a*cos(a) - sin(a))/a^2.
    def integral(a):
      return integrate_function(
          lambda x: tf.cos(a * x), 0.0, 1.0, dtype=tf.float64, **args)

    a = tf.constant(0.5, dtype=tf.float64)
    di_da = tff.math.fwd_gradient(integral, a)

    true_di_da = lambda a: (a * np.cos(a) - np.sin(a)) / (a**2)
    self.assertAllClose(self.evaluate(di_da), true_di_da(0.5))

  def test_integrate_batches_and_types(self):
    self._test_batches_and_types(tff_int.integrate, {})
    for method in tff_int.IntegrationMethod:
      self._test_batches_and_types(tff_int.integrate, {'method': method})

  def test_integrate_accuracy(self):
    for test_case in BASIC_TEST_CASES:
      self._test_accuracy(tff_int.integrate, {}, test_case, 1e-8)
      for method in tff_int.IntegrationMethod:
        self._test_accuracy(tff_int.integrate, {'method': method}, test_case,
                            1e-8)

  def test_gauss_legendre_accuracy_compared_to_scipy_results(self):
    for test_case in BASIC_TEST_CASES:
      with self.subTest('gauss'):
        self._test_against_scipy_results(
            tff_int.integrate,
            {'method': tff_int.IntegrationMethod.GAUSS_LEGENDRE}, test_case,
            test_case.expected_gauss_n32_result, 1e-14)

  def test_adaptive_update(self):
    for test_case in ADAPTIVE_UPDATE_TEST_CASES:
      self._test_adaptive_update(adaptive_update.update, {}, test_case)

  def test_kronrod_accuracy(self):
    for test_case in BASIC_TEST_CASES:
      # G-K requires a batch dimension
      test_case = IntegrationTestCase(
          func=test_case.func,
          lower=[test_case.lower],
          upper=[test_case.upper],
          tolerance=test_case.tolerance,
          antiderivative=test_case.antiderivative,
          expected_gauss_n32_result=test_case.expected_gauss_n32_result)
      self._test_kronrod_accuracy(tff_int.gauss_kronrod, {}, test_case, 1e-8)

  def test_kronrod_accuracy_multiple_intervals(self):
    for test_case in TEST_CASE_MULTIPLE_INTERVALS:
      for num_points in [15, 21, 31]:
        with self.subTest(f'num_points={num_points}'):
          self._test_kronrod_accuracy(tff_int.gauss_kronrod, {
              'num_points': num_points,
              'max_depth': 30
          }, test_case, 1e-8)

  def test_integrate_gradient(self):
    for method in tff_int.IntegrationMethod:
      self._test_gradient(tff_int.integrate, {'method': method})

  def test_integrate_int_limits(self):
    for method in tff_int.IntegrationMethod:
      result = tff_int.integrate(tf.sin, 0, 1, method=method, dtype=tf.float64)
      result = self.evaluate(result)
      self.assertAllClose(0.459697694, result)

  def test_simpson_batches_and_types(self):
    self._test_batches_and_types(tff_int.simpson, {})

  def test_simpson_accuracy(self):
    for test_case in BASIC_TEST_CASES:
      self._test_accuracy(tff_int.simpson, {}, test_case, 1e-8)

  def test_simpson_rapid_change(self):
    self._test_accuracy(tff_int.simpson, {'num_points': 1001},
                        TEST_CASE_RAPID_CHANGE, 2e-1)
    self._test_accuracy(tff_int.simpson, {'num_points': 10001},
                        TEST_CASE_RAPID_CHANGE, 3e-2)
    self._test_accuracy(tff_int.simpson, {'num_points': 100001},
                        TEST_CASE_RAPID_CHANGE, 5e-4)
    self._test_accuracy(tff_int.simpson, {'num_points': 1000001},
                        TEST_CASE_RAPID_CHANGE, 3e-6)

  def test_simpson_gradient(self):
    self._test_gradient(tff_int.simpson, {})


if __name__ == '__main__':
  tf.test.main()
