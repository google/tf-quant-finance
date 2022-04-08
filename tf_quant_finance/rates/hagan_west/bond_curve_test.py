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
"""Tests for bond_curve."""

import math
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tf_quant_finance.rates.hagan_west import bond_curve
from tf_quant_finance.rates.hagan_west import monotone_convex


@test_util.run_all_in_graph_and_eager_modes
class BondCurveTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_cashflow_times_cashflow_before_settelment_error(self, dtype):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          bond_curve.bond_curve(
              bond_cashflows=[
                  np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
                  np.array([30.0, 30.0, 30.0, 1030.0], dtype=dtype)
              ],
              bond_cashflow_times=[
                  np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
                  np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype)
              ],
              present_values=np.array([999.0, 1022.0], dtype=dtype),
              present_values_settlement_times=np.array([0.25, 0.25],
                                                       dtype=dtype),
              validate_args=True,
              dtype=dtype))

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_cashflow_times_are_strongly_ordered_error(self, dtype):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          bond_curve.bond_curve(
              bond_cashflows=[
                  np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
                  np.array([30.0, 30.0, 30.0, 1030.0], dtype=dtype)
              ],
              bond_cashflow_times=[
                  np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
                  np.array([0.5, 1.0, 1.5, 1.5], dtype=dtype)
              ],
              present_values=np.array([999.0, 1022.0], dtype=dtype),
              validate_args=True,
              dtype=dtype))

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_final_cashflow_is_the_largest_error(self, dtype):
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          bond_curve.bond_curve(
              bond_cashflows=[
                  np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
                  np.array([30.0, 30.0, 30.0, 3.0], dtype=dtype)
              ],
              bond_cashflow_times=[
                  np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
                  np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype)
              ],
              present_values=np.array([999.0, 1022.0], dtype=dtype),
              validate_args=True,
              dtype=dtype))

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_correctness(self, dtype):
    cashflows = [
        # 1 year bond with 5% three monthly coupon.
        np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
        # 2 year bond with 6% semi-annual coupon.
        np.array([30, 30, 30, 1030], dtype=dtype),
        # 3 year bond with 8% semi-annual coupon.
        np.array([40, 40, 40, 40, 40, 1040], dtype=dtype),
        # 4 year bond with 3% semi-annual coupon.
        np.array([15, 15, 15, 15, 15, 15, 15, 1015], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    ]
    pvs = np.array([
        999.68155223943393, 1022.322872470043, 1093.9894418810143,
        934.20885689015677
    ],
                   dtype=dtype)
    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows, cashflow_times, pvs, validate_args=True, dtype=dtype))
    with self.subTest('Times'):
      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 4.0])
    with self.subTest('Converged'):
      self.assertTrue(results.converged)
    with self.subTest('NotFailed'):
      self.assertFalse(results.failed)
    expected_discount_rates = np.array([5.0, 4.75, 4.53333333, 4.775],
                                       dtype=dtype) / 100
    expected_discount_factors = np.exp(-expected_discount_rates *
                                       [1.0, 2.0, 3.0, 4.0])
    with self.subTest('DiscountRates'):
      np.testing.assert_allclose(
          results.discount_rates, expected_discount_rates, atol=1e-6)
    with self.subTest('DiscountFactors'):
      np.testing.assert_allclose(
          results.discount_factors, expected_discount_factors, atol=1e-6)

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_unstable(self, dtype):
    """Demonstrates the instability of Hagan West for extreme cases."""
    cashflows = [
        # 1 year bond with 5% three monthly coupon.
        np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
        # 2 year bond with 6% semi-annual coupon.
        np.array([30, 30, 30, 1030], dtype=dtype),
        # 3 year bond with 8% semi-annual coupon.
        np.array([40, 40, 40, 40, 40, 1040], dtype=dtype),
        # 4 year bond with 3% semi-annual coupon.
        np.array([15, 15, 15, 15, 15, 15, 15, 1015], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    ]
    # Computed with discount rates of [5.0, 4.75, 4.53333333, 4.775]
    # which are 100 times the values in the previous test case.
    pvs = np.array([
        11.561316110080888, 2.6491572753698067, 3.4340789041846866,
        1.28732090544209
    ],
                   dtype=dtype)
    true_discount_rates = np.array([5.0, 4.75, 4.53333333, 4.775],
                                   dtype=dtype)
    # Check failure with default initial rates.
    results_default = self.evaluate(
        bond_curve.bond_curve(
            cashflows,
            cashflow_times,
            pvs,
            maximum_iterations=100,
            validate_args=True,
            dtype=dtype))
    self.assertFalse(results_default.converged)
    self.assertTrue(results_default.failed)
    self.assertFalse(np.isnan(results_default.discount_rates[0]))
    self.assertTrue(np.isnan(results_default.discount_rates[1]))

    # It even fails if we underestimate the result even marginally.
    # However the behaviour is different if we start above the true values.
    # See next test.
    results_close = self.evaluate(
        bond_curve.bond_curve(
            cashflows,
            cashflow_times,
            pvs,
            initial_discount_rates=true_discount_rates * 0.9999,
            maximum_iterations=100))
    with self.subTest('Converged'):
      self.assertFalse(results_close.converged)
    with self.subTest('Failed'):
      self.assertTrue(results_close.failed)
    with self.subTest('DiscountRates'):
      self.assertFalse(np.isnan(results_close.discount_rates[0]))
      self.assertFalse(np.isnan(results_close.discount_rates[1]))
      self.assertTrue(np.isnan(results_close.discount_rates[2]))

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_non_convex(self, dtype):
    """Demonstrates the nonconvexity of Hagan West for extreme cases."""
    # This is the same example as the previous one but with different starting
    # point.
    cashflows = [
        # 1 year bond with 5% three monthly coupon.
        np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
        # 2 year bond with 6% semi-annual coupon.
        np.array([30, 30, 30, 1030], dtype=dtype),
        # 3 year bond with 8% semi-annual coupon.
        np.array([40, 40, 40, 40, 40, 1040], dtype=dtype),
        # 4 year bond with 3% semi-annual coupon.
        np.array([15, 15, 15, 15, 15, 15, 15, 1015], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    ]
    # Computed with discount rates of [5.0, 4.75, 4.53333333, 4.775]
    # which are 100 times the values in the previous test case.
    pvs = np.array([
        11.561316110080888, 2.6491572753698067, 3.4340789041846866,
        1.28732090544209
    ],
                   dtype=dtype)
    true_discount_rates = np.array([5.0, 4.75, 4.53333333, 4.775],
                                   dtype=dtype)
    initial_rates = true_discount_rates * 1.01
    # Check failure with default initial rates.
    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows,
            cashflow_times,
            pvs,
            initial_discount_rates=initial_rates,
            maximum_iterations=100,
            validate_args=True,
            dtype=dtype))
    self.assertTrue(results.converged)
    self.assertFalse(results.failed)
    # It converges to a different set of rates.
    np.testing.assert_allclose(
        results.discount_rates,
        [4.96098643, 4.17592063, 2.83970042, 2.38685078],
        atol=1e-6)

    # However, the actual bond prices with the returned rates are indeed
    # correct.
    implied_pvs = self.evaluate(
        _compute_pv(cashflows, cashflow_times, results.discount_rates,
                    np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)))

    np.testing.assert_allclose(implied_pvs, pvs, rtol=1e-5)

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_flat_curve(self, dtype):
    """Checks that flat curves work."""
    cashflows = [
        # 1 year bond with 5% three monthly coupon.
        np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
        # 2 year bond with 6% semi-annual coupon.
        np.array([30, 30, 30, 1030], dtype=dtype),
        # 3 year bond with 8% semi-annual coupon.
        np.array([40, 40, 40, 40, 40, 1040], dtype=dtype),
        # 4 year bond with 3% semi-annual coupon.
        np.array([15, 15, 15, 15, 15, 15, 15, 1015], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    ]
    # Computed with a flat curve of 15%.
    pvs = np.array([906.27355957, 840.6517334, 823.73626709, 635.7076416],
                   dtype=dtype)
    true_discount_rates = np.array([0.15] * 4, dtype=dtype)
    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows, cashflow_times, pvs,
            discount_tolerance=1e-6, validate_args=True, dtype=dtype))
    with self.subTest('Converged'):
      self.assertTrue(results.converged)
    with self.subTest('NotFailed'):
      self.assertFalse(results.failed)
    with self.subTest('DiscountRates'):
      np.testing.assert_allclose(
          results.discount_rates, true_discount_rates, atol=1e-6)

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_negative_rates(self, dtype):
    """Checks that method works even if the actual rates are negative."""
    cashflows = [
        # 1 year bond with 5% three monthly coupon.
        np.array([12.5, 12.5, 12.5, 1012.5], dtype=dtype),
        # 2 year bond with 6% semi-annual coupon.
        np.array([30, 30, 30, 1030], dtype=dtype),
        # 3 year bond with 8% semi-annual coupon.
        np.array([40, 40, 40, 40, 40, 1040], dtype=dtype),
        # 4 year bond with 3% semi-annual coupon.
        np.array([15, 15, 15, 15, 15, 15, 15, 1015], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.25, 0.5, 0.75, 1.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    ]
    pvs = np.array(
        [1029.54933442, 1097.95320227, 1268.65376174, 1249.84175959],
        dtype=dtype)
    true_discount_rates = np.array([0.02, 0.01, -0.01, -0.03], dtype=dtype)
    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows, cashflow_times, pvs, validate_args=True, dtype=dtype))
    with self.subTest('Converged'):
      self.assertTrue(results.converged)
    with self.subTest('NotFailed'):
      self.assertFalse(results.failed)
    with self.subTest('DiscountRates'):
      np.testing.assert_allclose(
          results.discount_rates, true_discount_rates, atol=1e-4)

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_negative_forwards(self, dtype):
    """Checks that method works if the rates are positive by fwds are not."""
    true_discount_rates = np.array([0.12, 0.09, 0.02, 0.01, 0.01318182],
                                   dtype=dtype)
    # Note the implied forward rates for this rate curve are:
    # [0.12, 0.06, -0.05, -0.01, 0.02]
    cashflows = [
        np.array([1.2, 10.], dtype=dtype),
        np.array([1.1, 2.2, 1.4, 15.5], dtype=dtype),
        np.array([1.22, 0.45, 2.83, 96.0], dtype=dtype),
        np.array([12.33, 9.84, 1.15, 11.87, 0.66, 104.55], dtype=dtype),
        np.array([5.84, 0.23, 5.23, 114.95], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.15, 0.25], dtype=dtype),
        np.array([0.1, 0.2, 0.4, 0.5], dtype=dtype),
        np.array([0.22, 0.45, 0.93, 1.0], dtype=dtype),
        np.array([0.33, 0.84, 0.92, 1.22, 1.45, 1.5], dtype=dtype),
        np.array([0.43, 0.77, 1.3, 2.2], dtype=dtype)
    ]
    pvs = np.array(
        [10.88135262, 19.39268844, 98.48426722, 137.91938533, 122.63546542],
        dtype=dtype)
    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows, cashflow_times, pvs, validate_args=True, dtype=dtype))
    with self.subTest('Converged'):
      self.assertTrue(results.converged)
    with self.subTest('NotFailed'):
      self.assertFalse(results.failed)
    with self.subTest('NumIterations'):
      self.assertEqual(results.iterations, 6)
    with self.subTest('DiscountRates'):
      np.testing.assert_allclose(
          results.discount_rates, true_discount_rates, atol=1e-6)
    with self.subTest('Times'):
      np.testing.assert_allclose(
          results.times, [0.25, 0.5, 1., 1.5, 2.2], atol=1e-6)

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_zero_coupon_bond(self, dtype):
    cashflows = [
        # 6 months bond with no coupons.
        np.array([1020], dtype=dtype),
        # 1 year bond with 5% semi-annual coupon.
        np.array([25, 1025], dtype=dtype),
        # 2 year bond with 8% annual coupon.
        np.array([80, 1080], dtype=dtype),
        # 3 year bond with 3% annual coupon.
        np.array([30, 30, 1030], dtype=dtype)
    ]
    cashflow_times = [
        np.array([0.5], dtype=dtype),
        np.array([0.5, 1.0], dtype=dtype),
        np.array([1.0, 2.0], dtype=dtype),
        np.array([1.0, 2.0, 3.0], dtype=dtype)
    ]
    pvs = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=dtype)
    # We can calculate discount rates going step-by-step.
    r1 = -math.log(pvs[0] / cashflows[0][0]) / cashflow_times[0]
    r2 = -(
        math.log(
            (pvs[1] - cashflows[1][0] * math.exp(-r1 * cashflow_times[1][0]))
            / cashflows[1][1]) / cashflow_times[1][1])
    r3 = -(
        math.log(
            (pvs[2] - cashflows[2][0] * math.exp(-r2 * cashflow_times[2][0]))
            / cashflows[2][1]) / cashflow_times[2][1])
    r4 = -(
        math.log(
            (pvs[3] - cashflows[3][0] * math.exp(-r2 * cashflow_times[3][0]) -
             cashflows[3][1] * math.exp(-r3 * cashflow_times[3][1])) /
            cashflows[3][2]) / cashflow_times[3][2])
    true_discount_rates = np.array([r1, r2, r3, r4], dtype=dtype)

    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows, cashflow_times, pvs, validate_args=True, dtype=dtype))
    with self.subTest('Converged'):
      self.assertTrue(results.converged)
    with self.subTest('NotFailed'):
      self.assertFalse(results.failed)
    with self.subTest('NumIterations'):
      self.assertEqual(results.iterations, 4)
    with self.subTest('DiscountRates'):
      np.testing.assert_allclose(
          results.discount_rates, true_discount_rates, atol=1e-6)
    with self.subTest('Times'):
      np.testing.assert_allclose(results.times, [0.5, 1.0, 2.0, 3.0], atol=1e-6)

  @parameterized.named_parameters(
      ('single_precision', np.float32),
      ('double_precision', np.float64),
  )
  def test_only_zero_coupon_bonds(self, dtype):
    cashflows = [
        # 1 year bond with no coupons.
        np.array([1010], dtype=dtype),
        # 2 year bond with no coupons.
        np.array([1030], dtype=dtype),
        # 3 year bond with no coupons.
        np.array([1020], dtype=dtype),
        # 4 year bond with no coupons.
        np.array([1040], dtype=dtype)
    ]
    cashflow_times = [
        np.array([1.0], dtype=dtype),
        np.array([2.0], dtype=dtype),
        np.array([3.0], dtype=dtype),
        np.array([4.0], dtype=dtype)
    ]
    true_discount_rates = np.array([0.001, 0.2, 0.03, 0.0], dtype=dtype)
    pvs = np.array([(cashflows[i][0] * math.exp(-rate * cashflow_times[i][0]))
                    for i, rate in enumerate(true_discount_rates)],
                   dtype=dtype)
    results = self.evaluate(
        bond_curve.bond_curve(
            cashflows, cashflow_times, pvs, discount_tolerance=1e-6,
            validate_args=True, dtype=dtype))
    with self.subTest('Converged'):
      self.assertTrue(results.converged)
    with self.subTest('NotFailed'):
      self.assertFalse(results.failed)
    with self.subTest('NumIterations'):
      self.assertEqual(results.iterations, 1)
    with self.subTest('DiscountRates'):
      np.testing.assert_allclose(
          results.discount_rates, true_discount_rates, atol=1e-6)
    with self.subTest('ResultTimes'):
      np.testing.assert_allclose(results.times, [1.0, 2.0, 3.0, 4.0], atol=1e-6)


def _compute_pv(cashflows, cashflow_times, reference_rates, reference_times):
  times = tf.concat(cashflow_times, axis=0)
  groups = tf.concat([
      tf.zeros_like(cashflow, dtype=tf.int32) + i
      for i, cashflow in enumerate(cashflows)
  ],
                     axis=0)
  rates = monotone_convex.interpolate_yields(
      times, reference_times, yields=reference_rates)
  discounts = tf.math.exp(-times * rates)
  cashflows = tf.concat(cashflows, axis=0)
  return tf.math.segment_sum(discounts * cashflows, groups)


if __name__ == '__main__':
  tf.test.main()
