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

# Lint as: python2, python3
"""Tests for implied_volatility.newton_vol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from nomisma_quant_finance.implied_volatility import newton_vol
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class NewtonVolTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods in newton_vol module."""

  def test_newton_root_finder(self):
    """Tests that the newton root finder works on a square root example."""

    # Set up the problem of finding the square roots of three numbers.
    constants = np.array([4.0, 9.0, 16.0])
    initial_values = np.ones(len(constants))

    def objective_and_gradient(values):
      objective = values**2 - constants
      gradient = 2.0 * values
      return objective, gradient

    # Obtain and evaluate a tensor containing the roots.
    roots = newton_vol.newton_root_finder(objective_and_gradient,
                                          initial_values)
    root_values, converged, failed = self.evaluate(roots)

    # Reference values.
    roots_bench = np.array([2.0, 3.0, 4.0])
    converged_bench = np.array([True, True, True])
    failed_bench = np.array([False, False, False])

    # Assert that the values we obtained are close to the true values.
    np.testing.assert_array_equal(converged, converged_bench)
    np.testing.assert_array_equal(failed, failed_bench)
    np.testing.assert_almost_equal(root_values, roots_bench, decimal=7)

  def test_failure_and_non_convergence(self):
    """Tests that we can determine when the root finder has failed."""

    # Set up the problem of finding the square roots of three numbers.
    constants = np.array([4.0, 9.0, 16.0])
    # Choose a bad initial position.
    initial_values = np.zeros(len(constants))

    def objective_and_gradient(values):
      objective = values**2 - constants
      gradient = 2.0 * values
      return objective, gradient

    # Obtain and evaluate a tensor containing the roots.
    roots = newton_vol.newton_root_finder(objective_and_gradient,
                                          initial_values)
    _, converged, failed = self.evaluate(roots)

    # Reference values - we should not have converged and should have failed.
    converged_bench = np.array([False, False, False])
    failed_bench = np.array([True, True, True])

    # Assert that the values we obtained are close to the true values.
    np.testing.assert_array_equal(converged, converged_bench)
    np.testing.assert_array_equal(failed, failed_bench)

  def test_too_low_max_iterations(self):
    """Tests that we can determine when max_iterations was too small."""

    # Set up the problem of finding the square roots of three numbers.
    constants = np.array([4.0, 9.0, 16.0])
    initial_values = np.ones(len(constants))

    def objective_and_gradient(values):
      objective = values**2 - constants
      gradient = 2.0 * values
      return objective, gradient

    # Obtain and evaluate a tensor containing the roots.
    roots = newton_vol.newton_root_finder(
        objective_and_gradient, initial_values, max_iterations=1)
    _, converged, failed = self.evaluate(roots)

    # Reference values - we should neither have converged nor failed.
    converged_bench = np.array([False, False, False])
    failed_bench = np.array([False, False, False])

    # Assert that the values we obtained are close to the true values.
    np.testing.assert_array_equal(converged, converged_bench)
    np.testing.assert_array_equal(failed, failed_bench)

  def test_basic_newton_finder(self):
    """Tests the Newton root finder recovers the volatility on a few cases."""
    forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0])
    expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0])
    discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    init_vols = np.array([2.0, 0.5, 2.0, 0.5, 1.5, 1.5])
    option_signs = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    prices = np.array([
        0.38292492, 0.19061012, 0.38292492, 0.09530506, 0.27632639, 0.52049988
    ])
    results = newton_vol.newton_implied_vol(
        forwards,
        strikes,
        expiries,
        discounts,
        prices,
        init_vols,
        option_signs,
        max_iterations=100)
    implied_vols, converged, failed = self.evaluate(results)
    num_volatilities = len(volatilities)
    self.assertAllEqual(np.ones(num_volatilities, dtype=np.bool), converged)
    self.assertAllEqual(np.zeros(num_volatilities, dtype=np.bool), failed)
    self.assertArrayNear(volatilities, implied_vols, 1e-7)

  def test_basic_radiocic_newton_combination_finder(self):
    """Tests the Newton root finder recovers the volatility on a few cases."""
    forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0])
    expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0])
    discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    option_signs = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    prices = np.array([
        0.38292492, 0.19061012, 0.38292492, 0.09530506, 0.27632639, 0.52049988
    ])
    results = newton_vol.implied_vol(forwards, strikes, expiries, discounts,
                                     prices, option_signs)
    implied_vols, converged, failed = self.evaluate(results)
    num_volatilities = len(volatilities)
    self.assertAllEqual(np.ones(num_volatilities, dtype=np.bool), converged)
    self.assertAllEqual(np.zeros(num_volatilities, dtype=np.bool), failed)
    self.assertArrayNear(volatilities, implied_vols, 1e-7)

  @parameterized.named_parameters(
      # This case should hit the call lower bound since C = F - K.
      ('call_lower', 0.0, 1.0, 1.0, 1.0, 1.0),
      # This case should hit the call upper bound since C = F
      ('call_upper', 1.0, 1.0, 1.0, 1.0, 1.0),
      # This case should hit the put upper bound since C = K
      ('put_lower', 1.0, 1.0, 1.0, 1.0, -1.0),
      # This case should hit the call lower bound since C = F - K.
      ('put_upper', 0.0, 1.0, 1.0, 1.0, -1.0))
  def test_implied_vol_validate_raises(self, price, forward, strike, expiry,
                                       option_sign):
    """Tests validation errors raised where BS model assumptions violated."""
    prices = np.array([price])
    forwards = np.array([forward])
    strikes = np.array([strike])
    expiries = np.array([expiry])
    option_signs = np.array([option_sign])
    discounts = np.array([1.0])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      results = newton_vol.implied_vol(forwards, strikes, expiries, discounts,
                                       prices, option_signs)
      self.evaluate(results)


if __name__ == '__main__':
  tf.test.main()
