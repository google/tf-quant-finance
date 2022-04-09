# Copyright 2020 Google LLC
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
"""Tests for the regression Monte Carlo algorithm."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.lsm_algorithm import lsm
from tf_quant_finance.experimental.lsm_algorithm import payoff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class LsmTest(tf.test.TestCase):

  def setUp(self):
    """Sets `samples` as in the Longstaff-Schwartz paper."""
    super(LsmTest, self).setUp()
    # See Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach.
    samples = [[1.0, 1.09, 1.08, 1.34],
               [1.0, 1.16, 1.26, 1.54],
               [1.0, 1.22, 1.07, 1.03],
               [1.0, 0.93, 0.97, 0.92],
               [1.0, 1.11, 1.56, 1.52],
               [1.0, 0.76, 0.77, 0.90],
               [1.0, 0.92, 0.84, 1.01],
               [1.0, 0.88, 1.22, 1.34]]
    # Expand dims to reflect that `samples` represent sample paths of
    # a 1-dimensional process
    self.samples = np.expand_dims(samples, -1)
    # Interest rates between exercise times
    interest_rates = [0.06, 0.06, 0.06]
    # Corresponding discount factors
    self.discount_factors = np.exp(-np.cumsum(interest_rates))

  def test_loop_condition(self):
    """Tests that the loop will stop countdown at zero and not before."""
    self.assertTrue(lsm.lsm_loop_cond(1, None))
    self.assertFalse(lsm.lsm_loop_cond(0, None))

  def test_continuation_value(self):
    """Tests continuation value returns the discounted sum of later payoffs."""
    exercise_index = 2
    for dtype in (np.float32, np.float64):
      discount_factors = tf.constant(
          [[1.0, 0.9, 0.8, 0.7, 0.6]], dtype=dtype)
      cashflow = tf.ones(shape=[10, 5, 4], dtype=dtype)
      continuation_value = lsm.continuation_value_fn(cashflow,
                                                     discount_factors,
                                                     exercise_index)
      expected_continuation = 1.625 * np.ones([10, 5])
      self.assertAllClose(
          continuation_value, expected_continuation, rtol=1e-8, atol=1e-8)

  def test_expected_continuation(self):
    """Tests that expected continuation works in V=1 case.

    In particular this verifies that the regression done to get the expected
    continuation value is performed on those elements which have a positive
    exercise value.
    """
    for dtype in (np.float32, np.float64):
      a = tf.range(start=-2, limit=3, delta=1, dtype=dtype)
      design = tf.concat([a, a], axis=0)
      design = tf.concat([[tf.ones_like(design), design]], axis=1)

      # These values ensure that the expected continuation value is `(1,...,1).`
      exercise_now = tf.expand_dims(
          tf.concat([tf.ones_like(a), tf.zeros_like(a)], axis=0), -1)
      cashflow = tf.expand_dims(
          tf.concat([tf.ones_like(a), -tf.ones_like(a)], axis=0), -1)

      expected_exercise = lsm.expected_exercise_fn(
          design, cashflow, exercise_now)
      self.assertAllClose(expected_exercise, tf.ones_like(cashflow))

  def test_european_option_put(self):
    """Tests that LSM price of European put option is computed as expected."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm.make_polynomial_basis(2)
    for dtype in (np.float32, np.float64):
      payoff_fn = payoff.make_basket_put_payoff([1.1], dtype=dtype)
      # Option price
      european_put_price = lsm.least_square_mc(
          self.samples, [3], payoff_fn, basis_fn,
          discount_factors=[self.discount_factors[-1]], dtype=dtype)
      self.assertAllClose(european_put_price, [0.0564],
                          rtol=1e-4, atol=1e-4)

  def test_american_option_put(self):
    """Tests that LSM price of American put option is computed as expected."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    basis_fn = lsm.make_polynomial_basis(2)
    for dtype in (np.float32, np.float64):
      payoff_fn = payoff.make_basket_put_payoff([1.1], dtype=dtype)
      # Option price
      american_put_price = lsm.least_square_mc(
          self.samples, [1, 2, 3], payoff_fn, basis_fn,
          discount_factors=self.discount_factors, dtype=dtype)
      self.assertAllClose(american_put_price, [0.1144],
                          rtol=1e-4, atol=1e-4)

  def test_american_basket_option_put(self):
    """Tests the LSM price of American Basket put option."""
    # This is the same example as in Section 1 of
    # Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
    # simulation: a simple least-squares approach. The review of financial
    # studies, 14(1), pp.113-147.
    # This is the minimum number of basis functions for the tests to pass.
    basis_fn = lsm.make_polynomial_basis(10)
    exercise_times = [1, 2, 3]
    dtype = np.float64
    payoff_fn = payoff.make_basket_put_payoff([1.1, 1.2, 1.3], dtype=dtype)
    # Create a 2-d process which is simply follows the `samples` paths:
    samples = tf.convert_to_tensor(self.samples, dtype=dtype)
    samples_2d = tf.concat([samples, samples], -1)
    # Price American basket option
    american_basket_put_price = lsm.least_square_mc(
        samples_2d, exercise_times, payoff_fn, basis_fn,
        discount_factors=self.discount_factors, dtype=dtype)
    # Since the marginal processes of `samples_2d` are 100% correlated, the
    # price should be the same as of the American option computed for
    # `samples`
    american_put_price = lsm.least_square_mc(
        self.samples, exercise_times, payoff_fn, basis_fn,
        discount_factors=self.discount_factors, dtype=dtype)
    self.assertAllClose(american_basket_put_price, american_put_price,
                        rtol=1e-4, atol=1e-4)
    self.assertAllEqual(american_basket_put_price.shape, [3])

if __name__ == '__main__':
  tf.test.main()
