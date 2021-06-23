# Lint as: python3
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
"""Tests for american_option."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

adesi_whaley = tff.black_scholes.approximations.adesi_whaley
bjerksund_stensland = tff.black_scholes.approximations.bjerksund_stensland


@test_util.run_all_in_graph_and_eager_modes
class AmericanPrice(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods for the american pricing module."""

  @parameterized.parameters(
      (0.08, 0.12, 0.2, 0.25,
       [0.03, 0.59, 3.52, 10.31, 20.0, 20.42, 11.25, 4.40, 1.12, 0.18]),
      (0.12, 0.16, 0.2, 0.25,
       [0.03, 0.59, 3.51, 10.29, 20.0, 20.25, 11.15, 4.35, 1.11, 0.18]),
      (0.08, 0.12, 0.4, 0.25,
       [1.07, 3.28, 7.41, 13.50, 21.23, 21.46, 13.93, 8.27, 4.52, 2.30]),
      (0.08, 0.12, 0.2, 0.5,
       [0.23, 1.39, 4.72, 10.96, 20.0, 20.98, 12.64, 6.37, 2.65, 0.92]),
  )
  def test_option_prices_neg_carries(self, discount_rates, dividends,
                                     volatilities, expiries, expected_prices):
    """Tests the Baron-Adesi Whaley prices for negative cost_of_carries."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = np.array([True] * 5 + [False] * 5)
    computed_prices, converged, failed = adesi_whaley(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        dividend_rates=dividends,
        is_call_options=is_call_options,
        spots=spots,
        dtype=tf.float64)
    expected_prices = np.array(expected_prices)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3)
    with self.subTest(name='AllConverged'):
      self.assertAllEqual(converged, tf.ones_like(computed_prices))
    with self.subTest(name='NonFailed'):
      self.assertAllEqual(failed, tf.zeros_like(computed_prices))

  @parameterized.parameters(
      (0.08, 0.04, 0.2, 0.25,
       [0.05, 0.85, 4.44, 11.66, 20.90, 20.00, 10.18, 3.54, 0.80, 0.12]),
      (0.12, 0.08, 0.2, 0.25,
       [0.05, 0.84, 4.40, 11.55, 20.69, 20.00, 10.16, 3.53, 0.79, 0.12]),
      (0.08, 0.04, 0.4, 0.25,
       [1.29, 3.82, 8.35, 14.80, 22.72, 20.53, 12.93, 7.46, 3.96, 1.95]),
      (0.08, 0.04, 0.2, 0.5,
       [0.41, 2.18, 6.50, 13.42, 22.06, 20.00, 10.71, 4.77, 1.76, 0.55]),
  )
  def test_option_prices_pos_carries(self, discount_rates, dividends,
                                     volatilities, expiries, expected_prices):
    """Tests the Baron-Adesi Whaley prices for positive cost_of_carries."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = [True] * 5 + [False] * 5
    computed_prices, converged, failed = adesi_whaley(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        dividend_rates=dividends,
        spots=spots,
        is_call_options=is_call_options,
        dtype=tf.float64)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3)
    with self.subTest(name='AllConverged'):
      self.assertAllEqual(converged, tf.ones_like(computed_prices))
    with self.subTest(name='NonFailed'):
      self.assertAllEqual(failed, tf.zeros_like(computed_prices))

  @parameterized.parameters(
      (0.08, 0.2, 0.25,
       [0.04, 0.70, 3.93, 10.81, 20.02, 20.00, 10.58, 3.93, 0.94, 0.15]),
      (0.12, 0.2, 0.25,
       [0.04, 0.70, 3.90, 10.75, 20.0, 20.00, 10.53, 3.90, 0.93, 0.15]),
      (0.08, 0.4, 0.25,
       [1.17, 3.53, 7.84, 14.08, 21.86, 20.93, 13.39, 7.84, 4.23, 2.12]),
      (0.08, 0.2, 0.5,
       [0.30, 1.72, 5.48, 11.90, 20.34, 20.04, 11.48, 5.48, 2.15, 0.70]),
  )
  def test_option_prices_zero_cost_of_carries(self,
                                              discount_rates,
                                              volatilities,
                                              expiries,
                                              expected_prices):
    """Tests the Baron-Adesi Whaley prices when cost_of_carries is zero."""
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = [True] * 5 + [False] * 5
    dividends = discount_rates
    computed_prices, converged, failed = adesi_whaley(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        dividend_rates=dividends,
        forwards=forwards,
        is_call_options=is_call_options,
        dtype=tf.float64)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3)
    with self.subTest(name='AllConverged'):
      self.assertAllEqual(converged, tf.ones_like(computed_prices))
    with self.subTest(name='NonFailed'):
      self.assertAllEqual(failed, tf.zeros_like(computed_prices))

  @parameterized.parameters(
      (tf.float64, 0.08, 0.2, 0.25, [20.0, 10.01, 3.22, 0.68, 0.10]),
      (tf.float64, 0.12, 0.2, 0.25, [20.0, 10.0, 2.93, 0.58, 0.08]),
      (tf.float64, 0.08, 0.4, 0.25, [20.25, 12.51, 7.10, 3.71, 1.81]),
      (tf.float64, 0.08, 0.2, 0.5, [20.0, 10.23, 4.19, 1.45, 0.42]),
      (tf.float32, 0.08, 0.2, 0.25, [20.0, 10.01, 3.22, 0.68, 0.10]),
      (tf.float32, 0.12, 0.2, 0.25, [20.0, 10.0, 2.93, 0.58, 0.08]),
      (tf.float32, 0.08, 0.4, 0.25, [20.25, 12.51, 7.10, 3.71, 1.81]),
      (tf.float32, 0.08, 0.2, 0.5, [20.0, 10.23, 4.19, 1.45, 0.42]),
  )
  def test_option_prices_no_dividends(self, dtype, discount_rates, volatilities,
                                      expiries, expected_prices):
    """Tests the Baron-Adesi Whaley prices when no dividends are supplied."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    strikes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    is_call_options = False
    computed_prices, converged, failed = adesi_whaley(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        spots=spots,
        is_call_options=is_call_options,
        tolerance=1e-5,  # float32 does not converge to tolerance 1e-8
        dtype=dtype)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3)
    with self.subTest(name='AllConverged'):
      self.assertAllEqual(converged, tf.ones_like(computed_prices))
    with self.subTest(name='NonFailed'):
      self.assertAllEqual(failed, tf.zeros_like(computed_prices))

  def test_option_prices_zero_discount_rates(self):
    """Tests Baron-Adesi Whaley prices with zero discount."""
    dtype = tf.float64
    computed_prices, converged, failed = adesi_whaley(
        volatilities=[0.2], strikes=[104, 90],
        expiries=[0.1, 1.0], spots=[100],
        tolerance=1e-08,
        is_call_options=[True, False],
        discount_rates=[0.0, 0.0],
        dtype=dtype)
    # Computed using tff.black_scholes.option_price_binomial
    expected_prices = [1.05226366, 3.58957787]
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=1e-4, atol=5e-4)
    with self.subTest(name='AllConverged'):
      self.assertAllEqual(converged, tf.ones_like(computed_prices))
    with self.subTest(name='NonFailed'):
      self.assertAllEqual(failed, tf.zeros_like(computed_prices))

  def test_option_prices_all_call_options(self):
    """Tests Baron-Adesi Whaley call prices with zero discount."""
    dtype = tf.float64
    computed_prices, converged, failed = adesi_whaley(
        volatilities=[0.2], strikes=[104, 90],
        expiries=[0.1, 1.0], spots=[100],
        tolerance=1e-08,
        discount_rates=[0.0, 0.0],
        dtype=dtype)
    # Computed using tff.black_scholes.option_price_binomial
    expected_prices = [1.05226366, 13.58957787]
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=1e-4, atol=5e-4)
    with self.subTest(name='AllConverged'):
      self.assertAllEqual(converged, tf.ones_like(computed_prices))
    with self.subTest(name='NonFailed'):
      self.assertAllEqual(failed, tf.zeros_like(computed_prices))
  
  @parameterized.parameters(
      (0.08, 0.2, 0.25,
       [0.03, 0.57, 3.49, 10.32, 20.0, 20.41, 11.25, 4.40, 1.12, 0.18]),
      (0.12, 0.2, 0.25,
       [0.03, 0.57, 3.46, 10.29, 20.0, 20.22, 11.14, 4.35, 1.11, 0.18]),
      (0.08, 0.4, 0.25,
       [1.05, 3.25, 7.37, 13.47, 21.23, 21.44, 13.91, 8.27, 4.52, 2.29]),
      (0.08, 0.2, 0.5,
       [0.21, 1.34, 4.65, 10.94, 20.0, 20.95, 12.63, 6.37, 2.65, 0.92]),
  )
  def test_bs1993_prices_neg_carries(self, discount_rates, volatilities,
                                     expiries, expected_prices):
    """Tests Bjerksund Stensland 1993 prices for negative cost_of_carries."""
    cost_of_carries = np.array([-0.04] * 10)
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = np.array([True] * 5 + [False] * 5)
    computed_prices = bjerksund_stensland(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        cost_of_carries=cost_of_carries,
        forwards=forwards,
        is_call_options=is_call_options,
        modified_boundary=False,
        dtype=tf.float64)
    expected_prices = np.array(expected_prices)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3,
                          msg='Failed: Bjerksund Stensland 1993 negative carries test.')
  
  @parameterized.parameters(
      (0.08, 0.2, 0.25,
       [0.05, 0.85, 4.44, 11.66, 20.90, 20.00, 10.19, 3.51, 0.78, 0.11]),
      (0.12, 0.2, 0.25,
       [0.05, 0.84, 4.40, 11.55, 20.69, 20.00, 10.17, 3.49, 0.77, 0.11]),
      (0.08, 0.4, 0.25,
       [1.29, 3.82, 8.35, 14.80, 22.71, 20.53, 12.91, 7.42, 3.93, 1.93]),
      (0.08, 0.2, 0.5,
       [0.41, 2.18, 6.50, 13.42, 22.06, 20.00, 10.70, 4.70, 1.71, 0.52]),
  )
  def test_bs1993_prices_pos_carries(self, discount_rates, volatilities,
                                     expiries, expected_prices):
    """Tests Bjerksund Stensland 1993 prices for positive cost_of_carries."""
    cost_of_carries = np.array([0.04] * 10)
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = [True] * 5 + [False] * 5
    computed_prices = bjerksund_stensland(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        cost_of_carries=cost_of_carries,
        forwards=forwards,
        is_call_options=is_call_options,
        modified_boundary=False,
        dtype=tf.float64)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3,
                          msg='Failed: Bjerksund Stensland 1993 positive carries test.')
  
  @parameterized.parameters(
      (0.08, 0.08, 0.2, 0.25,
       [20.00, 10.01, 3.16, 0.65, 0.09]),
      (0.12, 0.12, 0.2, 0.25,
       [20.00, 10.00, 2.86, 0.54, 0.07]),
      (0.08, 0.08, 0.4, 0.25,
       [20.28, 12.48, 7.04, 3.66, 1.77]),
      (0.08, 0.08, 0.2, 0.5,
       [20.00, 10.24, 4.11, 1.37, 0.39]),
  )
  def test_bs1993_prices_carries_equal_rate(self, discount_rates, cost_of_carries,
                                      volatilities, expiries, expected_prices):
    """Tests Bjerksund Stensland 1993 prices for zero cost_of_carries."""
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    strikes = np.array([100.0] * 5)
    is_call_options = np.array([False] * 5)
    computed_prices = bjerksund_stensland(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        cost_of_carries=cost_of_carries,
        forwards=forwards,
        is_call_options=is_call_options,
        modified_boundary=False,
        dtype=tf.float64)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3,
                          msg='Failed: Bjerksund Stensland 1993 zero carries test.')
  
  @parameterized.parameters(
      (0.08, -0.04, 0.2, 3.0,
       [2.30, 4.71, 8.44, 13.74, 20.85, 25.61, 20.04, 15.47, 11.78, 8.87]),
      (0.08, 0.0, 0.2, 3.0,
       [3.95, 7.20, 11.64, 17.24, 23.93, 22.12, 16.14, 11.64, 8.31, 5.89]),
      #(0.08, 0.04, 0.4, 3.0,
      # [6.88, 11.49, 17.21, 23.84, 31.16, 20.32, 13.43, 8.86, 5.83, 3.83]),      # Results are suspect, see PR #54
  )
  def test_bs1993_prices_long_term1(self, discount_rates, cost_of_carries,
                                      volatilities, expiries, expected_prices):
    """Tests Bjerksund Stensland 1993 prices for long-term options."""
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = [True] * 5 + [False] * 5
    computed_prices = bjerksund_stensland(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        cost_of_carries=cost_of_carries,
        forwards=forwards,
        is_call_options=is_call_options,
        modified_boundary=False,
        dtype=tf.float64)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3,
                          msg='Failed: Bjerksund Stensland 1993 long-term 1 test.')
  
  @parameterized.parameters(
      (0.08, 0.08, 0.2, 3.0,
       [20.00, 11.67, 6.90, 4.12, 2.48])
  )
  def test_bs1993_prices_long_term2(self, discount_rates, cost_of_carries,
                                      volatilities, expiries, expected_prices):
    """Tests Bjerksund Stensland 1993 prices for long-term options."""
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    strikes = np.array([100.0] * 5)
    is_call_options = np.array([False] * 5)
    computed_prices = bjerksund_stensland(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        discount_rates=discount_rates,
        cost_of_carries=cost_of_carries,
        forwards=forwards,
        is_call_options=is_call_options,
        modified_boundary=False,
        dtype=tf.float64)
    with self.subTest(name='ExpectedPrices'):
      self.assertAllClose(expected_prices, computed_prices,
                          rtol=5e-3, atol=5e-3,
                          msg='Failed: Bjerksund Stensland 1993 long-term 2 test.')

if __name__ == '__main__':
  tf.test.main()
