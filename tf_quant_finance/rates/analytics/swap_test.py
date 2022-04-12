# Copyright 2021 Google LLC
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
"""Tests for swap module."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class SwapTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoBatch',
          'notional': 10000,
          'forward_prices': [110, 120, 140],
          'spots': 100,
          'dividends': [1, 1, 1],
          'expected_pv': [1000.01, 909.1, 1666.675],

      }, {
          'testcase_name': 'WithBatch',
          'notional': 10000,
          'forward_prices': [[110, 120, 140], [210, 220, 240]],
          'spots': [100, 200],
          'dividends': [[1, 1, 1], [2, 2, 2]],
          'expected_pv': [[1000.01, 909.1, 1666.675], [500.01, 476.2, 909.1]],

      })
  def test_equity_leg_cashflows(
      self, notional, forward_prices, spots, dividends, expected_pv):
    dtype = tf.float64
    actual_pv = self.evaluate(
        tff.rates.analytics.swap.equity_leg_cashflows(
            forward_prices=forward_prices,
            spots=spots,
            notional=notional,
            dividends=dividends,
            dtype=dtype))
    np.testing.assert_allclose(expected_pv, actual_pv)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoBatch',
          'notional': 1000,
          'coupon_rates': 0.1,
          'daycount_fractions': [1, 1, 1],
          'expected_pv': [100.0, 100.0, 100.0],

      }, {
          'testcase_name': 'WithBatch',
          'notional': 1000,
          'coupon_rates': [[0.1, 0.1, 0.1], [0.02, 0.12, 0.14]],
          'daycount_fractions': [[1, 1, 1], [1, 2, 1]],
          'expected_pv': [[100.0, 100.0, 100.0], [20.0, 240.0, 140.0]],

      })
  def test_rate_leg_cashflows(
      self, notional, coupon_rates, daycount_fractions, expected_pv):
    dtype = tf.float64
    actual_pv = self.evaluate(
        tff.rates.analytics.swap.rate_leg_cashflows(
            coupon_rates=coupon_rates,
            daycount_fractions=daycount_fractions,
            notional=notional,
            dtype=dtype))
    np.testing.assert_allclose(expected_pv, actual_pv)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoBatch',
          'pay_leg_cashflows': [100, 100, 100],
          'receive_leg_cashflows': [200, 250, 300, 300],
          'pay_leg_discount_factors': [0.95, 0.9, 0.8],
          'receive_leg_discount_factors': [0.95, 0.9, 0.8, 0.75],
          'expected_pv': 615.0,

      }, {
          'testcase_name': 'WithBatch',
          'pay_leg_cashflows': [[100, 100, 100], [200, 250, 300]],
          'receive_leg_cashflows': [[200, 250, 300, 300], [100, 100, 100, 100]],
          'pay_leg_discount_factors': [[0.95, 0.9, 0.8], [0.9, 0.85, 0.8]],
          'receive_leg_discount_factors': [[0.95, 0.9, 0.8, 0.75],
                                           [0.9, 0.85, 0.8, 0.75]],
          'expected_pv': [615.0, -302.5],

      })
  def test_swap_price(
      self, pay_leg_cashflows, receive_leg_cashflows,
      pay_leg_discount_factors, receive_leg_discount_factors, expected_pv):
    dtype = tf.float64
    actual_pv = self.evaluate(
        tff.rates.analytics.swap.swap_price(
            pay_leg_cashflows=pay_leg_cashflows,
            receive_leg_cashflows=receive_leg_cashflows,
            pay_leg_discount_factors=pay_leg_discount_factors,
            receive_leg_discount_factors=receive_leg_discount_factors,
            dtype=dtype))
    np.testing.assert_allclose(expected_pv, actual_pv)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoBatch',
          'pay_leg_coupon_rates': 0.1,
          'receive_leg_coupon_rates': [0.1, 0.2, 0.05],
          'notional': 1000,
          'pay_leg_daycount_fractions': 0.5,
          'receive_leg_daycount_fractions': 0.5,
          'discount_factors': [0.95, 0.9, 0.85],
          'expected_pv': 23.75,

      }, {
          'testcase_name': 'WithBatch',
          'pay_leg_coupon_rates': [[0.1], [0.15]],
          'receive_leg_coupon_rates': [[0.1, 0.2, 0.05], [0.1, 0.05, 0.2]],
          'notional': 1000,
          'pay_leg_daycount_fractions': 0.5,
          'receive_leg_daycount_fractions': [[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]],
          'discount_factors': [[0.95, 0.9, 0.85], [0.98, 0.92, 0.88]],
          'expected_pv': [23.75, -40.7],

      })
  def test_ir_swap_price(
      self, pay_leg_coupon_rates, receive_leg_coupon_rates, notional,
      pay_leg_daycount_fractions, receive_leg_daycount_fractions,
      discount_factors, expected_pv):
    dtype = tf.float64
    actual_pv = self.evaluate(
        tff.rates.analytics.swap.ir_swap_price(
            pay_leg_coupon_rates=pay_leg_coupon_rates,
            receive_leg_coupon_rates=receive_leg_coupon_rates,
            pay_leg_notional=notional,
            receive_leg_notional=notional,
            pay_leg_daycount_fractions=pay_leg_daycount_fractions,
            receive_leg_daycount_fractions=receive_leg_daycount_fractions,
            pay_leg_discount_factors=discount_factors,
            receive_leg_discount_factors=discount_factors,
            dtype=dtype))
    np.testing.assert_allclose(expected_pv, actual_pv)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoBatch',
          'floating_leg_start_times': [0.5, 1.0, 1.5],
          'floating_leg_end_times': [1.0, 1.5, 2.0],
          'fixed_leg_payment_times': [1.0, 1.5, 2.0],
          'fixed_leg_daycount_fractions': [0.5, 0.5, 0.5],
          'reference_rate_fn': lambda x: 0.01,
          'expected_values': [0.010025041718802, 1.477680223329493],
      }, {
          'testcase_name':
              'WithBatch',
          'floating_leg_start_times': [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]],
          'floating_leg_end_times': [[1.0, 1.5, 2.0], [1.0, 1.5, 2.0]],
          'fixed_leg_payment_times': [[1.0, 1.5, 2.0], [1.0, 1.5, 2.0]],
          'fixed_leg_daycount_fractions': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
          'reference_rate_fn':
              lambda x: 0.01,
          'expected_values': [[0.010025041718802, 0.010025041718802],
                              [1.477680223329493, 1.477680223329493]],
      })
  def test_ir_swap_par_rate_and_annuity(self, floating_leg_start_times,
                                        floating_leg_end_times,
                                        fixed_leg_payment_times,
                                        fixed_leg_daycount_fractions,
                                        reference_rate_fn, expected_values):
    dtype = tf.float64
    actual_parrate, actual_annuity = self.evaluate(
        tff.rates.analytics.swap.ir_swap_par_rate_and_annuity(
            floating_leg_start_times,
            floating_leg_end_times,
            fixed_leg_payment_times,
            fixed_leg_daycount_fractions,
            reference_rate_fn,
            dtype=dtype))
    np.testing.assert_allclose(expected_values[0], actual_parrate)
    np.testing.assert_allclose(expected_values[1], actual_annuity)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoBatch',
          'rate_leg_coupon_rates': [0.1, 0.2, 0.05],
          'forward_prices': [110, 120, 140, 150],
          'spots': 100,
          'notional': 1000,
          'daycount_fractions': [0.5, 0.5, 0.5],
          'rate_leg_discount_factors': [0.95, 0.9, 0.85],
          'equity_leg_discount_factors': [0.95, 0.9, 0.85, 0.8],
          'is_equity_receiver': None,
          'expected_pv': 216.87770563,

      }, {
          'testcase_name': 'WithBatch',
          'rate_leg_coupon_rates': [[0.1, 0.2, 0.05], [0.1, 0.05, 0.2]],
          'forward_prices': [[110, 120, 140, 150], [210, 220, 240, 0]],
          'spots': [100, 200],
          'notional': 1000,
          'daycount_fractions': [[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]],
          'rate_leg_discount_factors': [[0.95, 0.9, 0.85], [0.98, 0.92, 0.88]],
          'equity_leg_discount_factors': [[0.95, 0.9, 0.85, 0.8],
                                          [0.98, 0.92, 0.88, 0.0]],
          'is_equity_receiver': [True, False],
          'expected_pv': [216.87770563, -5.00952381],

      })
  def test_equity_swap_price(
      self, rate_leg_coupon_rates, forward_prices, spots, notional,
      daycount_fractions,
      rate_leg_discount_factors, equity_leg_discount_factors,
      is_equity_receiver, expected_pv):
    dtype = tf.float64
    actual_pv = self.evaluate(
        tff.rates.analytics.swap.equity_swap_price(
            rate_leg_coupon_rates=rate_leg_coupon_rates,
            equity_leg_forward_prices=forward_prices,
            equity_leg_spots=spots,
            rate_leg_notional=notional,
            equity_leg_notional=notional,
            rate_leg_daycount_fractions=daycount_fractions,
            rate_leg_discount_factors=rate_leg_discount_factors,
            equity_leg_discount_factors=equity_leg_discount_factors,
            is_equity_receiver=is_equity_receiver,
            dtype=dtype))
    np.testing.assert_allclose(expected_pv, actual_pv)


if __name__ == '__main__':
  tf.test.main()
