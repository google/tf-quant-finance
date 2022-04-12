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
"""Tests for Cox Ross Rubinstein Binomial tree method."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class BinomialModelPrice(parameterized.TestCase, tf.test.TestCase):
  """Tests for Binomial tree method prices."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'ZeroRank',
          'spots': 3.0,
          'strikes': 3.0,
          'volatilities': 0.32,
          'is_call_options': False,
          'is_american': True,
          'discount_rates': 0.035,
          'dividend_rates': 0.07,
          'expiries': 1.0,
          'expected': 0.41299509,
          'dtype': np.float64
      }, {
          'testcase_name': 'BatchShapeWithBroadcast',
          'spots': [1.0, 2.0, 3.0, 4.0, 5.0],
          'strikes': [3.0, 3.0, 3.0, 3.0, 3.0],
          'volatilities': [0.1, 0.22, 0.32, 0.01, 0.4],
          'is_call_options': [True, True, False, False, False],
          'is_american': [False, True, True, False, True],
          'discount_rates': 0.035,
          'dividend_rates': [0.02, 0.0, 0.07, 0.01, 0.0],
          'expiries': 1.0,
          'expected': [0.0, 0.0098847, 0.41299509, 0.0, 0.06046989],
          'dtype': np.float64
      }, {
          'testcase_name': 'BatchRank1',
          'spots': [1.0, 2.0, 3.0, 4.0, 5.0],
          'strikes': [1.0, 2.0, 3.0, 3.0, 5.0],
          'volatilities': [0.1, 0.2, 0.3, 0.01, 0.4],
          'is_call_options': [True, True, False, False, False],
          'is_american': [False, True, True, False, True],
          'discount_rates': [0.035, 0.01, 0.1, 0.01, 0.0],
          'dividend_rates': [0.02, 0.0, 0.07, 0.01, 0.0],
          'expiries': [0.5, 1.0, 1.0, 0.1, 2.0],
          'expected': [0.03160387, 0.1682701, 0.30367994, 0.0, 1.11073385],
          'dtype': np.float32
      }, {
          'testcase_name': 'BatchRank2',
          'spots': [[1.0, 2.0, 3.0, 4.0, 5.0],
                    [1.5, 2.5, 3.5, 4.5, 5.5]],
          'strikes': [[1.0, 2.0, 3.0, 3.0, 5.0],
                      [1.2, 2.2, 3.2, 3.2, 5.2]],
          'volatilities': [[0.1, 0.2, 0.3, 0.01, 0.4],
                           [0.15, 0.25, 0.35, 0.02, 0.35]],
          'is_call_options': [[True, True, False, False, False],
                              [False, True, False, True, False]],
          'is_american': [[False, True, True, False, True],
                          [True, True, False, False, True]],
          'discount_rates': [[0.035, 0.01, 0.1, 0.01, 0.0],
                             [0.03, 0.02, 0.05, 0.02, 0.01]],
          'dividend_rates': [[0.02, 0.0, 0.07, 0.01, 0.0],
                             [0.01, 0.01, 0.07, 0.01, 0.0]],
          'expiries': [[0.5, 1.0, 1.0, 0.1, 2.0],
                       [1.5, 1.5, 1.0, 0.5, 2.0]],
          'expected': [[0.031603, 0.16827, 0.303679, 0.0, 1.110733],
                       [0.009376, 0.472969, 0.337524, 1.309396, 0.856245]],
          'dtype': np.float32
      })
  def test_option_prices(self, spots, strikes, volatilities,
                         is_call_options, is_american, discount_rates,
                         dividend_rates, expiries, expected, dtype):
    """Tests that the BS prices are correct."""
    spots = tf.convert_to_tensor(spots, dtype=dtype)
    strikes = tf.convert_to_tensor(strikes, dtype=dtype)
    volatilities = tf.convert_to_tensor(volatilities, dtype=dtype)
    is_call_options = tf.convert_to_tensor(is_call_options)
    is_american = tf.convert_to_tensor(is_american)
    discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype)
    dividend_rates = tf.convert_to_tensor(dividend_rates, dtype=dtype)
    expiries = tf.convert_to_tensor(expiries, dtype=dtype)

    prices = self.evaluate(
        tff.black_scholes.option_price_binomial(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            spots=spots,
            discount_rates=discount_rates,
            dividend_rates=dividend_rates,
            is_call_options=is_call_options,
            is_american=is_american,
            dtype=dtype))
    expected_prices = np.array(expected)
    self.assertAllClose(expected_prices, prices, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  tf.test.main()
