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
"""Tests for vanilla_price."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


print(dir(tff))

@test_util.run_all_in_graph_and_eager_modes
class AmericanPrice(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods for the american pricing module."""

  @parameterized.parameters(
    (0.08, 0.2, 0.25,
     [0.03, 0.59, 3.52, 10.31, 20.0, 20.42, 11.25, 4.40, 1.12, 0.18],
     [0.031208, 0.581123, 3.506973, 10.305076, 20.0, 20.418007, 11.242258,
      4.377594, 1.102947, 0.179055]),
    (0.12, 0.2, 0.25,
     [0.03, 0.59, 3.51, 10.29, 20.0, 20.25, 11.15, 4.35, 1.11, 0.18],
     [0.031361, 0.578414, 3.488747, 10.279456, 20.0, 20.247196, 11.137532,
      4.335883, 1.092516, 0.177452]),
    (0.08, 0.4, 0.25,
     [1.07, 3.28, 7.41, 13.50, 21.23, 21.46, 13.93, 8.27, 4.52, 2.30],
     [1.062566, 3.276749, 7.401451, 13.493443, 21.227011, 21.458952, 13.919689,
      8.264548, 4.513709, 2.289232]),
    (0.08, 0.2, 0.5,
     [0.23, 1.39, 4.72, 10.96, 20.0, 20.98, 12.64, 6.37, 2.65, 0.92],
     [0.219648, 1.357535, 4.677300, 10.920374, 20.0, 20.972910, 12.613128,
      6.320176, 2.599509, 0.885091]),
  )
  def test_option_prices(self,
                         risk_free_rates,
                         volatilities,
                         expiries,
                         expected_prices,
                         fake_expected_prices):
    """Tests that the prices are correct."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = np.array([True] * 5 + [False] * 5)
    cost_of_carries = -0.04
    computed_prices = self.evaluate(
        tff.black_scholes.approximations.american_option.adesi_whaley(
          volatilities,
          strikes,
          expiries,
          risk_free_rates,
          cost_of_carries,
          is_call_options=is_call_options,
          spots=spots,
          dtype=tf.float32))
    fake_expected_prices = np.array(fake_expected_prices)
    expected_prices = np.array(expected_prices)
    self.assertArrayNear(fake_expected_prices, computed_prices, 1e-3)

  @parameterized.parameters(
    (0.08, 0.2, 0.25,
     [0.05, 0.85, 4.44, 11.66, 20.90, 20.00, 10.18, 3.54, 0.80, 0.12],
     [0.050095, 0.836672, 4.421259, 11.651242, 20.895369, 20.0, 10.176201,
      3.526063, 0.787715, 0.115339]),
    (0.12, 0.2, 0.25,
     [0.05, 0.84, 4.40, 11.55, 20.69, 20.00, 10.16, 3.53, 0.79, 0.12],
     [0.049612, 0.828424, 4.377595, 11.536527, 20.691483, 20.000000, 10.154543,
      3.507460, 0.783841, 0.115270]),
    (0.08, 0.4, 0.25,
     [1.29, 3.82, 8.35, 14.80, 22.72, 20.53, 12.93, 7.46, 3.96, 1.95],
     [1.284287, 3.814887, 8.340343, 14.788590, 22.709639, 20.523909, 12.918881,
      7.446249, 3.949303, 1.947205]),
    (0.08, 0.2, 0.5,
     [0.41, 2.18, 6.50, 13.42, 22.06, 20.00, 10.71, 4.77, 1.76, 0.55],
     [0.394618, 2.134961, 6.442985, 13.386993, 22.041672, 20.000000, 10.675407,
      4.723947, 1.724304, 0.527576]),
  )
  def test_option_prices_pos_b(self,
                         risk_free_rates,
                         volatilities,
                         expiries,
                         expected_prices,
                         fake_expected_prices):
    """Tests that the prices are correct."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = [True] * 5 + [False] * 5
    cost_of_carries = 0.04
    computed_prices = self.evaluate(
      tff.black_scholes.approximations.american_option.adesi_whaley(
        volatilities,
        strikes,
        expiries,
        risk_free_rates,
        cost_of_carries,
        spots=spots,
        is_call_options=is_call_options,
        dtype=tf.float32))

    fake_expected_prices = np.array(fake_expected_prices)
    expected_prices = np.array(expected_prices)
    self.assertArrayNear(fake_expected_prices, computed_prices, 1e-3)

  @parameterized.parameters(
    (0.08, 0.2, 0.25,
     [0.04, 0.70, 3.93, 10.81, 20.02, 20.00, 10.58, 3.93, 0.94, 0.15],
     [0.037441, 0.686889, 3.889322, 10.724886, 19.744852, 20.796013, 10.980298,
      3.993509, 0.948498, 0.147939]),
    (0.12, 0.2, 0.25,
     [0.04, 0.70, 3.90, 10.75, 20.0, 20.00, 10.53, 3.90, 0.93, 0.15],
     [0.037082, 0.680122, 3.850909, 10.619231, 19.551895, 20.796013, 10.963469,
      3.972912, 0.943664, 0.147720]),
    (0.08, 0.4, 0.25,
     [1.17, 3.53, 7.84, 14.08, 21.86, 20.93, 13.93, 7.84, 4.23, 2.12],
     [1.157728, 3.510153, 7.798741, 14.001305, 21.704327, 21.215557, 13.514194,
      7.896326, 4.249130, 2.126074]),
    (0.08, 0.2, 0.5,
     [0.30, 1.72, 5.48, 11.90, 20.34, 20.04, 11.48, 5.48, 2.15, 0.70],
     [0.28091, 1.661523, 5.362541, 11.690692, 19.886261, 21.584106, 12.109485,
      5.636985, 2.182215, 0.709079]),
  )
  def test_option_prices_futures_zero_b(self,
                                        risk_free_rates,
                                        volatilities,
                                        expiries,
                                        expected_prices,
                                        fake_expected_prices):
    """Tests that the prices are correct."""
    forwards = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
    strikes = np.array([100.0] * 10)
    is_call_options = [True] * 5 + [False] * 5
    cost_of_carries = 0.04
    computed_prices = self.evaluate(
      tff.black_scholes.approximations.american_option.adesi_whaley(
        volatilities,
        strikes,
        expiries,
        risk_free_rates,
        cost_of_carries,
        forwards=forwards,
        is_call_options=is_call_options,
        dtype=tf.float32))

    fake_expected_prices = np.array(fake_expected_prices)
    expected_prices = np.array(expected_prices)
    self.assertArrayNear(fake_expected_prices, computed_prices, 1e-3)

  @parameterized.parameters(
    (0.08, 0.2, 0.25,
     [20.0, 10.01, 3.22, 0.68, 0.10],
     [18.077206, 8.987599, 2.962302, 0.604106, 0.077939]),
    (0.12, 0.2, 0.25,
     [20.0, 10.0, 2.93, 0.58, 0.08],
     [17.098728, 8.118369, 2.470848, 0.454215, 0.052205]),
    (0.08, 0.4, 0.25,
     [20.25, 12.51, 7.10, 3.71, 1.81],
     [19.424828, 12.132558, 6.897156, 3.595204, 1.737887]),
    (0.08, 0.2, 0.5,
     [20.0, 10.23, 4.19, 1.45, 0.42],
     [16.541376, 8.632272, 3.588241, 1.192073, 0.324620]),
  )
  def test_option_prices_b_equal_r(self,
                                   risk_free_rates,
                                   volatilities,
                                   expiries,
                                   expected_prices,
                                   fake_expected_prices):
    """Tests that the prices are correct."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    strikes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    cost_of_carries = risk_free_rates
    is_call_options = False
    computed_prices = self.evaluate(
        tff.black_scholes.approximations.american_option.adesi_whaley(
          volatilities,
          strikes,
          expiries,
          risk_free_rates,
          cost_of_carries,
          spots=spots,
          is_call_options=is_call_options,
          dtype=tf.float32))

    fake_expected_prices = np.array(fake_expected_prices)
    expected_prices = np.array(expected_prices)

    self.assertArrayNear(fake_expected_prices, computed_prices, 1e-3)


if __name__ == '__main__':
  tf.test.main()
