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
"""Tests for vanilla.black_scholes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_quant_finance.volatility import black_scholes
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class BlackScholesTest(tf.test.TestCase):
  """Tests for methods in the Black Scholes pricing module."""

  def test_option_prices(self):
    """Tests that the BS prices are correct."""
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = self.evaluate(
        black_scholes.option_price(forwards, strikes, volatilities, expiries))
    expected_prices = np.array(
        [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933])
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_zero_vol(self):
    """Tests that zero volatility is handled correctly."""
    # If the volatility is zero, the option's value should be correct.
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.1, 0.9, 1.1, 0.9])
    volatilities = np.array([0.0, 0.0, 0.0, 0.0])
    expiries = 1.0
    is_call_options = np.array([True, True, False, False])
    expected_prices = np.array([0.0, 0.1, 0.1, 0.0])
    computed_prices = self.evaluate(
        black_scholes.option_price(
            forwards,
            strikes,
            volatilities,
            expiries,
            is_call_options=is_call_options))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_zero_expiry(self):
    """Tests that zero expiry is correctly handled."""
    # If the expiry is zero, the option's value should be correct.
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.1, 0.9, 1.1, 0.9])
    volatilities = np.array([0.1, 0.2, 0.5, 0.9])
    expiries = 0.0
    is_call_options = np.array([True, True, False, False])
    expected_prices = np.array([0.0, 0.1, 0.1, 0.0])
    computed_prices = self.evaluate(
        black_scholes.option_price(
            forwards,
            strikes,
            volatilities,
            expiries,
            is_call_options=is_call_options))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_long_expiry_calls(self):
    """Tests that very long expiry call option behaves like the asset."""
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([1.1, 0.9, 1.1, 0.9])
    volatilities = np.array([0.1, 0.2, 0.5, 0.9])
    expiries = 1e10
    expected_prices = forwards
    computed_prices = self.evaluate(
        black_scholes.option_price(forwards, strikes, volatilities, expiries))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_long_expiry_puts(self):
    """Tests that very long expiry put option is worth the strike."""
    forwards = np.array([1.0, 1.0, 1.0, 1.0])
    strikes = np.array([0.1, 10.0, 3.0, 0.0001])
    volatilities = np.array([0.1, 0.2, 0.5, 0.9])
    expiries = 1e10
    expected_prices = strikes
    computed_prices = self.evaluate(
        black_scholes.option_price(
            forwards, strikes, volatilities, expiries, is_call_options=False))
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)

  def test_price_vol_and_expiry_scaling(self):
    """Tests that the price is invariant under vol->k vol, T->T/k**2."""
    np.random.seed(1234)
    n = 20
    forwards = np.exp(np.random.randn(n))
    volatilities = np.exp(np.random.randn(n) / 2)
    strikes = np.exp(np.random.randn(n))
    expiries = np.exp(np.random.randn(n))
    scaling = 5.0
    base_prices = self.evaluate(
        black_scholes.option_price(forwards, strikes, volatilities, expiries))
    scaled_prices = self.evaluate(
        black_scholes.option_price(forwards, strikes, volatilities * scaling,
                                   expiries / scaling / scaling))

    self.assertArrayNear(base_prices, scaled_prices, 1e-10)


if __name__ == '__main__':
  tf.test.main()
