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
"""Tests for implied_volatility.approx_implied_vol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import nomisma_quant_finance.implied_volatility.approx_implied_vol as approx_implied_vol
import nomisma_quant_finance.vanilla.black_scholes as black_scholes
from google3.testing.pybase import parameterized
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class ApproxImpliedVolTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for methods in implied_vol module."""

  def test_polya_implied_vol(self):
    """Basic test of the implied vol calculation."""
    np.random.seed(6589)
    n = 100
    volatilities = np.exp(np.random.randn(n) / 2)
    forwards = np.exp(np.random.randn(n))
    strikes = forwards * (1 + (np.random.rand(n) - 0.5) * 0.2)
    expiries = np.exp(np.random.randn(n))
    prices = self.evaluate(
        black_scholes.option_price(forwards, strikes, volatilities, expiries))

    implied_vols = self.evaluate(
        approx_implied_vol.polya(prices, forwards, strikes, expiries))
    self.assertArrayNear(volatilities, implied_vols, 0.6)

  def test_polya_implied_vol_validate(self):
    """Test the Radiocic-Polya approx doesn't raise where it shouldn't."""
    np.random.seed(6589)
    n = 100
    volatilities = np.exp(np.random.randn(n) / 2)
    forwards = np.exp(np.random.randn(n))
    strikes = forwards * (1 + (np.random.rand(n) - 0.5) * 0.2)
    expiries = np.exp(np.random.randn(n))
    prices = self.evaluate(
        black_scholes.option_price(forwards, strikes, volatilities, expiries))

    implied_vols = self.evaluate(
        approx_implied_vol.polya(
            prices, forwards, strikes, expiries, validate_args=True))
    self.assertArrayNear(volatilities, implied_vols, 0.6)

  @parameterized.named_parameters(
      # This case should hit the call lower bound since C = F - K.
      ('call_lower', 0.0, 1.0, 1.0, 1.0, True),
      # This case should hit the call upper bound since C = F
      ('call_upper', 1.0, 1.0, 1.0, 1.0, True),
      # This case should hit the put upper bound since C = K
      ('put_lower', 1.0, 1.0, 1.0, 1.0, False),
      # This case should hit the call lower bound since C = F - K.
      ('put_upper', 0.0, 1.0, 1.0, 1.0, False))
  def test_polya_implied_vol_validate_raises(self, price, forward, strike,
                                             expiry, is_call_option):
    """Test the Radiocic-Polya approximation raises appropriately."""
    prices = np.array([price])
    forwards = np.array([forward])
    strikes = np.array([strike])
    expiries = np.array([expiry])
    is_call_options = np.array([is_call_option])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      implied_vols = approx_implied_vol.polya(
          prices,
          forwards,
          strikes,
          expiries,
          is_call_options=is_call_options,
          validate_args=True)
      self.evaluate(implied_vols)


if __name__ == '__main__':
  tf.test.main()
