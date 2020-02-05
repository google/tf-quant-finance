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

import numpy as np
import tensorflow as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


print(dir(tff))

@test_util.run_all_in_graph_and_eager_modes
class AmericanPrice(tf.test.TestCase):
  """Tests for methods for the american pricing module."""

  def test_option_prices(self):
    """Tests that the prices are correct."""
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    strikes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    volatilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cost_of_carries = -0.04
    risk_free_rates = 0.08
    expiries = 0.25
    computed_prices = self.evaluate(
        tff.black_scholes.american_price(
          volatilities,
          strikes,
          expiries,
          risk_free_rates,
          cost_of_carries,
          spots=spots,
          dtype=None))
    expected_prices = np.array(
        [0.03, 0.59, 3.52, 10.31, 20.0])
    self.assertArrayNear(expected_prices, computed_prices, 1e-10)
