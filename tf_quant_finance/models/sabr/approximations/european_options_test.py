# Lint as: python3
# Copyright 2021 Google LLC
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
"""Tests for Sabr Approximations of European Option prices."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

# Helper aliases.
NORMAL = tff.models.sabr.approximations.SabrImpliedVolatilityType.NORMAL
LOGNORMAL = tff.models.sabr.approximations.SabrImpliedVolatilityType.LOGNORMAL


class SabrApproximationEuropeanOptionsTest(parameterized.TestCase,
                                           tf.test.TestCase):
  """Tests for SABR approximations to implied volatility."""

  def test_european_option_docstring_example(self):
    prices = tff.models.sabr.approximations.european_option_price(
        forwards=np.array([100.0, 110.0]),
        strikes=np.array([90.0, 100.0]),
        expiries=np.array([0.5, 1.0]),
        is_call_options=np.array([True, False]),
        alpha=3.2,
        beta=0.2,
        nu=1.4,
        rho=0.0005,
        dtype=tf.float64)

    prices = self.evaluate(prices)
    self.assertAllClose(prices, [10.41244961, 1.47123225])

  def test_european_option_normal(self):
    prices = tff.models.sabr.approximations.european_option_price(
        forwards=np.array([100.0, 110.0]),
        strikes=np.array([90.0, 100.0]),
        expiries=np.array([0.5, 1.0]),
        is_call_options=np.array([True, False]),
        alpha=3.2,
        beta=0.2,
        nu=1.4,
        rho=0.0005,
        volatility_type=NORMAL,
        dtype=tf.float64)

    prices = self.evaluate(prices)
    self.assertAllClose(prices, [10.412692, 1.472544])


if __name__ == '__main__':
  tf.test.main()
