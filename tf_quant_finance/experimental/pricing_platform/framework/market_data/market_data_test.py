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
"""Tests for the market data."""
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

market_data = tff.experimental.pricing_platform.framework.market_data
interpolation_method = tff.experimental.pricing_platform.framework.core.interpolation_method


@test_util.run_all_in_graph_and_eager_modes
class MarketDataTest(tf.test.TestCase):

  def setUp(self):
    date = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
            [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    discount = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                0.72494879, 0.37602059]
    libor_3m_config = market_data.config.RateConfig(
        interpolation_method=interpolation_method.InterpolationMethod.LINEAR)
    self._rate_config = {"USD": {"LIBOR_3M_USD": libor_3m_config}}
    self._market_data_dict = {"USD": {
        "OIS_USD":
        {"date": date, "discount": discount},
        "LIBOR_3M_USD":
        {"date": date, "discount": discount},}}
    self._valuation_date = [(2020, 6, 24)]
    self._discount = discount
    super(MarketDataTest, self).setUp()

  def test_discount_curve(self):
    market = market_data.MarketDataDict(
        self._valuation_date,
        self._market_data_dict,
        config=self._rate_config)
    # Get the discount curve
    curve_type = market_data.config.curve_type_from_id("LIBOR_3M_USD")
    yield_curve = market.yield_curve(curve_type)
    discount_factor_nodes = yield_curve.discount_factor_nodes
    self.assertAllClose(discount_factor_nodes, self._discount)


if __name__ == "__main__":
  tf.test.main()
