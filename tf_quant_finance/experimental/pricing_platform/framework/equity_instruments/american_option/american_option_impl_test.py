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
"""Tests for equity american option."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


framework = tff.experimental.pricing_platform.framework
business_days = framework.core.business_days
currencies = framework.core.currencies
daycount_conventions = framework.core.daycount_conventions
interpolation_method = framework.core.interpolation_method

instrument_protos = tff.experimental.pricing_platform.instrument_protos
date_pb2 = instrument_protos.date
decimal_pb2 = instrument_protos.decimal
period_pb2 = instrument_protos.period
american_option_pb2 = instrument_protos.american_equity_option

equity_instruments = tff.experimental.pricing_platform.framework.equity_instruments
american_option = equity_instruments.american_option

market_data = tff.experimental.pricing_platform.framework.market_data
market_data_config = market_data.config

DayCountConventions = daycount_conventions.DayCountConventions
BusinessDayConvention = business_days.BusinessDayConvention
RateIndex = instrument_protos.rate_indices.RateIndex
Currency = currencies.Currency


@test_util.run_all_in_graph_and_eager_modes
class AmericanEquityOptionTest(tf.test.TestCase):

  def setUp(self):
    self._american_option_1 = american_option_pb2.AmericanEquityOption(
        short_position=True,
        expiry_date=date_pb2.Date(year=2022, month=5, day=21),
        contract_amount=decimal_pb2.Decimal(units=10000),
        strike=decimal_pb2.Decimal(units=1500),
        equity="GOOG",
        currency=Currency.USD(),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
        settlement_days=2,
        is_call_option=False)

    self._american_option_2 = american_option_pb2.AmericanEquityOption(
        short_position=True,
        expiry_date=date_pb2.Date(year=2022, month=3, day=21),
        contract_amount=decimal_pb2.Decimal(units=10000),
        strike=decimal_pb2.Decimal(units=590),
        equity="EZJ",
        currency=Currency.GBP(),
        business_day_convention=BusinessDayConvention.FOLLOWING(),
        settlement_days=2,
        is_call_option=False)

    self._american_option_3 = american_option_pb2.AmericanEquityOption(
        short_position=True,
        expiry_date=date_pb2.Date(year=2022, month=2, day=21),
        contract_amount=decimal_pb2.Decimal(units=10000),
        strike=decimal_pb2.Decimal(units=590),
        equity="EZJ",
        currency=Currency.GBP(),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
        settlement_days=2,
        is_call_option=True)
    risk_free_dates = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
                       [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    risk_free_discounts = [0.97197441, 0.94022746, 0.91074031, 0.85495089,
                           0.8013675, 0.72494879, 0.37602059]

    vol_dates = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8],
                 [2025, 2, 8], [2027, 2, 8]]
    strikes_goog = [[1450, 1500, 1550],
                    [1450, 1500, 1550],
                    [1450, 1500, 1550],
                    [1450, 1500, 1550],
                    [1450, 1500, 1550]]
    strikes_ezj = [[570, 590, 610],
                   [570, 590, 610],
                   [570, 590, 610],
                   [570, 590, 610],
                   [570, 590, 610]]
    volatilities = [[0.1, 0.12, 0.13],
                    [0.15, 0.2, 0.15],
                    [0.1, 0.2, 0.1],
                    [0.1, 0.2, 0.1],
                    [0.1, 0.1, 0.3]]
    self._market_data_dict = {
        "rates": {
            "USD": {
                "risk_free_curve": {
                    "dates": risk_free_dates,
                    "discounts": risk_free_discounts
                }
            },
            "GBP": {
                "risk_free_curve": {
                    "dates": risk_free_dates,
                    "discounts": risk_free_discounts
                }
            },
        },
        "equities": {
            "USD": {
                "GOOG": {
                    "spot_price": 1500,
                    "volatility_surface": {
                        "dates": vol_dates,
                        "strikes": strikes_goog,
                        "implied_volatilities": volatilities
                    }
                }
            },
            "GBP": {
                "EZJ": {
                    "spot_price": 590,
                    "volatility_surface": {
                        "dates": vol_dates,
                        "strikes": strikes_ezj,
                        "implied_volatilities": volatilities
                    }
                }
            }
        },
        "reference_date": [(2021, 2, 5)],
    }
    super(AmericanEquityOptionTest, self).setUp()

  def test_from_proto_price(self):
    """Creates ir swap from proto and tests pricing method."""
    market = market_data.MarketDataDict(self._market_data_dict)
    config = american_option.AmericanOptionConfig(
        num_samples=1000, num_exercise_times=10, seed=[1, 2])
    am_option = american_option.AmericanOption.from_protos(
        [self._american_option_1,
         self._american_option_2,
         self._american_option_3], config)

    with self.subTest("Batching"):
      self.assertLen(am_option, 2)
    price1 = am_option[0].price(market)
    expected1 = np.array([4855122.1403, 512094.7353])
    with self.subTest("PriceBatch"):
      self.assertAllClose(price1, expected1, rtol=1e-2, atol=0)
    price2 = am_option[1].price(market)
    expected2 = np.array([897927.1153])
    with self.subTest("PriceSingle"):
      self.assertAllClose(price2, expected2, rtol=1e-2, atol=0)

  def test_default_config(self):
    """Creates ir swap from proto and tests pricing method."""
    am_option = american_option.AmericanOption.from_protos(
        [self._american_option_1,
         self._american_option_2,
         self._american_option_3])

    default_config = american_option.AmericanOptionConfig()
    with self.subTest("Batching"):
      self.assertLen(am_option, 2)
    with self.subTest("Seed"):
      self.assertAllEqual(am_option[0]._seed, default_config.seed)
    with self.subTest("Model"):
      self.assertAllEqual(am_option[0]._model, default_config.model)
    with self.subTest("NumSamples"):
      self.assertAllEqual(am_option[0]._num_samples,
                          default_config.num_samples)
    with self.subTest("NumExerciseTimes"):
      self.assertAllEqual(am_option[0]._num_exercise_times,
                          default_config.num_exercise_times)
    with self.subTest("NumCalibrationSamples"):
      self.assertAllEqual(am_option[0]._num_calibration_samples,
                          default_config.num_calibration_samples)

  def test_from_proto_price_num_calibration(self):
    """Creates ir swap from proto and tests pricing method."""
    market = market_data.MarketDataDict(self._market_data_dict)
    config = american_option.AmericanOptionConfig(
        num_samples=1500, num_calibration_samples=500,
        num_exercise_times=10, seed=[1, 2])
    am_option = american_option.AmericanOption.from_protos(
        [self._american_option_1,
         self._american_option_2,
         self._american_option_3], config)

    with self.subTest("Batching"):
      self.assertLen(am_option, 2)
    price1 = am_option[0].price(market)
    expected1 = np.array([4850253.3942, 495803.5293])
    with self.subTest("PriceBatch"):
      self.assertAllClose(price1, expected1, rtol=1e-2, atol=0)
    price2 = am_option[1].price(market)
    expected2 = np.array([895160.0769])
    with self.subTest("PriceSingle"):
      self.assertAllClose(price2, expected2, rtol=1e-2, atol=0)

  def test_create_constructor_args_price(self):
    """Creates and prices swap from a dictionary representation."""
    config = american_option.AmericanOptionConfig(
        num_samples=100, num_calibration_samples=50,
        num_exercise_times=10, seed=[1, 2])
    am_option_dict = american_option.AmericanOption.create_constructor_args(
        [self._american_option_1, self._american_option_3], config)
    market = market_data.MarketDataDict(self._market_data_dict)
    am_options = american_option.AmericanOption(
        **list(am_option_dict.values())[0])
    price = am_options.price(market)
    expected = np.array([4943331.4747, 413834.5428])
    self.assertAllClose(price, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
