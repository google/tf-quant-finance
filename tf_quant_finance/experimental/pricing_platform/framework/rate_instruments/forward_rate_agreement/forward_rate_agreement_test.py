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
"""Tests for forward rate agreement."""

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
fra_pb2 = instrument_protos.forward_rate_agreement

rate_instruments = tff.experimental.pricing_platform.framework.rate_instruments
forward_rate_agreement = rate_instruments.forward_rate_agreement

market_data = tff.experimental.pricing_platform.framework.market_data

DayCountConventions = daycount_conventions.DayCountConventions
BusinessDayConvention = business_days.BusinessDayConvention
RateIndex = instrument_protos.rate_indices.RateIndex
Currency = currencies.Currency


@test_util.run_all_in_graph_and_eager_modes
class ForwardRateAgreementTest(tf.test.TestCase):

  def setUp(self):
    self._fra_1 = fra_pb2.ForwardRateAgreement(
        short_position=True,
        fixing_date=date_pb2.Date(year=2021, month=5, day=21),
        currency=Currency.USD(),
        fixed_rate=decimal_pb2.Decimal(nanos=31340000),
        notional_amount=decimal_pb2.Decimal(units=10000),
        daycount_convention=DayCountConventions.ACTUAL_360(),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
        floating_rate_term=fra_pb2.FloatingRateTerm(
            floating_rate_type=RateIndex(type="LIBOR_3M"),
            term=period_pb2.Period(type="MONTH", amount=3)),
        settlement_days=2)

    self._fra_2 = fra_pb2.ForwardRateAgreement(
        short_position=False,
        fixing_date=date_pb2.Date(year=2021, month=5, day=21),
        currency=Currency.USD(),
        fixed_rate=decimal_pb2.Decimal(nanos=31340000),
        notional_amount=decimal_pb2.Decimal(units=10000),
        daycount_convention=DayCountConventions.ACTUAL_365(),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
        floating_rate_term=fra_pb2.FloatingRateTerm(
            floating_rate_type=RateIndex(type="LIBOR_3M"),
            term=period_pb2.Period(type="MONTH", amount=3)),
        settlement_days=2)

    self._fra_3 = fra_pb2.ForwardRateAgreement(
        short_position=True,
        fixing_date=date_pb2.Date(year=2021, month=5, day=21),
        currency=Currency.USD(),
        fixed_rate=decimal_pb2.Decimal(nanos=31340000),
        notional_amount=decimal_pb2.Decimal(units=10000),
        daycount_convention=DayCountConventions.ACTUAL_360(),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
        floating_rate_term=fra_pb2.FloatingRateTerm(
            floating_rate_type=RateIndex(type="LIBOR_3M"),
            term=period_pb2.Period(type="MONTH", amount=3)),
        settlement_days=2)

    dates = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
             [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    discounts = [0.97197441, 0.94022746, 0.91074031, 0.85495089,
                 0.8013675, 0.72494879, 0.37602059]
    self._market_data_dict = {
        "rates": {
            "USD": {
                "risk_free_curve": {
                    "dates": dates,
                    "discounts": discounts,
                },
                "LIBOR_3M": {
                    "dates": dates,
                    "discounts": discounts,
                }
            }
        },
        "reference_date": [(2020, 2, 8)],
    }

    super(ForwardRateAgreementTest, self).setUp()

  def test_from_proto_price(self):
    market = market_data.MarketDataDict(self._market_data_dict)
    fra_portfolio = forward_rate_agreement.ForwardRateAgreement.from_protos(
        [self._fra_1, self._fra_2, self._fra_3])
    with self.subTest("Batching"):
      self.assertLen(fra_portfolio, 2)
    price1 = fra_portfolio[0].price(market)
    expected1 = np.array([4.05463257, 4.05463257])
    with self.subTest("PriceBatch"):
      self.assertAllClose(price1, expected1)
    price2 = fra_portfolio[1].price(market)
    expected2 = np.array([-5.10228969])
    with self.subTest("PriceSingle"):
      self.assertAllClose(price2, expected2)

  def test_create_constructor_args_price(self):
    """Creates and prices FRAs from a dictionary representation."""
    fra_dict = (
        forward_rate_agreement.ForwardRateAgreement.create_constructor_args(
            [self._fra_1, self._fra_1]))
    market = market_data.MarketDataDict(
        self._market_data_dict)
    fras = forward_rate_agreement.ForwardRateAgreement(
        **list(fra_dict.values())[0])
    price = fras.price(market)
    expected = np.array([4.05463257, 4.05463257])
    self.assertAllClose(price, expected)

if __name__ == "__main__":
  tf.test.main()
