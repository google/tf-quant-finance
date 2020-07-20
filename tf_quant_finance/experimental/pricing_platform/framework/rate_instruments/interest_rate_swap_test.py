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
"""Tests for insterest rate swap."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


framework = tff.experimental.pricing_platform.framework
business_days = framework.core.business_days
currencies = framework.core.currencies
daycount_conventions = framework.core.daycount_conventions
interpolation_method = framework.core.interpolation_method
rate_indices = framework.core.rate_indices

instrument_protos = tff.experimental.pricing_platform.instrument_protos
date_pb2 = instrument_protos.date
decimal_pb2 = instrument_protos.decimal
ir_swap = instrument_protos.interest_rate_swap
period_pb2 = instrument_protos.period

rate_instruments = tff.experimental.pricing_platform.framework.rate_instruments
interest_rate_swap = rate_instruments.interest_rate_swap

market_data = tff.experimental.pricing_platform.framework.market_data
market_data_config = market_data.config


DayCountConventions = daycount_conventions.DayCountConventions
BusinessDayConvention = business_days.BusinessDayConvention
RateIndexType = rate_indices.RateIndexType
Currency = currencies.Currency


@test_util.run_all_in_graph_and_eager_modes
class InterestRateSwapTest(tf.test.TestCase):

  def setUp(self):
    self._swap_1 = ir_swap.InterestRateSwap(
        effective_date=date_pb2.Date(year=2019, month=10, day=3),
        maturity_date=date_pb2.Date(year=2029, month=10, day=3),
        currency=Currency.USD(),
        pay_leg=ir_swap.SwapLeg(
            fixed_leg=ir_swap.FixedLeg(
                currency=Currency.USD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=6),
                notional_amount=decimal_pb2.Decimal(units=1000000),
                fixed_rate=decimal_pb2.Decimal(nanos=31340000),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)),
        receive_leg=ir_swap.SwapLeg(
            floating_leg=ir_swap.FloatingLeg(
                currency=Currency.USD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=3),
                reset_frequency=period_pb2.Period(type="MONTH", amount=3),
                notional_amount=decimal_pb2.Decimal(units=1000000),
                floating_rate_type=RateIndexType.USD_LIBOR(),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)))
    self._swap_2 = ir_swap.InterestRateSwap(
        effective_date=date_pb2.Date(year=2019, month=10, day=3),
        maturity_date=date_pb2.Date(year=2029, month=10, day=3),
        currency=Currency.USD(),
        pay_leg=ir_swap.SwapLeg(
            fixed_leg=ir_swap.FixedLeg(
                currency=Currency.USD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=3),
                notional_amount=decimal_pb2.Decimal(units=1000000),
                fixed_rate=decimal_pb2.Decimal(nanos=31340000),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)),
        receive_leg=ir_swap.SwapLeg(
            floating_leg=ir_swap.FloatingLeg(
                currency=Currency.USD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=3),
                reset_frequency=period_pb2.Period(type="MONTH", amount=3),
                notional_amount=decimal_pb2.Decimal(units=1000000),
                floating_rate_type=RateIndexType.USD_LIBOR(),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)))
    date = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
            [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    discount = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                0.72494879, 0.37602059]
    libor_3m_config = market_data_config.RateConfig(
        interpolation_method=interpolation_method.InterpolationMethod.LINEAR)
    self._rate_config = {"USD": {"LIBOR_3M_USD": libor_3m_config}}
    self._market_data_dict = {"USD": {
        "OIS_USD":
        {"date": date, "discount": discount},
        "LIBOR_3M_USD":
        {"date": date, "discount": discount},}}
    self._valuation_date = [(2020, 6, 24)]
    super(InterestRateSwapTest, self).setUp()

  def test_from_proto_price(self):
    """Creates ir swap from proto and tests pricing method."""
    market = market_data.MarketDataDict(
        self._valuation_date,
        self._market_data_dict,
        config=self._rate_config)
    swaps = interest_rate_swap.InterestRateSwap.from_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    with self.subTest("Batching"):
      self.assertLen(swaps, 2)
    price1 = swaps[0].price(market)
    expected1 = np.array([7655.98694587, 7655.98694587])
    with self.subTest("PriceBatch"):
      self.assertAllClose(price1, expected1)
    price2 = swaps[1].price(market)
    expected2 = np.array([6569.04475892])
    with self.subTest("PriceSingle"):
      self.assertAllClose(price2, expected2)

  def test_ir_delta_parallel(self):
    """Creates ir swap from proto and tests IR delta parallel method."""
    market = market_data.MarketDataDict(
        self._valuation_date,
        self._market_data_dict,
        config=self._rate_config)
    swaps = interest_rate_swap.InterestRateSwap.from_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    # Automatic differentiation test
    delta1 = swaps[0].ir_delta_parallel(market)
    expected1 = np.array([7689639.46004707, 7689639.46004707])
    with self.subTest("DeltaBatchAutograd"):
      self.assertAllClose(delta1, expected1)
    delta2 = swaps[1].ir_delta_parallel(market)
    expected2 = np.array([7662889.94933313])
    with self.subTest("DeltaSingleAutograd"):
      self.assertAllClose(delta2, expected2)
    # Shock size test
    delta1 = swaps[0].ir_delta_parallel(market, shock_size=1e-4)
    expected1 = np.array([7685967.85230533, 7685967.85230533])
    with self.subTest("DeltaBatch"):
      self.assertAllClose(delta1, expected1)
    delta2 = swaps[1].ir_delta_parallel(market, shock_size=1e-4)
    expected2 = np.array([7659231.64891894])
    with self.subTest("DeltaSingle"):
      self.assertAllClose(delta2, expected2)

if __name__ == "__main__":
  tf.test.main()
