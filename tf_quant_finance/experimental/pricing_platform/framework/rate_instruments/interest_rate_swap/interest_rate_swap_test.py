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

from absl.testing import parameterized
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
ir_swap = instrument_protos.interest_rate_swap
period_pb2 = instrument_protos.period

rate_instruments = tff.experimental.pricing_platform.framework.rate_instruments
interest_rate_swap = rate_instruments.interest_rate_swap

market_data = tff.experimental.pricing_platform.framework.market_data
market_data_config = market_data.config

DayCountConventions = daycount_conventions.DayCountConventions
BusinessDayConvention = business_days.BusinessDayConvention
RateIndex = instrument_protos.rate_indices.RateIndex
Currency = currencies.Currency


@test_util.run_all_in_graph_and_eager_modes
class InterestRateSwapTest(tf.test.TestCase, parameterized.TestCase):

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
                floating_rate_type=RateIndex(type="LIBOR_3M"),
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
                floating_rate_type=RateIndex(type="LIBOR_3M"),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)))
    dates = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
             [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    discounts = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                 0.72494879, 0.37602059]
    libor_3m_config = {
        "interpolation_method": interpolation_method.InterpolationMethod.LINEAR
    }
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
                    "config": libor_3m_config,
                }
            }
        },
        "reference_date": [(2020, 6, 24)],
    }
    super(InterestRateSwapTest, self).setUp()

  def test_from_proto_price(self):
    """Creates ir swap from proto and tests pricing method."""
    market = market_data.MarketDataDict(
        self._market_data_dict)
    swaps = interest_rate_swap.InterestRateSwap.from_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    with self.subTest("Batching"):
      self.assertLen(swaps, 1)
    price1 = swaps[0].price(market)
    expected1 = np.array([7655.98694587, 6569.04475892, 7655.98694587])
    with self.subTest("PriceBatch"):
      self.assertAllClose(price1, expected1)

  def test_price_with_fixings(self):
    """Creates swap from proto and tests pricing method with supplied fixings.
    """
    fixing_dates = [(2019, 10, 3), (2020, 1, 3), (2020, 4, 7), (2020, 5, 3)]
    fixing_rates = [0.01, 0.02, 0.03, 0.025]
    market_data_dict = self._market_data_dict
    market_data_dict["rates"]["USD"]["LIBOR_3M"]["fixing_dates"] = fixing_dates
    market_data_dict["rates"]["USD"]["LIBOR_3M"]["fixing_rates"] = fixing_rates
    market_data_dict["rates"]["USD"]["LIBOR_3M"][
        "fixing_daycount"] = "ACTUAL_365"
    market = market_data.MarketDataDict(
        market_data_dict)
    swaps = interest_rate_swap.InterestRateSwap.from_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    with self.subTest("Batching"):
      self.assertLen(swaps, 1)
    price1 = swaps[0].price(market)
    expected1 = np.array([9543.57776645, 8456.63557949, 9543.57776645])
    with self.subTest("PriceBatch"):
      self.assertAllClose(price1, expected1)

  def test_ir_delta_parallel(self):
    """Creates ir swap from proto and tests IR delta parallel method."""
    market = market_data.MarketDataDict(
        self._market_data_dict)
    swaps = interest_rate_swap.InterestRateSwap.from_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    # Automatic differentiation test
    delta1 = swaps[0].ir_delta_parallel(market)
    expected1 = np.array([7689639.46004707, 7662889.94933313,
                          7689639.46004707])
    # Autograd test
    with self.subTest("DeltaBatchAutograd"):
      self.assertAllClose(delta1, expected1)

  def test_ir_delta_parallel_shock_size(self):
    """Creates ir swap from proto and tests IR delta parallel method."""
    market = market_data.MarketDataDict(
        self._market_data_dict)
    swaps = interest_rate_swap.InterestRateSwap.from_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    # Shock size test
    delta1 = swaps[0].ir_delta_parallel(market, shock_size=1e-4)
    expected1 = np.array([7685967.85230533, 7659231.64891894,
                          7685967.85230533])
    with self.subTest("DeltaBatch"):
      self.assertAllClose(delta1, expected1)

  def test_create_constructor_args_price(self):
    """Creates and prices swap from a dictionary representation."""

    swaps_dict = interest_rate_swap.InterestRateSwap.create_constructor_args(
        [self._swap_1, self._swap_2, self._swap_1])
    market = market_data.MarketDataDict(
        self._market_data_dict)
    swaps = interest_rate_swap.InterestRateSwap(**list(swaps_dict.values())[0])
    price1 = swaps.price(market)
    expected1 = np.array([7655.98694587, 6569.04475892, 7655.98694587])
    self.assertAllClose(price1, expected1)

  @parameterized.named_parameters({
      "testcase_name": "no_fixigns",
      "past_fixing": 0.0,
      "expecter_res": [-0.7533],
  }, {
      "testcase_name": "with_fixigns",
      "past_fixing": 0.1,
      "expecter_res": [251.8048],
  })
  def test_swap_constructor(self, past_fixing, expecter_res):
    fixed_coupon = rate_instruments.coupon_specs.FixedCouponSpecs(
        currency=currencies.Currency.USD,
        coupon_frequency=tff.datetime.periods.months(3),
        notional_amount=10000,
        fixed_rate=0.03,
        daycount_convention=daycount_conventions.DayCountConventions.ACTUAL_360,
        businessday_rule=business_days.BusinessDayConvention.FOLLOWING,
        settlement_days=2,
        calendar=business_days.BankHolidays.US)
    float_coupon = rate_instruments.coupon_specs.FloatCouponSpecs(
        currency=currencies.Currency.USD,
        coupon_frequency=tff.datetime.periods.months(3),
        reset_frequency=tff.datetime.periods.months(3),
        notional_amount=10000,
        floating_rate_type=framework.core.rate_indices.RateIndex(
            type="LIBOR_3M"),
        daycount_convention=daycount_conventions.DayCountConventions.ACTUAL_360,
        businessday_rule=business_days.BusinessDayConvention.FOLLOWING,
        settlement_days=2,
        spread=0.0,
        calendar=business_days.BankHolidays.US)
    schedule = tff.datetime.PeriodicSchedule(
        start_date=tff.datetime.dates_from_tensor([2019, 1, 1]),
        end_date=tff.datetime.dates_from_tensor([2021, 1, 1]),
        tenor=tff.datetime.periods.months(3)).dates()
    swap = interest_rate_swap.InterestRateSwap(
        pay_leg=fixed_coupon,
        receive_leg=float_coupon,
        pay_leg_schedule=schedule,
        receive_leg_schedule=schedule,
        config=interest_rate_swap.InterestRateSwapConfig(
            past_fixing=past_fixing))
    market = market_data.MarketDataDict(self._market_data_dict)
    price = swap.price(market)
    self.assertAllClose(price, expecter_res, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  tf.test.main()
