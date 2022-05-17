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
"""Tests for cashflow streams."""
from absl.testing import parameterized
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

cashflow_streams = tff.experimental.pricing_platform.framework.rate_instruments.cashflow_streams
coupon_specs = tff.experimental.pricing_platform.framework.rate_instruments.coupon_specs
core = tff.experimental.pricing_platform.framework.core
daycount_conventions = core.daycount_conventions
business_days = core.business_days
market_data = tff.experimental.pricing_platform.framework.market_data


@test_util.run_all_in_graph_and_eager_modes
class CashflowStreamsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    dates = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
             [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
    discounts = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
                 0.72494879, 0.37602059]
    self._market_data_dict = {
        "rates": {
            "USD": {
                "risk_free_curve":
                    {
                        "dates": dates, "discounts": discounts},
                "LIBOR_3M":
                    {"dates": dates, "discounts": discounts},
            }
        },
        "reference_date": [(2020, 6, 24)],
    }
    super(CashflowStreamsTest, self).setUp()

  @parameterized.named_parameters(
      {
          "testcase_name": "ZeroFixings",
          "fixing_dates": [(2019, 10, 3), (2020, 1, 3), (2020, 2, 3),
                           (2020, 5, 3)],
          "fixing_rates": [0.0, 0.0, 0.0, 0.0],
          "expected_fixings": [0.0, 0.0, 0.04494299],
      }, {
          "testcase_name": "SupplyFixings",
          "fixing_dates": [(2019, 10, 3), (2020, 1, 3), (2020, 5, 4),
                           (2020, 6, 3)],
          "fixing_rates": [0.01, 0.02, 0.03, 0.025],
          "expected_fixings": [0.00756164, 0.00609589, 0.04494299],
      })

  def test_fixings(self, fixing_dates, fixing_rates, expected_fixings):
    market_data_dict = self._market_data_dict
    market_data_dict["rates"]["USD"]["LIBOR_3M"]["fixing_dates"] = fixing_dates
    market_data_dict["rates"]["USD"]["LIBOR_3M"]["fixing_rates"] = fixing_rates
    market_data_dict["rates"]["USD"]["LIBOR_3M"][
        "fixing_daycount"] = "ACTUAL_365"
    market = market_data.MarketDataDict(
        market_data_dict)
    coupon_spec = coupon_specs.FloatCouponSpecs(
        currency=core.currencies.Currency.USD,
        reset_frequency=tff.datetime.periods.months(3),
        coupon_frequency=tff.datetime.periods.months(3),
        notional_amount=100,
        floating_rate_type=core.rate_indices.RateIndex(type="LIBOR_3M"),
        daycount_convention=daycount_conventions.DayCountConventions.ACTUAL_360,
        businessday_rule=business_days.BusinessDayConvention.FOLLOWING,
        settlement_days=0,
        spread=0.0,
        calendar=business_days.BankHolidays.US)
    cashflow_stream = cashflow_streams.FloatingCashflowStream(
        coupon_spec=coupon_spec,
        discount_curve_type=core.curve_types.RateIndexCurve(
            currency=core.currencies.Currency.USD,
            index=core.rate_indices.RateIndex(type="LIBOR_3M")),
        start_date=[[2020, 5, 2], [2020, 6, 4], [2020, 8, 1]],
        end_date=[[2020, 10, 2], [2020, 9, 1], [2021, 3, 1]],
        dtype=tf.float64)

    _, forward_rates = cashflow_stream.forward_rates(market)
    with self.subTest("ForwardRateShape"):
      self.assertAllEqual(forward_rates.shape, [3, 5])
    forward_rates = self.evaluate(forward_rates)
    with self.subTest("PastFixings"):
      self.assertAllClose(forward_rates[:, 1], expected_fixings)
    with self.subTest("NextForwards"):
      self.assertAllClose(forward_rates[:, 2], [0.07008059, 0.0, 0.04494299])


if __name__ == "__main__":
  tf.test.main()
