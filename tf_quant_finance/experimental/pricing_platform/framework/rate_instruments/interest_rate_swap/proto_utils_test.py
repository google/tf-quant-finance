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
"""Tests for proto_utils."""

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.interest_rate_swap import proto_utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

framework = tff.experimental.pricing_platform.framework
business_days = framework.core.business_days
currencies = framework.core.currencies
daycount_conventions = framework.core.daycount_conventions

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
class ProtoUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._pay_notional = 1000000
    self._receive_notional = 1200000
    self._swap_1 = ir_swap.InterestRateSwap(
        effective_date=date_pb2.Date(year=2019, month=10, day=3),
        maturity_date=date_pb2.Date(year=2029, month=10, day=3),
        currency=Currency.USD(),
        pay_leg=ir_swap.SwapLeg(
            fixed_leg=ir_swap.FixedLeg(
                currency=Currency.USD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=6),
                notional_amount=decimal_pb2.Decimal(
                    units=self._pay_notional),
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
                notional_amount=decimal_pb2.Decimal(
                    units=self._receive_notional),
                floating_rate_type=RateIndex(type="LIBOR_3M"),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)))
    self._swap_2 = ir_swap.InterestRateSwap(
        effective_date=date_pb2.Date(year=2019, month=10, day=8),
        maturity_date=date_pb2.Date(year=2031, month=4, day=27),
        currency=Currency.CAD(),
        receive_leg=ir_swap.SwapLeg(
            fixed_leg=ir_swap.FixedLeg(
                currency=Currency.CAD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=3),
                notional_amount=decimal_pb2.Decimal(
                    units=self._receive_notional),
                fixed_rate=decimal_pb2.Decimal(nanos=31340000),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)),
        pay_leg=ir_swap.SwapLeg(
            floating_leg=ir_swap.FloatingLeg(
                currency=Currency.CAD(),
                coupon_frequency=period_pb2.Period(type="MONTH", amount=3),
                reset_frequency=period_pb2.Period(type="MONTH", amount=3),
                notional_amount=decimal_pb2.Decimal(
                    units=self._pay_notional),
                floating_rate_type=RateIndex(type="LIBOR_3M"),
                daycount_convention=DayCountConventions.ACTUAL_360(),
                business_day_convention=BusinessDayConvention.
                MODIFIED_FOLLOWING(),
                settlement_days=2)))
    super(ProtoUtilsTest, self).setUp()

  def test_group_protos(self):
    proto_dict = proto_utils.group_protos(
        [self._swap_1, self._swap_2, self._swap_1])
    with self.subTest("NumGroups"):
      self.assertLen(proto_dict.keys(), 2)
    with self.subTest("CorrectBatches"):
      proto_list_1, proto_list_2 = proto_dict.values()
      # One of the lists contains two identical protos
      if len(proto_list_1) == 2:
        self.assertProtoEquals(proto_list_1[0], proto_list_1[1])
      else:
        self.assertProtoEquals(proto_list_2[0], proto_list_2[1])

  def test_from_protos(self):
    """Test that from_protos keeps pay leg as a fixed leg."""
    values_dict = proto_utils.from_protos([self._swap_2])
    values = list(values_dict.values())[0]
    # Notional amounts should have negative sign
    pay_leg_notional = values["pay_leg"].notional_amount[0]
    receive_leg_notional = values["receive_leg"].notional_amount[0]
    with self.subTest("PayLegNotional"):
      self.assertEqual(pay_leg_notional, -self._receive_notional)
    with self.subTest("PayLegFixed"):
      self.assertEqual(type(values["pay_leg"]),
                       rate_instruments.coupon_specs.FixedCouponSpecs)
    with self.subTest("ReceiveLegNotional"):
      self.assertEqual(receive_leg_notional, -self._pay_notional)
    with self.subTest("ReceiveLegFixed"):
      self.assertEqual(type(values["receive_leg"]),
                       rate_instruments.coupon_specs.FloatCouponSpecs)


if __name__ == "__main__":
  tf.test.main()
