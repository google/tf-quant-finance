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

from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.forward_rate_agreement import proto_utils
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
class ProtoUtilsTest(tf.test.TestCase):

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
        short_position=False,
        fixing_date=date_pb2.Date(year=2021, month=5, day=21),
        currency=Currency.USD(),
        fixed_rate=decimal_pb2.Decimal(nanos=31340000),
        notional_amount=decimal_pb2.Decimal(units=10000),
        daycount_convention=DayCountConventions.ACTUAL_365(),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
        floating_rate_term=fra_pb2.FloatingRateTerm(
            floating_rate_type=RateIndex(type="LIBOR_6M"),
            term=period_pb2.Period(type="MONTH", amount=6)),
        settlement_days=2)
    super(ProtoUtilsTest, self).setUp()

  def test_group_protos(self):
    proto_dict = proto_utils.group_protos(
        [self._fra_1, self._fra_2, self._fra_1, self._fra_3])
    with self.subTest("NumGroups"):
      self.assertLen(proto_dict.keys(), 3)
    with self.subTest("CorrectBatches"):
      # One of the lists contains two identical protos
      for proto_list in proto_dict.values():
        if len(proto_list) == 2:
          self.assertProtoEquals(proto_list[0], proto_list[1])

  def test_group_protos_v2(self):
    proto_dict = proto_utils.group_protos_v2(
        [self._fra_1, self._fra_2, self._fra_1, self._fra_3])
    with self.subTest("NumGroups"):
      self.assertLen(proto_dict.keys(), 2)


if __name__ == "__main__":
  tf.test.main()
