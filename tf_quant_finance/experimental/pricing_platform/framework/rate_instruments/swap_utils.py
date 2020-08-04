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
"""Utilities for creating a swap instrument."""

from typing import Union

from tf_quant_finance.experimental.pricing_platform.framework.core import business_days
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import coupon_specs
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import utils as instrument_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import interest_rate_swap_pb2 as ir_swap


def leg_from_proto(
    leg_proto: ir_swap.SwapLeg) -> Union[coupon_specs.FixedCouponSpecs,
                                         coupon_specs.FloatCouponSpecs]:
  """Initialized coupon specifications from a proto instance."""
  if leg_proto.HasField("fixed_leg"):
    leg = leg_proto.fixed_leg
    return coupon_specs.FixedCouponSpecs(
        currency=currencies.from_proto_value(leg.currency),
        coupon_frequency=leg.coupon_frequency,
        notional_amount=[instrument_utils.decimal_to_double(
            leg.notional_amount)],
        fixed_rate=[instrument_utils.decimal_to_double(
            leg.fixed_rate)],
        settlement_days=[leg.settlement_days],
        businessday_rule=business_days.convention_from_proto_value(
            leg.business_day_convention),
        daycount_convention=daycount_conventions.from_proto_value(
            leg.daycount_convention),
        calendar=business_days.holiday_from_proto_value(leg.bank_holidays))
  else:
    leg = leg_proto.floating_leg
    # Get the index rate object
    rate_index = leg.floating_rate_type
    rate_index = rate_indices.RateIndex.from_proto(rate_index)
    rate_index.name = [rate_index.name]
    rate_index.source = [rate_index.source]
    return coupon_specs.FloatCouponSpecs(
        currency=currencies.from_proto_value(leg.currency),
        coupon_frequency=leg.coupon_frequency,
        reset_frequency=leg.reset_frequency,
        notional_amount=[instrument_utils.decimal_to_double(
            leg.notional_amount)],
        floating_rate_type=rate_index,
        settlement_days=[leg.settlement_days],
        businessday_rule=business_days.convention_from_proto_value(
            leg.business_day_convention),
        daycount_convention=daycount_conventions.from_proto_value(
            leg.daycount_convention),
        spread=[instrument_utils.decimal_to_double(leg.spread)],
        calendar=business_days.holiday_from_proto_value(leg.bank_holidays))


def update_leg(
    current_leg: Union[coupon_specs.FixedCouponSpecs,
                       coupon_specs.FloatCouponSpecs],
    leg: Union[coupon_specs.FixedCouponSpecs, coupon_specs.FloatCouponSpecs]):
  """Adds new leg info to the current leg."""
  if isinstance(current_leg, coupon_specs.FixedCouponSpecs):
    if not isinstance(leg, coupon_specs.FixedCouponSpecs):
      raise ValueError("Both `current_leg` and `leg` should beof the same "
                       "fixed or float type.")
    current_leg.notional_amount += leg.notional_amount
    current_leg.fixed_rate += leg.fixed_rate
    current_leg.settlement_days += leg.settlement_days
  else:
    if not isinstance(leg, coupon_specs.FloatCouponSpecs):
      raise ValueError("Both `current_leg` and `leg` should beof the same "
                       "fixed or float type.")
    current_leg.notional_amount += leg.notional_amount
    update_rate_index(current_leg.floating_rate_type, leg.floating_rate_type)
    current_leg.settlement_days += leg.settlement_days
    current_leg.spread += leg.spread


def update_rate_index(
    current_index: rate_indices.RateIndex,
    index: rate_indices.RateIndex):
  """Concatenates index to current_index if possible."""
  if current_index.type != index.type:
    raise ValueError(f"Can not join {current_index.type} and {index.type}")
  current_index.name = current_index.name + index.name
  current_index.source = current_index.source + index.source
