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
"""Utilities for proto processing."""

from typing import Any, List, Dict, Tuple

from tf_quant_finance.experimental.pricing_platform.framework.core import business_days
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import utils as instrument_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.interest_rate_swap import proto_utils as swap_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import forward_rate_agreement_pb2 as fra
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2


def _get_hash(
    fra_proto: fra.ForwardRateAgreement
    ) -> Tuple[int, types.CurrencyProtoType,
               period_pb2.Period, rate_indices.RateIndex]:
  """Computes hash key for the batching strategy."""
  currency = currencies.from_proto_value(fra_proto.currency)
  bank_holidays = fra_proto.bank_holidays
  daycount_convention = fra_proto.daycount_convention
  business_day_convention = fra_proto.business_day_convention
  # Get rate index
  rate_index = fra_proto.floating_rate_term.floating_rate_type
  rate_index = rate_indices.RateIndex.from_proto(rate_index)
  rate_index.name = [rate_index.name]
  rate_index.source = [rate_index.source]

  rate_term = fra_proto.floating_rate_term.term

  h = hash(tuple([currency] + [bank_holidays] + [rate_term.type]
                 + [rate_term.amount] + [rate_index.type]
                 + [daycount_convention] + [business_day_convention]))
  return h, currency, rate_term, rate_index


def group_protos(
    proto_list: List[fra.ForwardRateAgreement],
    fra_config: "ForwardRateAgreementConfig" = None
    ) -> Dict[str, List["ForwardRateAgreement"]]:
  """Creates a dictionary of grouped protos."""
  del fra_config  # fra_config does not impact the batching
  grouped_fras = {}
  for fra_proto in proto_list:
    h, _, _, _ = _get_hash(fra_proto)
    if h in grouped_fras:
      grouped_fras[h].append(fra_proto)
    else:
      grouped_fras[h] = [fra_proto]
  return grouped_fras


def from_protos(
    proto_list: List[fra.ForwardRateAgreement],
    fra_config: "ForwardRateAgreementConfig" = None
    ) -> Dict[str, Any]:
  """Creates a dictionary of preprocessed swap data."""
  prepare_fras = {}
  for fra_proto in proto_list:
    short_position = fra_proto.short_position
    h, currency, rate_term, rate_index = _get_hash(fra_proto)
    fixing_date = fra_proto.fixing_date
    fixing_date = [fixing_date.year,
                   fixing_date.month,
                   fixing_date.day]
    notional_amount = instrument_utils.decimal_to_double(
        fra_proto.notional_amount)
    daycount_convention = daycount_conventions.from_proto_value(
        fra_proto.daycount_convention)
    business_day_convention = business_days.convention_from_proto_value(
        fra_proto.business_day_convention)
    fixed_rate = instrument_utils.decimal_to_double(fra_proto.fixed_rate)
    calendar = business_days.holiday_from_proto_value(
        fra_proto.bank_holidays)
    settlement_days = fra_proto.settlement_days
    name = fra_proto.metadata.id
    instrument_type = fra_proto.metadata.instrument_type
    if h not in prepare_fras:
      prepare_fras[h] = {"short_position": short_position,
                         "currency": currency,
                         "fixing_date": [fixing_date],
                         "fixed_rate": [fixed_rate],
                         "notional_amount": [notional_amount],
                         "daycount_convention": daycount_convention,
                         "business_day_convention": business_day_convention,
                         "calendar": calendar,
                         "rate_term": rate_term,
                         "rate_index": rate_index,
                         "settlement_days": [settlement_days],
                         "fra_config": fra_config,
                         "batch_names": [[name, instrument_type]]}
    else:
      current_index = prepare_fras[h]["rate_index"]
      swap_utils.update_rate_index(current_index, rate_index)
      prepare_fras[h]["fixing_date"].append(fixing_date)
      prepare_fras[h]["fixed_rate"].append(fixed_rate)
      prepare_fras[h]["notional_amount"].append(notional_amount)
      prepare_fras[h]["settlement_days"].append(settlement_days)
      prepare_fras[h]["batch_names"].append([name, instrument_type])
  return prepare_fras
