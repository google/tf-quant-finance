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

import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.pricing_platform.framework import utils
from tf_quant_finance.experimental.pricing_platform.framework.core import business_days
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
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
  h = utils.hasher([currency.value, bank_holidays, rate_term.type,
                    rate_term.amount, rate_index.type.value,
                    daycount_convention, business_day_convention])
  return h, currency, rate_term, rate_index


def _get_hash_v2(
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

  rate_term = fra_proto.floating_rate_term.term
  rate_type, multiplier = _frequency_and_multiplier(rate_term.type)
  h = utils.hasher([bank_holidays, daycount_convention,
                    business_day_convention, rate_type])
  return h, currency, (rate_type, [multiplier * rate_term.amount]), rate_index


def group_protos(
    proto_list: List[fra.ForwardRateAgreement],
    config: "ForwardRateAgreementConfig" = None
    ) -> Dict[str, List["ForwardRateAgreement"]]:
  """Creates a dictionary of grouped protos."""
  del config  # config does not impact the batching
  grouped_fras = {}
  for fra_proto in proto_list:
    h, _, _, _ = _get_hash(fra_proto)
    if h in grouped_fras:
      grouped_fras[h].append(fra_proto)
    else:
      grouped_fras[h] = [fra_proto]
  return grouped_fras


def group_protos_v2(
    proto_list: List[fra.ForwardRateAgreement],
    config: "ForwardRateAgreementConfig" = None
    ) -> Dict[str, List["ForwardRateAgreement"]]:
  """Creates a dictionary of grouped protos."""
  del config  # config does not impact the batching
  grouped_fras = {}
  for fra_proto in proto_list:
    h, _, _, _ = _get_hash_v2(fra_proto)
    if h in grouped_fras:
      grouped_fras[h].append(fra_proto)
    else:
      grouped_fras[h] = [fra_proto]
  return grouped_fras


def from_protos(
    proto_list: List[fra.ForwardRateAgreement],
    config: "ForwardRateAgreementConfig" = None
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
      prepare_fras[h] = {"short_position": [short_position],
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
                         "config": config,
                         "batch_names": [[name, instrument_type]]}
    else:
      current_index = prepare_fras[h]["rate_index"]
      swap_utils.update_rate_index(current_index, rate_index)
      prepare_fras[h]["fixing_date"].append(fixing_date)
      prepare_fras[h]["short_position"].append(short_position)
      prepare_fras[h]["fixed_rate"].append(fixed_rate)
      prepare_fras[h]["notional_amount"].append(notional_amount)
      prepare_fras[h]["settlement_days"].append(settlement_days)
      prepare_fras[h]["batch_names"].append([name, instrument_type])
  return prepare_fras


def from_protos_v2(
    proto_list: List[fra.ForwardRateAgreement],
    config: "ForwardRateAgreementConfig" = None
    ) -> Dict[str, Any]:
  """Creates a dictionary of preprocessed swap data."""
  prepare_fras = {}
  for fra_proto in proto_list:
    short_position = fra_proto.short_position
    h, currency, rate_term, rate_index = _get_hash_v2(fra_proto)
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
      prepare_fras[h] = {"short_position": [short_position],
                         "currency": [currency],
                         "fixing_date": [fixing_date],
                         "fixed_rate": [fixed_rate],
                         "notional_amount": [notional_amount],
                         "daycount_convention": daycount_convention,
                         "business_day_convention": business_day_convention,
                         "calendar": calendar,
                         "rate_term": rate_term,
                         "rate_index": [rate_index],
                         "settlement_days": [settlement_days],
                         "config": config,
                         "batch_names": [[name, instrument_type]]}
    else:
      prepare_fras[h]["currency"].append(currency)
      prepare_fras[h]["fixing_date"].append(fixing_date)
      prepare_fras[h]["short_position"].append(short_position)
      prepare_fras[h]["fixed_rate"].append(fixed_rate)
      prepare_fras[h]["rate_index"].append(rate_index)
      current_rate_term = prepare_fras[h]["rate_term"]
      rate_term_type = current_rate_term[0]
      rate_term_amount = current_rate_term[1]
      prepare_fras[h]["rate_term"] = (rate_term_type,
                                      rate_term_amount + rate_term[1])
      prepare_fras[h]["notional_amount"].append(notional_amount)
      prepare_fras[h]["settlement_days"].append(settlement_days)
      prepare_fras[h]["batch_names"].append([name, instrument_type])
  return prepare_fras


def tensor_repr(fra_data, dtype=None):
  """Creates a tensor representation of the FRA."""
  dtype = dtype or tf.float64
  res = dict()
  res["fixing_date"] = tf.convert_to_tensor(
      fra_data["fixing_date"], dtype=tf.int32)
  res["fixed_rate"] = tf.convert_to_tensor(
      fra_data["fixed_rate"], dtype=dtype)
  config = fra_data["config"]
  res["config"] = config
  res["batch_names"] = fra_data["batch_names"]
  currency_list = cashflow_streams.to_list(fra_data["currency"])
  discount_curve_type = []
  for currency in currency_list:
    if config is not None:
      if currency in config.discounting_curve:
        discount_curve = config.discounting_curve[currency]
        discount_curve_type.append(discount_curve)
      else:
        risk_free = curve_types_lib.RiskFreeCurve(currency=currency)
        discount_curve_type.append(risk_free)
    else:
      # Default discounting is the risk free curve
      risk_free = curve_types_lib.RiskFreeCurve(currency=currency)
      discount_curve_type.append(risk_free)
  discount_curve_type, mask = cashflow_streams.process_curve_types(
      discount_curve_type)
  res["discount_curve_mask"] = tf.convert_to_tensor(mask, dtype=tf.int32)
  res["discount_curve_type"] = discount_curve_type
  # Get reset frequency
  reset_frequency = tf.convert_to_tensor(
      fra_data["rate_term"][1], tf.int32)
  res["rate_term"] = {
      "type": fra_data["rate_term"][0],
      "frequency": reset_frequency}
  rate_index = cashflow_streams.to_list(fra_data["rate_index"])
  rate_index_curves = []
  for currency, r_ind in zip(currency_list, rate_index):
    rate_index_curves.append(curve_types_lib.RateIndexCurve(
        currency=currency, index=r_ind))
  [
      rate_index_curves,
      reference_mask
  ] = cashflow_streams.process_curve_types(rate_index_curves)
  res["reference_mask"] = tf.convert_to_tensor(reference_mask, tf.int32)
  res["rate_index_curves"] = rate_index_curves
  # Extract unique rate indices
  res["rate_index"] = [curve.index for curve in rate_index_curves]
  res["notional_amount"] = tf.convert_to_tensor(
      fra_data["notional_amount"], dtype=dtype)
  res["settlement_days"] = tf.convert_to_tensor(
      fra_data["settlement_days"], dtype=tf.int32, name="settlement_days")
  res["calendar"] = fra_data["calendar"]
  # Extract unique currencies
  res["currency"] = [curve.currency for curve in discount_curve_type]
  res["daycount_convention"] = fra_data["daycount_convention"]
  res["business_day_convention"] = fra_data["business_day_convention"]
  res["short_position"] = tf.convert_to_tensor(fra_data["short_position"],
                                               dtype=tf.bool)
  return res


def _frequency_and_multiplier(freq_type):
  multiplier = 1
  if freq_type == 5:
    freq_type = 3
  return freq_type, multiplier



