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
from typing import Any, List, Dict, Tuple, Optional

import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.pricing_platform.framework import utils
from tf_quant_finance.experimental.pricing_platform.framework.core import business_days
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.equity_instruments import utils as equity_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import utils as instrument_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import american_equity_option_pb2 as american_option_pb2


def _get_hash(
    american_option_proto: american_option_pb2.AmericanEquityOption
    ) -> Tuple[int, types.CurrencyProtoType]:
  """Computes hash key for the batching strategy."""
  currency = currencies.from_proto_value(american_option_proto.currency)
  bank_holidays = american_option_proto.bank_holidays
  business_day_convention = american_option_proto.business_day_convention
  h = utils.hasher([bank_holidays, business_day_convention])
  return h, currency


def group_protos(
    proto_list: List[american_option_pb2.AmericanEquityOption],
    config: "AmericanOptionConfig" = None
    ) -> Dict[str, List["AmericanOption"]]:
  """Creates a dictionary of grouped protos."""
  del config  # not used for now
  grouped_options = {}
  for american_option in proto_list:
    h, _ = _get_hash(american_option)
    if h in grouped_options:
      grouped_options[h].append(american_option)
    else:
      grouped_options[h] = [american_option]
  return grouped_options


def from_protos(
    proto_list: List[american_option_pb2.AmericanEquityOption],
    config: "AmericanOptionConfig" = None
    ) -> Dict[str, Any]:
  """Creates a dictionary of preprocessed swap data."""
  prepare_fras = {}
  for am_option_proto in proto_list:
    short_position = am_option_proto.short_position
    h, currency = _get_hash(am_option_proto)
    expiry_date = am_option_proto.expiry_date
    expiry_date = [expiry_date.year,
                   expiry_date.month,
                   expiry_date.day]
    equity = am_option_proto.equity
    contract_amount = instrument_utils.decimal_to_double(
        am_option_proto.contract_amount)
    business_day_convention = business_days.convention_from_proto_value(
        am_option_proto.business_day_convention)
    strike = instrument_utils.decimal_to_double(am_option_proto.strike)
    calendar = business_days.holiday_from_proto_value(
        am_option_proto.bank_holidays)
    settlement_days = am_option_proto.settlement_days
    is_call_option = am_option_proto.is_call_option
    name = am_option_proto.metadata.id
    instrument_type = am_option_proto.metadata.instrument_type
    if h not in prepare_fras:
      prepare_fras[h] = {"short_position": [short_position],
                         "currency": [currency],
                         "expiry_date": [expiry_date],
                         "equity": [equity],
                         "contract_amount": [contract_amount],
                         "business_day_convention": business_day_convention,
                         "calendar": calendar,
                         "strike": [strike],
                         "is_call_option": [is_call_option],
                         "settlement_days": [settlement_days],
                         "config": config,
                         "batch_names": [[name, instrument_type]]}
    else:
      prepare_fras[h]["short_position"].append(short_position)
      prepare_fras[h]["expiry_date"].append(expiry_date)
      prepare_fras[h]["equity"].append(equity)
      prepare_fras[h]["currency"].append(currency)
      prepare_fras[h]["contract_amount"].append(contract_amount)
      prepare_fras[h]["strike"].append(strike)
      prepare_fras[h]["is_call_option"].append(is_call_option)
      prepare_fras[h]["settlement_days"].append(settlement_days)
      prepare_fras[h]["batch_names"].append([name, instrument_type])
  return prepare_fras


def tensor_repr(am_option_data: Dict[str, Any],
                dtype: Optional[types.Dtype] = None):
  """Creates a tensor representation of an American option."""
  dtype = dtype or tf.float64
  res = dict()
  res["expiry_date"] = tf.convert_to_tensor(
      am_option_data["expiry_date"], dtype=tf.int32, name="expiry_date")
  am_option_config = am_option_data["config"]
  res["config"] = None
  if am_option_config is not None:
    res["config"] = config_to_dict(am_option_config)
  res["batch_names"] = am_option_data["batch_names"]
  res["is_call_option"] = tf.convert_to_tensor(
      am_option_data["is_call_option"], dtype=tf.bool,
      name="is_call_options")
  currency = am_option_data["currency"]
  if not isinstance(currency, (list, tuple)):
    currency = [currency]
  discount_curve_type = []
  for cur in currency:
    if res["config"] is not None:
      if cur in res["config"]["discounting_curve"]:
        discount_curve = res["config"]["discounting_curve"][cur]
        discount_curve_type.append(discount_curve)
      else:
        risk_free = curve_types_lib.RiskFreeCurve(currency=cur)
        discount_curve_type.append(risk_free)
    else:
      # Default discounting is the risk free curve
      risk_free = curve_types_lib.RiskFreeCurve(currency=cur)
      discount_curve_type.append(risk_free)
  discount_curve_type, mask = cashflow_streams.process_curve_types(
      discount_curve_type)
  res["discount_curve_mask"] = tf.convert_to_tensor(mask, dtype=tf.int32)
  res["discount_curve_type"] = discount_curve_type
  # Get equity mask
  equity_list = cashflow_streams.to_list(am_option_data["equity"])
  [
      equity,
      equity_mask,
  ] = equity_utils.process_equities(equity_list)
  res["equity_mask"] = tf.convert_to_tensor(equity_mask, tf.int32)
  res["equity"] = equity
  res["contract_amount"] = tf.convert_to_tensor(
      am_option_data["contract_amount"], dtype=dtype)
  res["strike"] = tf.convert_to_tensor(
      am_option_data["strike"], dtype=dtype)
  res["calendar"] = am_option_data["calendar"]
  res["currency"] = [curve.currency for curve in discount_curve_type]
  res["business_day_convention"] = am_option_data["business_day_convention"]
  res["short_position"] = tf.convert_to_tensor(am_option_data["short_position"],
                                               dtype=tf.bool)
  return res


def config_to_dict(am_option_config: "AmericanOptionConfig") -> Dict[str, Any]:
  """Converts AmericanOptionConfig to a dictionary."""
  config = {
      "model": am_option_config.model,
      "discounting_curve": am_option_config.discounting_curve,
      "num_samples": am_option_config.num_samples,
      "num_calibration_samples": am_option_config.num_calibration_samples,
      "num_exercise_times": am_option_config.num_exercise_times,
      "seed": tf.convert_to_tensor(am_option_config.seed, name="seed")
  }
  return config
