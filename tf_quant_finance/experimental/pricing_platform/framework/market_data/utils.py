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
"""Utility functions to create an instance of processed market data."""

import functools
from typing import Callable, Dict, Tuple, List, Union

from tf_quant_finance import datetime as dateslib

from tf_quant_finance.experimental.pricing_platform.framework.core import business_days
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2


_BusinessDayConvention = business_days.BusinessDayConvention
_DayCountConventions = daycount_conventions.DayCountConventions
_DayCountConventionsProtoType = types.DayCountConventionsProtoType
_BusinessDayConventionProtoType = types.BusinessDayConventionProtoType

_daycount_map = {
    _DayCountConventions.
        ACTUAL_ACTUAL_ISDA: dateslib.daycount_actual_actual_isda,
    _DayCountConventions.
        ACTUAL_360: dateslib.daycount_actual_360,
    _DayCountConventions.
        ACTUAL_365: dateslib.daycount_actual_365_fixed,
    _DayCountConventions.
        CONVENTION_30_360: dateslib.daycount_thirty_360_isda,}

_business_day_convention_map = {
    # Here True/False means whether to move the payments to the end of month
    _BusinessDayConvention.
        NO_ADJUSTMENT: (dateslib.BusinessDayConvention.NONE, False),
    _BusinessDayConvention.
        FOLLOWING: (dateslib.BusinessDayConvention.FOLLOWING, False),
    _BusinessDayConvention.
        MODIFIED_FOLLOWING: (dateslib.BusinessDayConvention.MODIFIED_FOLLOWING,
                             False),
    _BusinessDayConvention.
        PREVIOUS: (dateslib.BusinessDayConvention.PRECEDING, False),
    _BusinessDayConvention.
        MODIFIED_PREVIOUS: (dateslib.BusinessDayConvention.MODIFIED_PRECEDING,
                            False),
    _BusinessDayConvention.
        EOM_FOLLOWING: (dateslib.BusinessDayConvention.FOLLOWING, True),
    _BusinessDayConvention.
        EOM_PREVIOUS: (dateslib.BusinessDayConvention.PRECEDING, True),
    _BusinessDayConvention.
        EOM_NO_ADJUSTMENT: (dateslib.BusinessDayConvention.NONE, True),}


def get_daycount_fn(
    day_count_convention: Union[_DayCountConventionsProtoType,
                                Callable[..., types.FloatTensor]],
    dtype=None) -> Callable[..., types.FloatTensor]:
  """Converts input to a daycount convention function."""
  if not isinstance(day_count_convention, _DayCountConventionsProtoType):
    def _res(start_date, end_date):
      return day_count_convention(start_date.to_tensor(), end_date.to_tensor())
    return _res
  try:
    daycount_fn = _daycount_map[day_count_convention]
  except KeyError:
    raise KeyError(
        f"{day_count_convention} is not mapped to a daycount function")
  return functools.partial(daycount_fn, dtype=dtype)


def get_business_day_convention(
    business_day_convention: _BusinessDayConventionProtoType
    ) -> Tuple[dateslib.BusinessDayConvention, bool]:
  """Returns business day convention and the end of month flag."""
  try:
    return _business_day_convention_map[business_day_convention]
  except KeyError:
    raise KeyError(
        f"{business_day_convention} is not mapped to a business day convention")


def period_from_list(period: Tuple[int, List[int]]
                     ) -> dateslib.PeriodTensor:
  """Utility to convert a list of periods to a PeriodTensor.

  Args:
    period: A tuple of an integer (which corresponds to the proto type of the
      period (see `period_pb2.Period`)) and a list of period values.

  Returns:
    An instance of the `PeriodTensor`.
  """
  amount = period[1]
  period_type = period_pb2.PeriodType.Name(period[0])
  return dateslib.PeriodTensor(
      amount,
      dateslib.PeriodType[period_type])


def period_from_dict(period: Dict[int, List[int]]
                     ) -> dateslib.PeriodTensor:
  """Utility to convert a dictionary of periods to a PeriodTensor.

  Args:
    period: A dictionary with keys "type" (which corresponds to the proto type
      of the period (see `period_pb2.Period`)) and "frequency".

  Returns:
    An instance of the `PeriodTensor`.
  """
  amount = period["frequency"]
  period_type = period_pb2.PeriodType.Name(period["type"])
  return dateslib.PeriodTensor(
      amount,
      dateslib.PeriodType[period_type])


def get_period(period: period_pb2.Period) -> dateslib.PeriodTensor:
  period_type = period_pb2.PeriodType.Name(period.type)
  return dateslib.PeriodTensor(
      period.amount,
      dateslib.PeriodType[period_type])


def get_yield_and_time(
    discount_curve: pmd.RateCurve,
    valuation_date: types.DateTensor,
    dtype,
    ) -> Tuple[types.FloatTensor, types.FloatTensor]:
  """Extracts yileds and the corresponding times from a RateCurve."""
  discount_nodes = discount_curve.discount_factor_nodes
  node_dates = discount_curve.node_dates
  daycount_fn = discount_curve.daycount_fn()
  times = daycount_fn(start_date=valuation_date,
                      end_date=node_dates,
                      dtype=dtype)
  yields = (1 / discount_nodes)**(1 / times) - 1
  return yields, times


__all__ = ["get_daycount_fn", "get_business_day_convention",
           "get_period", "get_yield_and_time"]
