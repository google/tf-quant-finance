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
"""Cashflow stream coupons."""
from typing import Any, Callable, List, Union

import dataclasses
import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices
from tf_quant_finance.experimental.pricing_platform.framework.core import types


@dataclasses.dataclass
class FixedCouponSpecs:
  """Fixed coupon leg specification.

  Attributes:
    currency: An instance of `Currency` or a list of `Currency`.
    coupon_frequency: A `PeriodTensor` specifying the term of the underlying
      rate which determines the coupon payments.
    notional_amount: A real `Tensor` of  `batch_shape` specifying the notional
      for the payments.
    fixed_rate: A `Tensor` of the same `dtype` as `notional_amount` and
     of the shape that broadcasts with `batch_shape`. Represents the fixed
     rate of the leg.
    daycount_convention: An instance of `DayCountConventions`.
    businessday_rule: An instance of `BusinessDayConvention`.
    settlement_days: An integer `Tensor` of the shape broadcastable with the
      shape of `notional_amount`.
    calendar: A calendar to specify the weekend mask and bank holidays.
  """
  currency: Union[types.CurrencyProtoType, List[types.CurrencyProtoType]]
  coupon_frequency: types.Period
  notional_amount: Union[tf.Tensor, List[float]]
  fixed_rate: Union[tf.Tensor, List[float]]
  daycount_convention: Union[types.DayCountConventionsProtoType,
                             Callable[..., Any]]
  businessday_rule: types.BusinessDayConventionProtoType
  settlement_days: Union[int, List[int]]
  calendar: types.BankHolidaysProtoType


@dataclasses.dataclass
class FloatCouponSpecs:
  """Float rate leg specification.

  Attributes:
    currency: An instance of `Currency`.
    reset_frequency: A `PeriodTensor` specifying the frequency with which the
      underlying floating rate resets.
    coupon_frequency: A `PeriodTensor` specifying the term of the underlying
      rate which determines the coupon payments.
    notional_amount: A real `Tensor` of  `batch_shape` specifying the notional
      for the payments.
    floating_rate_type: A type of the floating leg. An instance of
        `core.rate_indices.RateIndex`.
    daycount_convention: An instance of `DayCountConventions`.
    businessday_rule: An instance of `BusinessDayConvention`.
    settlement_days: An integer `Tensor` of the shape broadcastable with the
      shape of `notional_amount`.
    spread: A `Tensor` of the same `dtype` as `notional_amount` and of the shape
      that broadcasts with `batch_shape`. Represents the spread for the floating
      leg.
    calendar: A calendar to specify the weekend mask and bank holidays.
  """
  currency: types.CurrencyProtoType
  reset_frequency: types.Period
  coupon_frequency: types.Period
  notional_amount: Union[tf.Tensor, List[float]]
  floating_rate_type: rate_indices.RateIndex
  daycount_convention: Union[types.DayCountConventionsProtoType,
                             Callable[..., Any]]
  businessday_rule: types.BusinessDayConventionProtoType
  settlement_days: Union[int, List[int]]
  spread: Union[tf.Tensor, List[tf.Tensor]]
  calendar: types.BankHolidaysProtoType


__all__ = ["FixedCouponSpecs", "FloatCouponSpecs"]
