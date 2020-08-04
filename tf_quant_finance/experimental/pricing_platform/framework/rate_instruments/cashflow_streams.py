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
"""Cashflow streams objects."""

from typing import Optional, Tuple, Callable, Any

import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import coupon_specs
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2

_CurveType = curve_types.CurveType


class FixedCashflowStream:
  """Represents a batch of fixed stream of cashflows."""

  def __init__(self,
               start_date: types.DateTensor,
               end_date: types.DateTensor,
               coupon_spec: coupon_specs.FixedCouponSpecs,
               discount_curve_type: _CurveType,
               first_coupon_date: Optional[types.DateTensor] = None,
               penultimate_coupon_date: Optional[types.DateTensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes a batch of fixed cashflow streams.

    Args:
      start_date: A `DateTensor` of `batch_shape` specifying the starting dates
        of the accrual of the first coupon of the cashflow stream. The shape of
        the input correspond to the number of streams being created.
      end_date: A `DateTensor` of `batch_shape`specifying the end dates for
        accrual of the last coupon in each cashflow stream. The shape of the
        input should be the same as that of `start_date`.
      coupon_spec: An instance of `FixedCouponSpecs` specifying the
        details of the coupon payment for the cashflow stream.
      discount_curve_type: An instance of `CurveType`.
      first_coupon_date: An optional `DateTensor` specifying the payment dates
        of the first coupon of the cashflow stream. Use this input for cashflows
        with irregular first coupon. Should be of the same shape as
        `start_date`.
        Default value: None which implies regular first coupon.
      penultimate_coupon_date: An optional `DateTensor` specifying the payment
        dates of the penultimate (next to last) coupon of the cashflow
        stream. Use this input for cashflows with irregular last coupon.
        Should be of the same shape as `end_date`.
        Default value: None which implies regular last coupon.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'fixed_cashflow_stream'.
    """
    self._name = name or "fixed_cashflow_stream"

    with tf.name_scope(self._name):
      self._discount_curve_type = discount_curve_type
      self._start_date = dateslib.convert_to_date_tensor(start_date)
      self._end_date = dateslib.convert_to_date_tensor(end_date)
      self._first_coupon_date = first_coupon_date
      self._penultimate_coupon_date = penultimate_coupon_date
      if self._first_coupon_date is not None:
        self._first_coupon_date = dateslib.convert_to_date_tensor(
            first_coupon_date)
      if self._penultimate_coupon_date is not None:
        self._penultimate_coupon_date = dateslib.convert_to_date_tensor(
            penultimate_coupon_date)

      coupon_frequency = coupon_spec.coupon_frequency
      if isinstance(coupon_frequency, period_pb2.Period):
        coupon_frequency = market_data_utils.get_period(
            coupon_spec.coupon_frequency)

      businessday_rule = coupon_spec.businessday_rule
      # Business day roll convention and the end of month flag
      roll_convention, eom = market_data_utils.get_business_day_convention(
          businessday_rule)

      notional = tf.convert_to_tensor(
          coupon_spec.notional_amount,
          dtype=dtype,
          name="notional")
      self._dtype = dtype or notional.dtype
      fixed_rate = tf.convert_to_tensor(coupon_spec.fixed_rate,
                                        dtype=self._dtype,
                                        name="fixed_rate")
      # TODO(b/160446193): Calendar is ignored and weekends only is used
      calendar = dateslib.create_holiday_calendar(
          weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
      daycount_fn = market_data_utils.get_daycount_fn(
          coupon_spec.daycount_convention)

      self._settlement_days = tf.convert_to_tensor(
          coupon_spec.settlement_days,
          dtype=tf.int32,
          name="settlement_days")

      coupon_dates = _generate_schedule(
          start_date=self._start_date,
          end_date=self._end_date,
          coupon_frequency=coupon_frequency,
          roll_convention=roll_convention,
          calendar=calendar,
          settlement_days=self._settlement_days,
          end_of_month=eom,
          first_coupon_date=self._first_coupon_date,
          penultimate_coupon_date=self._penultimate_coupon_date)

      self._batch_shape = coupon_dates.shape.as_list()[:-1]
      payment_dates = coupon_dates[..., 1:]

      daycount_fractions = daycount_fn(start_date=coupon_dates[..., :-1],
                                       end_date=coupon_dates[..., 1:],
                                       dtype=self._dtype)

      coupon_rate = tf.expand_dims(fixed_rate, axis=-1)

      self._num_cashflows = payment_dates.shape.as_list()[-1]
      self._payment_dates = payment_dates
      self._notional = notional
      self._daycount_fractions = daycount_fractions
      self._coupon_rate = coupon_rate
      self._calendar = coupon_rate
      self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype)
      self._daycount_fn = daycount_fn

  def daycount_fn(self) -> Callable[..., Any]:
    return self._daycount_fn

  @property
  def daycount_fractions(self) -> types.FloatTensor:
    return self._daycount_fractions

  @property
  def fixed_rate(self) -> types.FloatTensor:
    return self._fixed_rate

  @property
  def notional(self) -> types.FloatTensor:
    return self._notional

  @property
  def discount_curve_type(self) -> _CurveType:
    return self._discount_curve_type

  @property
  def batch_shape(self) -> types.StringTensor:
    return self._batch_shape

  @property
  def cashflow_dates(self) -> types.DateTensor:
    return self._payment_dates

  def cashflows(self,
                market: pmd.ProcessedMarketData,
                name: Optional[str] = None
                ) -> Tuple[types.DateTensor, types.FloatTensor]:
    """Returns cashflows for the fixed leg.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'cashflows'.

    Returns:
      A tuple of two `Tensor`s of shape `batch_shape + [num_cashflows]` and
      containing the dates and the corresponding cashflows price for each
      stream based on the input market data.
    """
    name = name or (self._name + "_cashflows")
    with tf.name_scope(name):
      valuation_date = dateslib.convert_to_date_tensor(market.date)
      future_cashflows = tf.cast(self._payment_dates >= valuation_date,
                                 dtype=self._dtype)
      # self._notional is of shape [batch_shape], so broadcasting is needed
      notional = tf.expand_dims(self._notional, axis=-1)
      # Cashflow present values.
      cashflows = notional * (
          future_cashflows * self._daycount_fractions * self._coupon_rate)
      return  self._payment_dates, cashflows

  def price(self,
            market: pmd.ProcessedMarketData,
            name: Optional[str] = None):
    """Returns the present value of the stream on the valuation date.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A `Tensor` of shape `batch_shape`  containing the modeled price of each
      stream based on the input market data.
    """
    name = name or (self._name + "_price")
    with tf.name_scope(name):
      discount_curve = market.yield_curve(self._discount_curve_type)
      discount_factors = discount_curve.discount_factor(
          self._payment_dates)
      _, cashflows = self.cashflows(market)
      # Cashflow present values
      cashflow_pvs = (cashflows * discount_factors)
      return tf.math.reduce_sum(cashflow_pvs, axis=1)


class FloatingCashflowStream:
  """Represents a batch of cashflows indexed to a floating rate."""

  def __init__(self,
               start_date: types.DateTensor,
               end_date: types.DateTensor,
               coupon_spec: coupon_specs.FloatCouponSpecs,
               discount_curve_type: _CurveType,
               first_coupon_date: Optional[types.DateTensor] = None,
               penultimate_coupon_date: Optional[types.DateTensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes a batch of floating cashflow streams.

    Args:
      start_date: A `DateTensor` of `batch_shape` specifying the starting dates
        of the accrual of the first coupon of the cashflow stream. The shape of
        the input correspond to the number of streams being created.
      end_date: A `DateTensor` of `batch_shape`specifying the end dates for
        accrual of the last coupon in each cashflow stream. The shape of the
        input should be the same as that of `start_date`.
      coupon_spec: An instance of `FloatCouponSpecs` specifying the
        details of the coupon payment for the cashflow stream.
      discount_curve_type: An instance of `CurveType`.
      first_coupon_date: An optional `DateTensor` specifying the payment dates
        of the first coupon of the cashflow stream. Use this input for cashflows
        with irregular first coupon. Should be of the same shape as
        `start_date`.
        Default value: None which implies regular first coupon.
      penultimate_coupon_date: An optional `DateTensor` specifying the payment
        dates of the penultimate (next to last) coupon of the cashflow
        stream. Use this input for cashflows with irregular last coupon.
        Should be of the same shape as `end_date`.
        Default value: None which implies regular last coupon.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'floating_cashflow_stream'.
    """

    self._name = name or "floating_cashflow_stream"
    with tf.name_scope(self._name):
      self._discount_curve_type = discount_curve_type
      self._floating_rate_type = coupon_spec.floating_rate_type
      self._first_coupon_date = None
      self._penultimate_coupon_date = None
      self._start_date = dateslib.convert_to_date_tensor(start_date)
      self._end_date = dateslib.convert_to_date_tensor(end_date)
      if self._first_coupon_date is not None:
        self._first_coupon_date = dateslib.convert_to_date_tensor(
            first_coupon_date)
      if self._penultimate_coupon_date is not None:
        self._penultimate_coupon_date = dateslib.convert_to_date_tensor(
            penultimate_coupon_date)
      # Ignored and weekends only is used
      calendar = dateslib.create_holiday_calendar(
          weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
      # Convert coupon and reset frequencies to PeriodTensor
      coupon_frequency = coupon_spec.coupon_frequency
      if isinstance(coupon_frequency, period_pb2.Period):
        coupon_frequency = market_data_utils.get_period(
            coupon_spec.coupon_frequency)
      self._reset_frequency = coupon_spec.reset_frequency
      reset_frequency = coupon_spec.reset_frequency
      if isinstance(reset_frequency, period_pb2.Period):
        reset_frequency = market_data_utils.get_period(
            coupon_spec.reset_frequency)

      businessday_rule = coupon_spec.businessday_rule
      roll_convention, eom = market_data_utils.get_business_day_convention(
          businessday_rule)
      notional = tf.convert_to_tensor(
          coupon_spec.notional_amount,
          dtype=dtype,
          name="notional")
      self._dtype = dtype or notional.dtype

      daycount_convention = coupon_spec.daycount_convention
      daycount_fn = market_data_utils.get_daycount_fn(
          coupon_spec.daycount_convention)
      self._daycount_convention = daycount_convention

      self._settlement_days = tf.convert_to_tensor(
          coupon_spec.settlement_days,
          dtype=tf.int32,
          name="settlement_days")
      spread = tf.convert_to_tensor(coupon_spec.spread,
                                    dtype=self._dtype,
                                    name="spread")

      coupon_dates = _generate_schedule(
          start_date=self._start_date,
          end_date=self._end_date,
          coupon_frequency=coupon_frequency,
          roll_convention=roll_convention,
          calendar=calendar,
          settlement_days=self._settlement_days,
          end_of_month=eom,
          first_coupon_date=self._first_coupon_date,
          penultimate_coupon_date=self._penultimate_coupon_date)
      # Extract batch shape
      self._batch_shape = coupon_dates.shape.as_list()[:-1]

      accrual_start_dates = coupon_dates[..., :-1]

      coupon_start_dates = coupon_dates[..., :-1]
      coupon_end_dates = coupon_dates[..., 1:]

      accrual_end_dates = accrual_start_dates + reset_frequency.expand_dims(
          axis=-1)

      # Adjust for irregular coupons
      accrual_end_dates = dateslib.DateTensor.concat(
          [coupon_end_dates[..., :1],
           accrual_end_dates[..., 1:-1],
           coupon_end_dates[..., -1:]], axis=-1)
      daycount_fractions = daycount_fn(
          start_date=coupon_start_dates,
          end_date=coupon_end_dates,
          dtype=self._dtype)

      self._num_cashflows = daycount_fractions.shape.as_list()[-1]
      self._coupon_start_dates = coupon_start_dates
      self._coupon_end_dates = coupon_end_dates
      self._accrual_start_date = accrual_start_dates
      self._accrual_end_date = accrual_end_dates
      self._notional = notional
      self._daycount_fractions = daycount_fractions
      self._spread = spread
      self._currency = coupon_spec.currency
      self._daycount_fn = daycount_fn
      # Construct the reference curve object
      rate_index_curve = curve_types.RateIndexCurve(
          currency=self._currency, index=self._floating_rate_type)
      self._reference_curve_type = rate_index_curve

  def daycount_fn(self) -> Callable[..., Any]:
    return self._daycount_fn

  @property
  def notional(self) -> types.FloatTensor:
    return self._notional

  @property
  def discount_curve_type(self) -> _CurveType:
    return self._discount_curve_type

  @property
  def reference_curve_type(self) -> _CurveType:
    return self._reference_curve_type

  @property
  def batch_shape(self) -> types.StringTensor:
    return self._batch_shape

  @property
  def daycount_fractions(self) -> types.FloatTensor:
    return self._daycount_fractions

  @property
  def cashflow_dates(self) -> types.DateTensor:
    return self._coupon_end_dates

  def forward_rates(self,
                    market: pmd.ProcessedMarketData,
                    name: Optional[str] = None
                    ) -> Tuple[types.DateTensor, types.FloatTensor]:
    """Returns forward rates for the floating leg.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'forward_rates'.

    Returns:
      A tuple of two `Tensor`s of shape `batch_shape + [num_cashflows]`
      containing the dates and the corresponding forward rates for each stream
      based on the input market data.
    """
    name = name or (self._name + "_forward_rates")
    with tf.name_scope(name):
      reference_curve = market.yield_curve(self._reference_curve_type)
      valuation_date = dateslib.convert_to_date_tensor(market.date)
      # TODO(cyrilchimisov): vectorize fixing computation
      past_fixing = market.fixings(
          self._start_date,
          self._reference_curve_type,
          self._reset_frequency)
      forward_rates = reference_curve.forward_rate(
          self._accrual_start_date,
          self._accrual_end_date,
          day_count_fraction=self._daycount_fractions)
      forward_rates = tf.where(self._daycount_fractions > 0., forward_rates,
                               tf.zeros_like(forward_rates))
      # If coupon end date is before the valuation date, the payment is in the
      # past. If valuation date is between coupon start date and coupon end
      # date, then the rate has been fixed but not paid. Otherwise the rate is
      # not fixed and should be read from the curve.
      forward_rates = tf.where(
          self._coupon_end_dates < valuation_date,
          tf.constant(0, dtype=self._dtype),
          tf.where(self._coupon_start_dates >= valuation_date,
                   forward_rates, tf.expand_dims(past_fixing, axis=-1)))
      return  self._coupon_end_dates, forward_rates

  def cashflows(self,
                market: pmd.ProcessedMarketData,
                name: Optional[str] = None
                ) -> Tuple[types.DateTensor, types.FloatTensor]:
    """Returns cashflows for the floating leg.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'cashflows'.

    Returns:
      A tuple of two `Tensor`s of shape `batch_shape + [num_cashflows]` and
      containing the dates and the corresponding cashflows price for each
      stream based on the input market data.
    """
    name = name or (self._name + "_cashflows")
    with tf.name_scope(name):
      _, forward_rates = self.forward_rates(market)

      coupon_rate = forward_rates + tf.expand_dims(
          self._spread, axis=-1)
      # self._notion is of shape [batch_shape], so broadcasting is needed
      notional = tf.expand_dims(self._notional, axis=-1)

      cashflows = notional * (
          self._daycount_fractions * coupon_rate)
      return  self._coupon_end_dates, cashflows

  def price(self,
            market: pmd.ProcessedMarketData,
            name: Optional[str] = None) -> types.FloatTensor:
    """Returns the present value of the stream on the valuation date.

    Args:
      market: An instance of `ProcessedMarketData`.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A `Tensor` of shape `batch_shape`  containing the modeled price of each
      stream based on the input market data.
    """

    name = name or (self._name + "_price")
    with tf.name_scope(name):
      discount_curve = market.yield_curve(self._discount_curve_type)
      discount_factors = discount_curve.discount_factor(self._coupon_end_dates)
      _, cashflows = self.cashflows(market)
      # Cashflows present values
      cashflow_pvs = cashflows * discount_factors
      return tf.math.reduce_sum(cashflow_pvs, axis=1)


def _generate_schedule(
    start_date: dateslib.DateTensor,
    end_date: dateslib.DateTensor,
    coupon_frequency: dateslib.PeriodTensor,
    calendar: dateslib.HolidayCalendar,
    roll_convention: dateslib.BusinessDayConvention,
    settlement_days: tf.Tensor,
    end_of_month: bool = False,
    first_coupon_date: Optional[dateslib.DateTensor] = None,
    penultimate_coupon_date: Optional[dateslib.DateTensor] = None) -> tf.Tensor:
  """Method to generate coupon dates.

  Args:
    start_date: Starting dates of schedule.
    end_date: End dates of the schedule.
    coupon_frequency: A `PeriodTensor` specifying the frequency of coupon
      payments.
    calendar: calendar: An instance of `BankHolidays`.
    roll_convention: Business day roll convention of the schedule.
    settlement_days: An integer `Tensor` with the shape compatible with
      `start_date` and `end_date` specifying the number of settlement days.
    end_of_month: Python `bool`. If `True`, shifts all dates in schedule to
      the ends of corresponding months, if `start_date` or `end_date` (
      depending on `backward`) is at the end of a month. The shift is applied
      before applying `roll_convention`.
    first_coupon_date: First day of the irregular coupon, if any.
    penultimate_coupon_date: Penultimate day of the coupon, if any.

  Returns:
    A `DateTensor` containing the generated date schedule of shape
    `batch_shape + [max_num_coupon_days]`, where `max_num_coupon_days` is the
    number of coupon days for the longest living swap in the batch. The coupon
    days for the rest of the swaps are padded with their final coupon day.
  """
  if first_coupon_date is not None and penultimate_coupon_date is not None:
    raise ValueError("Only first or last coupon dates can be specified "
                     " for an irregular coupon.")
  start_date = first_coupon_date or start_date
  # Adjust with settlement days
  start_date = calendar.add_business_days(
      start_date, settlement_days,
      roll_convention=roll_convention)
  if penultimate_coupon_date is None:
    backward = False
  else:
    backward = True
    end_date = end_date or penultimate_coupon_date
  # Adjust with settlement days
  end_date = calendar.add_business_days(
      end_date, settlement_days,
      roll_convention=roll_convention)
  coupon_dates = dateslib.PeriodicSchedule(
      start_date=start_date,
      end_date=end_date,
      tenor=coupon_frequency,
      roll_convention=roll_convention,
      backward=backward,
      end_of_month=end_of_month).dates()
  # Add the regular coupons
  coupon_dates = dateslib.DateTensor.concat(
      [start_date.expand_dims(-1),
       coupon_dates,
       end_date.expand_dims(-1)], axis=-1)
  return coupon_dates


__all__ = ["FixedCashflowStream", "FloatingCashflowStream"]
