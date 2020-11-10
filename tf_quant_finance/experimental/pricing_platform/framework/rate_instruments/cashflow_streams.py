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

from typing import Optional, Tuple, Callable, Any, List, Union

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import rate_curve
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import coupon_specs
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2
from tf_quant_finance.math import pad


_CurveType = curve_types_lib.CurveType


class FixedCashflowStream:
  """Represents a batch of fixed stream of cashflows."""

  def __init__(self,
               coupon_spec: coupon_specs.FixedCouponSpecs,
               discount_curve_type: _CurveType,
               start_date: types.DateTensor = None,
               end_date: types.DateTensor = None,
               discount_curve_mask: types.IntTensor = None,
               first_coupon_date: Optional[types.DateTensor] = None,
               penultimate_coupon_date: Optional[types.DateTensor] = None,
               schedule_fn: Optional[Callable[..., Any]] = None,
               schedule: Optional[types.DateTensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes a batch of fixed cashflow streams.

    Args:
      coupon_spec: An instance of `FixedCouponSpecs` specifying the
        details of the coupon payment for the cashflow stream.
      discount_curve_type: An instance of `CurveType` or a list of those.
        If supplied as a list and `discount_curve_mask` is not supplied,
        the size of the list should be the same as the number of priced
        instruments.
      start_date: A `DateTensor` of `batch_shape` specifying the starting dates
        of the accrual of the first coupon of the cashflow stream. The shape of
        the input correspond to the number of streams being created.
        Either this of `schedule` should be supplied
        Default value: `None`
      end_date: A `DateTensor` of `batch_shape`specifying the end dates for
        accrual of the last coupon in each cashflow stream. The shape of the
        input should be the same as that of `start_date`.
        Either this of `schedule` should be supplied
        Default value: `None`
      discount_curve_mask: An optional integer `Tensor` of values ranging from
        `0` to `len(discount_curve_type)` and of shape `batch_shape`. Identifies
        a mapping between `discount_curve_type` list and the underlying
        instruments.
        Default value: `None`.
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
      schedule_fn: A callable that accepts `start_date`, `end_date`,
        `coupon_frequency`, `settlement_days`, `first_coupon_date`, and
        `penultimate_coupon_date` as `Tensor`s and returns coupon payment
        days.
        Default value: `None`.
      schedule: A `DateTensor` of coupon payment dates.
        Default value: `None`.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'fixed_cashflow_stream'.
    """
    self._name = name or "fixed_cashflow_stream"

    with tf.name_scope(self._name):
      curve_list = to_list(discount_curve_type)
      [
          self._discount_curve_type,
          self._mask
      ] = process_curve_types(curve_list, discount_curve_mask)

      if schedule is None:
        if (start_date is None) or (end_date is None):
          raise ValueError("If `schedule` is not supplied both "
                           "`start_date` and `end_date` should be supplied")

      if schedule is None:
        if isinstance(start_date, tf.Tensor):
          self._start_date = dateslib.dates_from_tensor(
              start_date)
        else:
          self._start_date = dateslib.convert_to_date_tensor(
              start_date)
        if isinstance(start_date, tf.Tensor):
          self._end_date = dateslib.dates_from_tensor(
              end_date)
        else:
          self._end_date = dateslib.convert_to_date_tensor(
              end_date)
        self._first_coupon_date = first_coupon_date
        self._penultimate_coupon_date = penultimate_coupon_date
        if self._first_coupon_date is not None:
          if isinstance(start_date, tf.Tensor):
            self._first_coupon_date = dateslib.dates_from_tensor(
                first_coupon_date)
          else:
            self._first_coupon_date = dateslib.convert_to_date_tensor(
                first_coupon_date)
        if self._penultimate_coupon_date is not None:
          if isinstance(start_date, tf.Tensor):
            self._penultimate_coupon_date = dateslib.dates_from_tensor(
                penultimate_coupon_date)
          else:
            self._penultimate_coupon_date = dateslib.convert_to_date_tensor(
                penultimate_coupon_date)

      # Update coupon frequency
      coupon_frequency = _get_attr(coupon_spec, "coupon_frequency")
      if isinstance(coupon_frequency, period_pb2.Period):
        coupon_frequency = market_data_utils.get_period(
            _get_attr(coupon_spec, "coupon_frequency"))
      if isinstance(coupon_frequency, (list, tuple)):
        coupon_frequency = market_data_utils.period_from_list(
            *_get_attr(coupon_spec, "coupon_frequency"))
      if isinstance(coupon_frequency, dict):
        coupon_frequency = market_data_utils.period_from_dict(
            _get_attr(coupon_spec, "coupon_frequency"))

      businessday_rule = coupon_spec.businessday_rule
      # Business day roll convention and the end of month flag
      roll_convention, eom = market_data_utils.get_business_day_convention(
          businessday_rule)

      notional = tf.convert_to_tensor(
          _get_attr(coupon_spec, "notional_amount"),
          dtype=dtype,
          name="notional")
      self._dtype = dtype or notional.dtype
      fixed_rate = tf.convert_to_tensor(_get_attr(coupon_spec, "fixed_rate"),
                                        dtype=self._dtype,
                                        name="fixed_rate")
      # TODO(b/160446193): Calendar is ignored and weekends only is used
      calendar = dateslib.create_holiday_calendar(
          weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
      daycount_fn = market_data_utils.get_daycount_fn(
          _get_attr(coupon_spec, "daycount_convention"), self._dtype)

      self._settlement_days = tf.convert_to_tensor(
          _get_attr(coupon_spec, "settlement_days"),
          dtype=tf.int32,
          name="settlement_days")

      if schedule is not None:
        if isinstance(start_date, tf.Tensor):
          coupon_dates = dateslib.dates_from_tensor(schedule)
        else:
          coupon_dates = dateslib.convert_to_date_tensor(schedule)
      elif schedule_fn is None:
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
      else:
        if first_coupon_date is not None:
          first_coupon_date = self._first_coupon_date.to_tensor()
        if penultimate_coupon_date is not None:
          penultimate_coupon_date = self._penultimate_coupon_date.to_tensor()
          coupon_dates = schedule_fn(
              start_date=self._start_date.to_tensor(),
              end_date=self._end_date.to_tensor(),
              coupon_frequency=coupon_frequency.quantity(),
              settlement_days=self._settlement_days,
              first_coupon_date=first_coupon_date,
              penultimate_coupon_date=penultimate_coupon_date)

      # Convert to DateTensor if the result comes from a tf.function
      coupon_dates = dateslib.convert_to_date_tensor(coupon_dates)

      self._batch_shape = tf.shape(coupon_dates.ordinal())[:-1]
      payment_dates = coupon_dates[..., 1:]

      daycount_fractions = daycount_fn(
          start_date=coupon_dates[..., :-1],
          end_date=coupon_dates[..., 1:])

      coupon_rate = tf.expand_dims(fixed_rate, axis=-1)

      self._num_cashflows = tf.shape(payment_dates.ordinal())[-1]
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
      discount_curve = get_discount_curve(
          self._discount_curve_type, market, self._mask)
      discount_factors = discount_curve.discount_factor(
          self._payment_dates)
      _, cashflows = self.cashflows(market)
      # Cashflow present values
      cashflow_pvs = (cashflows * discount_factors)
      return tf.math.reduce_sum(cashflow_pvs, axis=1)


class FloatingCashflowStream:
  """Represents a batch of cashflows indexed to a floating rate."""

  def __init__(self,
               coupon_spec: coupon_specs.FloatCouponSpecs,
               discount_curve_type: _CurveType,
               start_date: types.DateTensor = None,
               end_date: types.DateTensor = None,
               discount_curve_mask: types.IntTensor = None,
               rate_index_curves: curve_types_lib.RateIndexCurve = None,
               reference_mask: types.IntTensor = None,
               first_coupon_date: Optional[types.DateTensor] = None,
               penultimate_coupon_date: Optional[types.DateTensor] = None,
               schedule_fn: Optional[Callable[..., Any]] = None,
               schedule: Optional[types.DateTensor] = None,
               dtype: Optional[types.Dtype] = None,
               name: Optional[str] = None):
    """Initializes a batch of floating cashflow streams.

    Args:
      coupon_spec: An instance of `FloatCouponSpecs` specifying the
        details of the coupon payment for the cashflow stream.
      discount_curve_type: An instance of `CurveType` or a list of those.
        If supplied as a list and `discount_curve_mask` is not supplied,
        the size of the list should be the same as the number of priced
        instruments.
      start_date: A `DateTensor` of `batch_shape` specifying the starting dates
        of the accrual of the first coupon of the cashflow stream. The shape of
        the input correspond to the number of streams being created.
        Either this of `schedule` should be supplied
        Default value: `None`
      end_date: A `DateTensor` of `batch_shape`specifying the end dates for
        accrual of the last coupon in each cashflow stream. The shape of the
        input should be the same as that of `start_date`.
        Either this of `schedule` should be supplied
        Default value: `None`
      discount_curve_mask: An optional integer `Tensor` of values ranging from
        `0` to `len(discount_curve_type)` and of shape `batch_shape`. Identifies
        a mapping between `discount_curve_type` list and the underlying
        instruments.
        Default value: `None`.
      rate_index_curves: An instance of `RateIndexCurve` or a list of those.
        If supplied as a list and `reference_mask` is not supplid,
        the size of the list should be the same as the number of priced
        instruments. Defines the index curves for each instrument. If not
        supplied, `coupon_spec.floating_rate_type` is used to identify the
        curves.
        Default value: `None`.
      reference_mask: An optional integer `Tensor` of values ranging from
        `0` to `len(rate_index_curves)` and of shape `batch_shape`. Identifies
        a mapping between `rate_index_curves` list and the underlying
        instruments.
        Default value: `None`.
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
      schedule_fn: A callable that accepts `start_date`, `end_date`,
        `coupon_frequency`, `settlement_days`, `first_coupon_date`, and
        `penultimate_coupon_date` as `Tensor`s and returns coupon payment
        days.
        Default value: `None`.
      schedule: A `DateTensor` of coupon payment dates.
        Default value: `None`.
      dtype: `tf.Dtype` of the input and output real `Tensor`s.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'floating_cashflow_stream'.
    """

    self._name = name or "floating_cashflow_stream"
    with tf.name_scope(self._name):
      curve_list = to_list(discount_curve_type)
      [
          self._discount_curve_type,
          self._mask
      ] = process_curve_types(curve_list, discount_curve_mask)
      self._first_coupon_date = None
      self._penultimate_coupon_date = None
      if schedule is None:
        if (start_date is None) or (end_date is None):
          raise ValueError("If `schedule` is not supplied both "
                           "`start_date` and `end_date` should be supplied")

      if schedule is None:
        if isinstance(start_date, tf.Tensor):
          self._start_date = dateslib.dates_from_tensor(
              start_date)
        else:
          self._start_date = dateslib.convert_to_date_tensor(
              start_date)
        if isinstance(start_date, tf.Tensor):
          self._end_date = dateslib.dates_from_tensor(
              end_date)
        else:
          self._end_date = dateslib.convert_to_date_tensor(
              end_date)
        self._first_coupon_date = first_coupon_date
        self._penultimate_coupon_date = penultimate_coupon_date
        if self._first_coupon_date is not None:
          if isinstance(start_date, tf.Tensor):
            self._first_coupon_date = dateslib.dates_from_tensor(
                first_coupon_date)
          else:
            self._first_coupon_date = dateslib.convert_to_date_tensor(
                first_coupon_date)
        if self._penultimate_coupon_date is not None:
          if isinstance(start_date, tf.Tensor):
            self._penultimate_coupon_date = dateslib.dates_from_tensor(
                penultimate_coupon_date)
          else:
            self._penultimate_coupon_date = dateslib.convert_to_date_tensor(
                penultimate_coupon_date)
      # Ignored and weekends only is used
      calendar = dateslib.create_holiday_calendar(
          weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
      # Convert coupon and reset frequencies to PeriodTensor
      coupon_frequency = _get_attr(coupon_spec, "coupon_frequency")
      # Update coupon frequency
      if isinstance(coupon_frequency, period_pb2.Period):
        coupon_frequency = market_data_utils.get_period(
            _get_attr(coupon_spec, "coupon_frequency"))
      if isinstance(coupon_frequency, (list, tuple)):
        coupon_frequency = market_data_utils.period_from_list(
            *_get_attr(coupon_spec, "coupon_frequency"))
      if isinstance(coupon_frequency, dict):
        coupon_frequency = market_data_utils.period_from_dict(
            _get_attr(coupon_spec, "coupon_frequency"))
      # Update reset frequency
      reset_frequency = _get_attr(coupon_spec, "reset_frequency")
      if isinstance(reset_frequency, period_pb2.Period):
        reset_frequency = market_data_utils.get_period(
            _get_attr(coupon_spec, "reset_frequency"))
      if isinstance(reset_frequency, (list, tuple)):
        reset_frequency = market_data_utils.period_from_list(
            *_get_attr(coupon_spec, "reset_frequency"))
      if isinstance(reset_frequency, dict):
        reset_frequency = market_data_utils.period_from_dict(
            _get_attr(coupon_spec, "reset_frequency"))
      self._reset_frequency = reset_frequency
      businessday_rule = _get_attr(coupon_spec, "businessday_rule")
      roll_convention, eom = market_data_utils.get_business_day_convention(
          businessday_rule)
      notional = tf.convert_to_tensor(
          _get_attr(coupon_spec, "notional_amount"),
          dtype=dtype,
          name="notional")
      self._dtype = dtype or notional.dtype

      daycount_convention = _get_attr(coupon_spec, "daycount_convention")

      daycount_fn = market_data_utils.get_daycount_fn(
          _get_attr(coupon_spec, "daycount_convention"), self._dtype)
      self._daycount_convention = daycount_convention

      self._settlement_days = tf.convert_to_tensor(
          _get_attr(coupon_spec, "settlement_days"),
          dtype=tf.int32,
          name="settlement_days")
      spread = tf.convert_to_tensor(_get_attr(coupon_spec, "spread"),
                                    dtype=self._dtype,
                                    name="spread")
      if schedule is not None:
        coupon_dates = dateslib.convert_to_date_tensor(schedule)
      elif schedule_fn is None:
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
      else:
        if first_coupon_date is not None:
          first_coupon_date = self._first_coupon_date.to_tensor()
        if penultimate_coupon_date is not None:
          penultimate_coupon_date = self._penultimate_coupon_date.to_tensor()
          coupon_dates = schedule_fn(
              start_date=self._start_date.to_tensor(),
              end_date=self._end_date.to_tensor(),
              coupon_frequency=coupon_frequency.quantity(),
              settlement_days=self._settlement_days,
              first_coupon_date=first_coupon_date,
              penultimate_coupon_date=penultimate_coupon_date)
      # Convert to DateTensor if the result comes from a tf.function
      coupon_dates = dateslib.convert_to_date_tensor(coupon_dates)
      # Extract batch shape
      self._batch_shape = tf.shape(coupon_dates.ordinal())[:-1]

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
          end_date=coupon_end_dates)

      self._num_cashflows = tf.shape(daycount_fractions)[-1]
      self._coupon_start_dates = coupon_start_dates
      self._coupon_end_dates = coupon_end_dates
      self._accrual_start_date = accrual_start_dates
      self._accrual_end_date = accrual_end_dates
      self._notional = notional
      self._daycount_fractions = daycount_fractions
      self._spread = spread
      self._currency = _get_attr(coupon_spec, "currency")
      self._daycount_fn = daycount_fn
      # Construct the reference curve object
      # Extract all rate_curves
      self._floating_rate_type = to_list(
          _get_attr(coupon_spec, "floating_rate_type"))
      self._currency = to_list(self._currency)
      if rate_index_curves is None:
        rate_index_curves = []
        for currency, floating_rate_type in zip(self._currency,
                                                self._floating_rate_type):
          rate_index_curves.append(curve_types_lib.RateIndexCurve(
              currency=currency, index=floating_rate_type))
      [
          self._reference_curve_type,
          self._reference_mask
      ] = process_curve_types(rate_index_curves, reference_mask)

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

  @property
  def coupon_start_dates(self) -> types.DateTensor:
    return self._coupon_start_dates

  @property
  def coupon_end_dates(self) -> types.DateTensor:
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
      reference_curve = get_discount_curve(
          self._reference_curve_type, market, self._reference_mask)
      valuation_date = dateslib.convert_to_date_tensor(market.date)

      past_fixing = _get_fixings(
          self._start_date,
          self._reference_curve_type,
          self._reset_frequency,
          self._reference_mask,
          market)
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
      discount_curve = get_discount_curve(
          self._discount_curve_type, market, self._mask)
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


def get_discount_curve(
    discount_curve_types: List[Union[curve_types_lib.RiskFreeCurve,
                                     curve_types_lib.RateIndexCurve]],
    market: pmd.ProcessedMarketData,
    mask: List[int]) -> rate_curve.RateCurve:
  """Builds a batched discount curve.

  Given a list of discount curve an integer mask, creates a discount curve
  object to compute discount factors against the list of discount curves.

  #### Example
  ```none
  curve_types = [RiskFreeCurve("USD"), RiskFreeCurve("AUD")]
  # A mask to price a batch of 7 instruments with the corresponding discount
  # curves ["USD", "AUD", "AUD", "AUD" "USD", "USD", "AUD"].
  mask = [0, 1, 1, 1, 0, 0, 1]
  market = MarketDataDict(...)
  get_discount_curve(curve_types, market, mask)
  # Returns a RateCurve object that can compute a discount factors for a
  # batch of 7 dates.
  ```

  Args:
    discount_curve_types: A list of curve types.
    market: an instance of the processed market data.
    mask: An integer mask.

  Returns:
    An instance of `RateCurve`.
  """
  discount_curves = [market.yield_curve(curve_type)
                     for curve_type in discount_curve_types]
  discounts = []
  dates = []
  interpolation_method = None
  interpolate_rates = None
  for curve in discount_curves:
    discount, date = curve.discount_factors_and_dates()
    discounts.append(discount)
    dates.append(date)
    interpolation_method = curve.interpolation_method
    interpolate_rates = curve.interpolate_rates

  all_discounts = tf.stack(pad.pad_tensors(discounts), axis=0)
  all_dates = pad.pad_date_tensors(dates)
  all_dates = dateslib.DateTensor.stack(dates, axis=0)
  prepare_discounts = tf.gather(all_discounts, mask)
  prepare_dates = dateslib.dates_from_ordinals(
      tf.gather(all_dates.ordinal(), mask))
  # All curves are assumed to have the same interpolation method
  # TODO(b/168411153): Extend to the case with multiple curve configs.
  discount_curve = rate_curve.RateCurve(
      prepare_dates, prepare_discounts, market.date,
      interpolator=interpolation_method,
      interpolate_rates=interpolate_rates)
  return discount_curve


def _get_fixings(start_dates,
                 reference_curve_types,
                 reset_frequencies,
                 reference_mask,
                 market):
  """Computes fixings for a list of reference curves."""
  split_indices = [tf.squeeze(tf.where(tf.equal(reference_mask, i)), -1)
                   for i in range(len(reference_curve_types))]
  fixings = []
  for idx, reference_curve_type in zip(split_indices, reference_curve_types):
    start_date = dateslib.dates_from_ordinals(
        tf.gather(start_dates.ordinal(), idx))
    reset_quant = reset_frequencies.quantity()
    # Do not use gather, if only one reset frequency is supplied
    if reset_quant.shape.rank > 1:
      reset_quant = tf.gather(reset_quant, idx)
    fixings.append(market.fixings(
        start_date,
        reference_curve_type,
        reset_quant))
  fixings = pad.pad_tensors(fixings)
  all_indices = tf.concat(split_indices, axis=0)
  all_fixings = tf.concat(fixings, axis=0)
  return tf.gather(all_fixings, tf.argsort(all_indices))


def process_curve_types(
    curve_types: List[Union[curve_types_lib.RiskFreeCurve,
                            curve_types_lib.RateIndexCurve]],
    mask=None
    ) -> Tuple[
        List[Union[curve_types_lib.RiskFreeCurve,
                   curve_types_lib.RateIndexCurve]],
        List[int]]:
  """Extracts unique curves and computes an integer mask.

  #### Example
  ```python
  curve_types = [RiskFreeCurve("USD"), RiskFreeCurve("AUD"),
                 RiskFreeCurve("USD")]
  process_curve_types(curve_types)
  # Returns [RiskFreeCurve("AUD"), RiskFreeCurve("USD")], [1, 0, 1]
  ```
  Args:
    curve_types: A list of either `RiskFreeCurve` or `RateIndexCurve`.
    mask: An optional integer mask for the sorted curve type sequence. If
      supplied, the function returns does not do anything and returns
      `(curve_types, mask)`.

  Returns:
    A Tuple of `(curve_list, mask)` where  `curve_list` is  a list of unique
    curves in `curve_types` and `mask` is a list of integers which is the
    mask for `curve_types`.
  """
  def _get_signature(curve):
    """Converts curve infromation to a string."""
    if isinstance(curve, curve_types_lib.RiskFreeCurve):
      return curve.currency.value
    elif isinstance(curve, curve_types_lib.RateIndexCurve):
      return (curve.currency.value + "_" + curve.index.type.value
              + "_" + "_".join(curve.index.source)
              + "_" + "_".join(curve.index.name))
    else:
      raise ValueError(f"{type(curve)} is not supported.")
  curve_list = to_list(curve_types)
  if mask is not None:
    return curve_list, mask
  curve_hash = [_get_signature(curve_type) for curve_type in curve_list]
  hash_discount_map = {
      _get_signature(curve_type): curve_type for curve_type in curve_list}
  mask, mask_map, num_unique_discounts = create_mask(curve_hash)
  discount_curve_types = [
      hash_discount_map[mask_map[i]]
      for i in range(num_unique_discounts)]
  return discount_curve_types, mask


def create_mask(x):
  """Given a list of object creates integer mask for unique values in the list.

  Args:
    x: 1-d numpy array.

  Returns:
    A tuple of three objects:
      * A list of integers that is the mask for `x`,
      * A dictionary map between  entries of `x` and the list
      * The number of unique elements.
  """
  # For example, create_mask(["USD", "AUD", "USD"]) returns
  # a list [1, 0, 1], a map {0: "AUD", 1: "USD"} and the number of unique
  # elements which is 2.
  # Note that elements of `x` are being sorted
  unique = np.unique(x)
  num_unique_elems = len(unique)
  keys = range(num_unique_elems)
  d = dict(zip(unique, keys))
  mask_map = dict(zip(keys, unique))
  return [d[el] for el in x], mask_map, num_unique_elems


def to_list(x):
  """Converts input to a list if necessary."""
  if isinstance(x, (list, tuple)):
    return x
  else:
    return [x]


def _get_attr(obj, key):
  if isinstance(obj, dict):
    return obj[key]
  else:
    return obj.__getattribute__(key)


__all__ = ["FixedCashflowStream", "FloatingCashflowStream"]
