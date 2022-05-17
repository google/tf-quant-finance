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
"""Cashflow streams."""

import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import rates_common as rc


class CashflowStream:
  """Base class for Fixed or Floating cashflow streams."""

  def _generate_schedule(self, cpn_frequency, roll_convention):
    """Method to generate coupon dates.

    Args:
      cpn_frequency: A `PeriodTensor` specifying the frequency of coupon
        payments.
      roll_convention: Scalar of type `BusinessDayConvention` specifying how
        dates are rolled if they fall on holidays.

    Returns:
      A tuple containing the generated date schedule and a boolean `Tensor`
      of the same shape as the schedule specifying whether the coupons are
      regular coupons.
    """
    if (self._first_coupon_date is None) and (self._penultimate_coupon_date is
                                              None):
      cpn_dates = dates.PeriodicSchedule(
          start_date=self._start_date,
          end_date=self._end_date,
          tenor=cpn_frequency,
          roll_convention=roll_convention).dates()
      is_regular_cpn = tf.constant(
          True, dtype=bool, shape=cpn_dates[:, :-1].shape)
    elif self._first_coupon_date is not None:
      cpn_dates = dates.PeriodicSchedule(
          start_date=self._first_coupon_date,
          end_date=self._end_date,
          tenor=cpn_frequency,
          roll_convention=roll_convention).dates()
      cpn_dates = dates.DateTensor.concat(
          [self._start_date.expand_dims(-1), cpn_dates], axis=1)

      is_irregular_cpn = tf.constant(
          False, dtype=bool, shape=self._start_date.shape)
      is_regular_cpn = tf.concat([
          tf.expand_dims(is_irregular_cpn, axis=-1),
          tf.constant(True, dtype=bool, shape=cpn_dates[:, :-2].shape)
      ],
                                 axis=1)
    else:
      cpn_dates = dates.PeriodicSchedule(
          start_date=self._start_date,
          end_date=self._penultimate_coupon_date,
          backward=True,
          tenor=cpn_frequency,
          roll_convention=roll_convention).dates()
      cpn_dates = dates.DateTensor.concat(
          [cpn_dates, self._end_date.expand_dims(-1)], axis=1)

      is_irregular_cpn = tf.constant(
          False, dtype=bool, shape=self._end_date.shape)
      is_regular_cpn = tf.concat([
          tf.constant(True, dtype=bool, shape=cpn_dates[:, :-2].shape),
          tf.expand_dims(is_irregular_cpn, axis=-1)
      ],
                                 axis=1)

    return cpn_dates, is_regular_cpn


class FixedCashflowStream(CashflowStream):
  """Represents a batch of fixed stream of cashflows.

  #### Example:
  The following example illustrates the construction of an FixedCashflowStream
  and calculating its present value.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  start_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  maturity_date = dates.convert_to_date_tensor([(2023, 2, 2)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  period_6m = dates.periods.months(6)
  fix_spec = instruments.FixedCouponSpecs(
              coupon_frequency=period_6m, currency='usd',
              notional=1.e6, coupon_rate=0.03134,
              daycount_convention=rc.DayCountConvention.ACTUAL_365,
              businessday_rule=dates.BusinessDayConvention.NONE)

  cf_stream = instruments.FixedCashflowStream([start_date], [maturity_date],
                                              [fix_spec], dtype=dtype)

  curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
        0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
        0.03213901, 0.03257991
        ], dtype=dtype),
      dtype=dtype)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = cf_stream.price(valuation_date, market)
  # Expected result: 89259.267853547
  ```
  """

  def __init__(self,
               start_date,
               end_date,
               coupon_spec,
               first_coupon_date=None,
               penultimate_coupon_date=None,
               dtype=None,
               name=None):
    """Initialize a batch of fixed cashflow streams.

    Args:
      start_date: A rank 1 `DateTensor` specifying the starting dates of the
        accrual of the first coupon of the cashflow stream. The shape of the
        input correspond to the numbercof streams being created.
      end_date: A rank 1 `DateTensor` specifying the end dates for accrual of
        the last coupon in each cashflow stream. The shape of the input should
        be the same as that of `start_date`.
      coupon_spec: A scalar or a list of `FixedCouponSpecs` specifying the
        details of the coupon payment for the cashflow stream. If specified as
        a list then the length of the list should be the same as the number of
        streams being created and each coupon within the list must have the
        same daycount_convention and businessday_rule. If specified as
        a scalar, then the elements of the namedtuple must be of the same shape
        as (or compatible to) the shape of `start_date`.
      first_coupon_date: An optional rank 1 `DateTensor` specifying the payment
        dates of the first coupon of the cashflow stream. Use this input for
        cashflows with irregular first coupon.
        Default value: None which implies regular first coupon.
      penultimate_coupon_date: An optional rank 1 `DateTensor` specifying the
        payment date of the penultimate (next to last) coupon of the cashflow
        stream. Use this input for cashflows with irregular last coupon.
        Default value: None which implies regular last coupon.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the FixedCashflowStream object or created by the
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'fixed_cashflow_stream'.
    """

    super(FixedCashflowStream, self).__init__()
    self._name = name or 'fixed_cashflow_stream'

    with tf.name_scope(self._name):
      self._start_date = dates.convert_to_date_tensor(start_date)
      self._end_date = dates.convert_to_date_tensor(end_date)
      self._batch_size = self._start_date.shape[0]
      self._dtype = dtype
      if first_coupon_date is None:
        self._first_coupon_date = None
      else:
        self._first_coupon_date = dates.convert_to_date_tensor(
            first_coupon_date)

      if penultimate_coupon_date is None:
        self._penultimate_coupon_date = None
      else:
        self._penultimate_coupon_date = dates.convert_to_date_tensor(
            penultimate_coupon_date)

      self._setup(coupon_spec)

  def price(self, valuation_date, market, model=None, pricing_context=None,
            name=None):
    """Returns the present value of the stream on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the cashflow stream.
      model: Reserved for future use.
      pricing_context: Additional context relevant for pricing.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each stream
      based on the input market data.
    """

    del model, pricing_context
    name = name or (self._name + '_price')
    with tf.name_scope(name):
      discount_curve = market.discount_curve
      discount_factors = discount_curve.get_discount_factor(
          self._payment_dates)
      future_cashflows = tf.cast(self._payment_dates >= valuation_date,
                                 dtype=self._dtype)
      cashflow_pvs = self._notional * (
          future_cashflows * self._daycount_fractions * self._coupon_rate *
          discount_factors)
      return tf.math.reduce_sum(
          tf.reshape(cashflow_pvs, (self._batch_size, self._num_cashflows)),
          axis=1)

  @property
  def payment_dates(self):
    return self._payment_dates

  @property
  def contract_index(self):
    return self._contract_index

  @property
  def daycount_fractions(self):
    return self._daycount_fractions

  @property
  def fixed_rate(self):
    return self._fixed_rate

  @property
  def notional(self):
    return self._notional

  def _setup(self, coupon_spec):
    """Setup tensors for efficient computations."""

    if isinstance(coupon_spec, list):
      cpn_frequency = dates.PeriodTensor.stack(
          [x.coupon_frequency for x in coupon_spec], axis=0)
      businessday_rule = coupon_spec[-1].businessday_rule
      notional = tf.convert_to_tensor([x.notional for x in coupon_spec],
                                      dtype=self._dtype)
      fixed_rate = tf.convert_to_tensor([x.coupon_rate for x in coupon_spec],
                                        dtype=self._dtype)
      daycount_convention = coupon_spec[-1].daycount_convention
    else:
      cpn_frequency = coupon_spec.coupon_frequency
      businessday_rule = coupon_spec.businessday_rule
      notional = tf.broadcast_to(
          tf.convert_to_tensor(coupon_spec.notional, dtype=self._dtype),
          self._start_date.shape)
      fixed_rate = tf.broadcast_to(
          tf.convert_to_tensor(coupon_spec.coupon_rate, dtype=self._dtype),
          self._start_date.shape)
      daycount_convention = coupon_spec.daycount_convention

    cpn_dates, _ = self._generate_schedule(cpn_frequency, businessday_rule)
    payment_dates = cpn_dates[:, 1:]

    notional = tf.repeat(notional, payment_dates.shape.as_list()[-1])
    daycount_fractions = rc.get_daycount_fraction(
        cpn_dates[:, :-1],
        cpn_dates[:, 1:],
        daycount_convention,
        dtype=self._dtype)

    coupon_rate = tf.expand_dims(fixed_rate, axis=-1)
    coupon_rate = tf.repeat(coupon_rate, payment_dates.shape.as_list()[-1])
    contract_index = tf.repeat(tf.range(0, self._batch_size),
                               payment_dates.shape.as_list()[-1])

    self._num_cashflows = payment_dates.shape.as_list()[-1]
    self._payment_dates = payment_dates.reshape([-1])
    self._notional = notional
    self._daycount_fractions = tf.reshape(daycount_fractions, [-1])
    self._coupon_rate = coupon_rate
    self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype)
    self._contract_index = contract_index


class FloatingCashflowStream(CashflowStream):
  """Represents a batch of cashflows indexed to a floating rate.

  #### Example:
  The following example illustrates the construction of an floating
  cashflow stream and calculating its present value.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  start_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  maturity_date = dates.convert_to_date_tensor([(2023, 2, 2)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  period_3m = dates.months(3)
  flt_spec = instruments.FloatCouponSpecs(
              coupon_frequency=periods_3m, reference_rate_term=periods_3m,
              reset_frequency=periods_3m, currency='usd', notional=1.,
              businessday_rule=dates.BusinessDayConvention.NONE,
              coupon_basis=0., coupon_multiplier=1.,
              daycount_convention=rc.DayCountConvention.ACTUAL_365)

  cf_stream = instruments.FloatingCashflowStream([start_date], [maturity_date],
                                                 [flt_spec], dtype=dtype)

  curve_dates = valuation_date + dates.years([1, 2, 3, 5, 7, 10, 30])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
        0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
        0.03213901, 0.03257991
        ], dtype=dtype),
      dtype=dtype)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = cf_stream.price(valuation_date, market)
  # Expected result: 89259.685614769
  ```
  """

  def __init__(self,
               start_date,
               end_date,
               coupon_spec,
               first_coupon_date=None,
               penultimate_coupon_date=None,
               dtype=None,
               name=None):
    """Initialize a batch of floating cashflow streams.

    Args:
      start_date: A rank 1 `DateTensor` specifying the starting dates of the
        accrual of the first coupon of the cashflow stream. The shape of the
        input correspond to the numbercof streams being created.
      end_date: A rank 1 `DateTensor` specifying the end dates for accrual of
        the last coupon in each cashflow stream. The shape of the input should
        be the same as that of `start_date`.
      coupon_spec: A scalar or a list of `FloatCouponSpecs` specifying the
        details of the coupon payment for the cashflow stream. If specified as
        a list then the length of the list should be the same as the number of
        streams being created and each coupon within the list must have the
        same daycount_convention and businessday_rule. If specified as
        a scalar, then the elements of the namedtuple must be of the same shape
        as (or compatible to) the shape of `start_date`.
      first_coupon_date: An optional rank 1 `DateTensor` specifying the payment
        dates of the first coupon of the cashflow stream. Use this input for
        cashflows with irregular first coupon.
        Default value: None which implies regular first coupon.
      penultimate_coupon_date: An optional rank 1 `DateTensor` specifying the
        payment date of the penultimate (next to last) coupon of the cashflow
        stream. Use this input for cashflows with irregular last coupon.
        Default value: None which implies regular last coupon.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the FloatingCashflowStream object or created by the
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'floating_cashflow_stream'.
    """

    super(FloatingCashflowStream, self).__init__()
    self._name = name or 'floating_cashflow_stream'

    with tf.name_scope(self._name):
      self._start_date = dates.convert_to_date_tensor(start_date)
      self._end_date = dates.convert_to_date_tensor(end_date)
      self._batch_size = self._start_date.shape[0]
      if first_coupon_date is None:
        self._first_coupon_date = None
      else:
        self._first_coupon_date = dates.convert_to_date_tensor(
            first_coupon_date)

      if penultimate_coupon_date is None:
        self._penultimate_coupon_date = None
      else:
        self._penultimate_coupon_date = dates.convert_to_date_tensor(
            penultimate_coupon_date)

      self._dtype = dtype

      self._setup(coupon_spec)

  def price(self, valuation_date, market, model=None, pricing_context=None,
            name=None):
    """Returns the present value of the stream on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the cashflow stream.
      model: Reserved for future use.
      pricing_context: Additional context relevant for pricing.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each stream
      contract based on the input market data.
    """

    del model, pricing_context
    name = name or (self._name + '_price')
    with tf.name_scope(name):
      discount_curve = market.discount_curve
      reference_curve = market.reference_curve
      libor_rate = rc.get_rate_index(market, self._start_date,
                                     rc.RateIndexType.LIBOR,
                                     dtype=self._dtype)
      libor_rate = tf.repeat(tf.convert_to_tensor(
          libor_rate, dtype=self._dtype), self._num_cashflows)

      discount_factors = discount_curve.get_discount_factor(self._payment_dates)
      forward_rates = reference_curve.get_forward_rate(self._accrual_start_date,
                                                       self._accrual_end_date,
                                                       self._daycount_fractions)

      forward_rates = tf.where(self._daycount_fractions > 0., forward_rates,
                               tf.zeros_like(forward_rates))
      # If coupon end date is before the valuation date, the payment is in the
      # past. If valuation date is between coupon start date and coupon end
      # date, then the rate has been fixed but not paid. Otherwise the rate is
      # not fixed and should be read from the curve.
      forward_rates = tf.where(
          self._coupon_end_dates < valuation_date,
          tf.constant(0., dtype=self._dtype),
          tf.where(self._coupon_start_dates < valuation_date,
                   libor_rate, forward_rates))

      coupon_rate = self._coupon_multiplier * (
          forward_rates + self._coupon_basis)

      cashflow_pvs = self._notional * (
          self._daycount_fractions * coupon_rate * discount_factors)
      return tf.math.reduce_sum(
          tf.reshape(cashflow_pvs, (self._batch_size, self._num_cashflows)),
          axis=1)

  @property
  def notional(self):
    return self._notional

  def _setup(self, coupon_spec):
    """Setup tensors for efficient computations."""

    if isinstance(coupon_spec, list):
      cpn_frequency = dates.PeriodTensor.stack(
          [x.coupon_frequency for x in coupon_spec], axis=0)
      businessday_rule = coupon_spec[-1].businessday_rule
      ref_term = dates.PeriodTensor.stack(
          [x.reference_rate_term for x in coupon_spec], axis=0)
      daycount_convention = coupon_spec[-1].daycount_convention
      notional = tf.convert_to_tensor([x.notional for x in coupon_spec],
                                      dtype=self._dtype)
      coupon_basis = tf.convert_to_tensor(
          [x.coupon_basis for x in coupon_spec], dtype=self._dtype)
      coupon_multiplier = tf.convert_to_tensor(
          [x.coupon_multiplier for x in coupon_spec], dtype=self._dtype)
    else:
      cpn_frequency = coupon_spec.coupon_frequency
      businessday_rule = coupon_spec.businessday_rule
      ref_term = coupon_spec.reference_rate_term
      daycount_convention = coupon_spec.daycount_convention
      notional = tf.broadcast_to(
          tf.convert_to_tensor(coupon_spec.notional, dtype=self._dtype),
          self._start_date.shape)
      coupon_basis = tf.broadcast_to(
          tf.convert_to_tensor(coupon_spec.coupon_basis, dtype=self._dtype),
          self._start_date.shape)
      coupon_multiplier = tf.broadcast_to(
          tf.convert_to_tensor(coupon_spec.coupon_multiplier,
                               dtype=self._dtype), self._start_date.shape)

    cpn_dates, is_regular_cpn = self._generate_schedule(cpn_frequency,
                                                        businessday_rule)
    accrual_start_dates = cpn_dates[:, :-1]

    accrual_end_dates = cpn_dates[:, :-1] + ref_term.expand_dims(
        axis=-1).broadcast_to(accrual_start_dates.shape)
    coupon_start_dates = cpn_dates[:, :-1]
    coupon_end_dates = cpn_dates[:, 1:]
    accrual_end_dates = dates.DateTensor.where(is_regular_cpn,
                                               accrual_end_dates,
                                               coupon_end_dates)
    payment_dates = cpn_dates[:, 1:]

    daycount_fractions = rc.get_daycount_fraction(
        cpn_dates[:, :-1],
        cpn_dates[:, 1:],
        daycount_convention,
        dtype=self._dtype)

    notional = tf.repeat(notional, payment_dates.shape.as_list()[-1])
    coupon_basis = tf.repeat(coupon_basis, payment_dates.shape.as_list()[-1])
    coupon_multiplier = tf.repeat(coupon_multiplier,
                                  payment_dates.shape.as_list()[-1])

    contract_index = tf.repeat(tf.range(0, self._batch_size),
                               payment_dates.shape.as_list()[-1])

    self._num_cashflows = daycount_fractions.shape.as_list()[-1]
    self._coupon_start_dates = coupon_start_dates.reshape([-1])
    self._coupon_end_dates = coupon_end_dates.reshape([-1])
    self._payment_dates = payment_dates.reshape([-1])
    self._accrual_start_date = accrual_start_dates.reshape([-1])
    self._accrual_end_date = accrual_end_dates.reshape([-1])
    self._notional = notional
    self._daycount_fractions = tf.reshape(daycount_fractions, [-1])
    self._coupon_basis = coupon_basis
    self._coupon_multiplier = coupon_multiplier
    self._contract_index = contract_index
    self._is_regular_coupon = tf.reshape(is_regular_cpn, [-1])
