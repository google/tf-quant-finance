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
"""Cashflow streams."""

import tensorflow.compat.v2 as tf
from tf_quant_finance.experimental import dates
from tf_quant_finance.experimental.instruments import rates_common as rc


class FixedCashflowStream:
  """Represents a batch of fixed stream of cashflows.

  ### Example:
  The following example illustrates the construction of an FixedCashflowStream
  and calculating its present value.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.experimental.dates
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  start_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  maturity_date = dates.convert_to_date_tensor([(2023, 2, 2)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  period_6m = dates.periods.PeriodTensor(6, dates.PeriodType.MONTH)
  fix_spec = instruments.FixedCouponSpecs(
              coupon_frequency=period_6m, currency='usd',
              notional=1.e6, coupon_rate=0.03134,
              daycount_convention=rc.DayCountConvention.ACTUAL_365,
              businessday_rule=dates.BusinessDayConvention.NONE)

  cf_stream = instruments.FixedCashflowStream([start_date], [maturity_date],
                                              [fix_spec], dtype=dtype)

  curve_dates = valuation_date + dates.periods.PeriodTensor(
        [1, 2, 3, 5, 7, 10, 30], dates.PeriodType.YEAR)
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
      coupon_spec: A list of `FixedCouponSpecs` specifying the details of the
        coupon payment for the cashflow stream. The length of the list should
        be the same as the number of streams being created. Each coupon within
        the list must have the same daycount_convention and businessday_rule.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the FixedCashflowStream object or created by the
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'fixed_cashflow_stream'.
    """

    self._name = name or 'fixed_cashflow_stream'

    with tf.name_scope(self._name):
      self._start_date = dates.convert_to_date_tensor(start_date)
      self._end_date = dates.convert_to_date_tensor(end_date)
      self._batch_size = self._start_date.shape[0]
      self._dtype = dtype
      self._setup(coupon_spec)

  def price(self, valuation_date, market, model=None, name=None):
    """Returns the present value of the stream on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the cashflow stream.
      model: Reserved for future use.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each stream
      based on the input market data.
    """

    del valuation_date, model
    name = name or (self._name + '_price')
    with tf.name_scope(name):
      discount_curve = market.discount_curve
      discount_factors = discount_curve.get_discount_factor(
          self._payment_dates)
      cashflow_pvs = self._notional * (
          self._daycount_fractions * self._coupon_rate * discount_factors)
      return tf.math.segment_sum(cashflow_pvs, self._contract_index)

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

  def _setup(self, coupon_spec):
    """Setup tensors for efficient computations."""

    cpn_frequency = dates.periods.PeriodTensor.stack(
        [x.coupon_frequency for x in coupon_spec], axis=0)
    cpn_dates = dates.PeriodicSchedule(
        start_date=self._start_date,
        end_date=self._end_date,
        tenor=cpn_frequency,
        roll_convention=coupon_spec[-1].businessday_rule).dates()
    payment_dates = cpn_dates[:, 1:]
    notional = tf.expand_dims(
        tf.convert_to_tensor([x.notional for x in coupon_spec],
                             dtype=self._dtype),
        axis=-1)
    notional = tf.repeat(notional, payment_dates.shape.as_list()[-1], axis=-1)
    daycount_fractions = rc.get_daycount_fraction(
        cpn_dates[:, :-1],
        cpn_dates[:, 1:],
        coupon_spec[-1].daycount_convention,
        dtype=self._dtype)
    fixed_rate = tf.convert_to_tensor([x.coupon_rate for x in coupon_spec],
                                      dtype=self._dtype)
    coupon_rate = tf.expand_dims(fixed_rate, axis=-1)
    coupon_rate = tf.repeat(coupon_rate, payment_dates.shape.as_list()[-1],
                            axis=-1)
    contract_index = tf.repeat(tf.range(0, len(coupon_spec)),
                               notional.shape.as_list()[-1])

    self._payment_dates = dates.DateTensor.reshape(payment_dates, [-1])
    self._notional = tf.reshape(notional, [-1])
    self._daycount_fractions = tf.reshape(daycount_fractions, [-1])
    self._coupon_rate = tf.reshape(coupon_rate, [-1])
    self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype)
    self._contract_index = contract_index


class FloatingCashflowStream:
  """Represents a batch of cashflows indexed to a floating rate.

  ### Example:
  The following example illustrates the construction of an floating
  cashflow stream and calculating its present value.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.experimental.dates
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  start_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  maturity_date = dates.convert_to_date_tensor([(2023, 2, 2)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
  period_3m = dates.periods.PeriodTensor(3, dates.PeriodType.MONTH)
  flt_spec = instruments.FloatCouponSpecs(
              coupon_frequency=periods_3m, reference_rate_term=periods_3m,
              reset_frequency=periods_3m, currency='usd', notional=1.,
              businessday_rule=dates.BusinessDayConvention.NONE,
              coupon_basis=0., coupon_multiplier=1.,
              daycount_convention=rc.DayCountConvention.ACTUAL_365)

  cf_stream = instruments.FloatingCashflowStream([start_date], [maturity_date],
                                                 [flt_spec], dtype=dtype)

  curve_dates = valuation_date + dates.periods.PeriodTensor(
        [1, 2, 3, 5, 7, 10, 30], dates.PeriodType.YEAR)
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
      coupon_spec: A list of `FloatCouponSpecs` specifying the details of the
        coupon payment for the cashflow stream. The length of the list should
        be the same as the number of streams being created. Each coupon within
        the list must have the same daycount_convention and businessday_rule.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the FloatingCashflowStream object or created by the
        object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'floating_cashflow_stream'.
    """

    self._name = name or 'floating_cashflow_stream'

    with tf.name_scope(self._name):
      self._start_date = dates.convert_to_date_tensor(start_date)
      self._end_date = dates.convert_to_date_tensor(end_date)
      self._batch_size = self._start_date.shape[0]
      self._dtype = dtype

      self._setup(coupon_spec)

  def price(self, valuation_date, market, model=None, name=None):
    """Returns the present value of the stream on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the cashflow stream.
      model: Reserved for future use.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real type containing the modeled price of each stream
      contract based on the input market data.
    """

    del valuation_date, model
    name = name or (self._name + '_price')
    with tf.name_scope(name):
      discount_curve = market.discount_curve
      reference_curve = market.reference_curve

      discount_factors = discount_curve.get_discount_factor(self._payment_dates)
      forward_rates = reference_curve.get_forward_rate(self._accrual_start_date,
                                                       self._accrual_end_date,
                                                       self._daycount_fractions)

      forward_rates = tf.where(self._daycount_fractions > 0., forward_rates,
                               tf.zeros_like(forward_rates))

      coupon_rate = self._coupon_multiplier * (
          forward_rates + self._coupon_basis)

      cashflow_pvs = self._notional * (
          self._daycount_fractions * coupon_rate * discount_factors)
      return tf.math.segment_sum(cashflow_pvs, self._contract_index)

  def _setup(self, coupon_spec):
    """Setup tensors for efficient computations."""

    cpn_frequency = dates.periods.PeriodTensor.stack(
        [x.coupon_frequency for x in coupon_spec], axis=0)
    cpn_dates = dates.PeriodicSchedule(
        start_date=self._start_date,
        end_date=self._end_date,
        tenor=cpn_frequency,
        roll_convention=coupon_spec[-1].businessday_rule).dates()
    accrual_start_dates = cpn_dates[:, :-1]
    ref_term = dates.periods.PeriodTensor.stack(
        [x.reference_rate_term for x in coupon_spec], axis=0)

    accrual_end_dates = cpn_dates[:, :
                                  -1] + dates.periods.PeriodTensor.expand_dims(
                                      ref_term, axis=-1).broadcast_to(
                                          accrual_start_dates.shape)
    payment_dates = cpn_dates[:, 1:]

    daycount_fractions = rc.get_daycount_fraction(
        cpn_dates[:, :-1],
        cpn_dates[:, 1:],
        coupon_spec[-1].daycount_convention,
        dtype=self._dtype)

    notional = tf.repeat(
        tf.convert_to_tensor([x.notional for x in coupon_spec],
                             dtype=self._dtype),
        payment_dates.shape.as_list()[-1])

    coupon_basis = tf.repeat(tf.convert_to_tensor(
        [x.coupon_basis for x in coupon_spec], dtype=self._dtype),
                             payment_dates.shape.as_list()[-1])

    coupon_multiplier = tf.repeat(tf.convert_to_tensor(
        [x.coupon_multiplier for x in coupon_spec], dtype=self._dtype),
                                  payment_dates.shape.as_list()[-1])

    contract_index = tf.repeat(
        tf.range(0, len(coupon_spec)),
        payment_dates.shape.as_list()[-1])

    self._payment_dates = dates.DateTensor.reshape(payment_dates, [-1])
    self._accrual_start_date = dates.DateTensor.reshape(accrual_start_dates,
                                                        [-1])
    self._accrual_end_date = dates.DateTensor.reshape(accrual_end_dates, [-1])
    self._notional = notional
    self._daycount_fractions = tf.reshape(daycount_fractions, [-1])
    self._coupon_basis = coupon_basis
    self._coupon_multiplier = coupon_multiplier
    self._contract_index = contract_index
