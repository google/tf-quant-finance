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
"""Floating rate note."""

import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import cashflow_stream as cs


class FloatingRateNote:
  """Represents a batch of floating rate notes.

  Floating rate notes are bond securities where the value of the coupon is not
  fixed at the time of issuance but is rather reset for every coupon period
  typically based on a benchmark index such as LIBOR rate [1].

  For example, consider a floating rate note with settlement date T_0 and
  maturity date T_n and equally spaced coupon payment dates T_1, T_2, ..., T_n
  such that

  T_0 < T_1 < T_2 < ... < T_n and dt_i = T_(i+1) - T_i    (A)

  The floating rate is fixed on T_0, T_1, ..., T_(n-1) and the payments are
  typically made on T_1, T_2, ..., T_n (payment dates) and the i-th coupon
  payment is given by:

  c_i = N * tau_i * L[T_{i-1}, T_i]                        (B)

  where N is the notional amount, tau_i is the daycount fraction for the period
  [T_{i-1}, T_i] and L[T_{i-1}, T_i] is the flotaing rate reset at T_{i-1}.

  The FloatingRateNote class can be used to create and price multiple FRNs
  simultaneously. However all FRNs within a FloatingRateNote object must be
  priced using a common reference and discount curve.

  #### Example:
  The following example illustrates the construction of an IRS instrument and
  calculating its price.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  settlement_date = dates.convert_to_date_tensor([(2021, 1, 15)])
  maturity_date = dates.convert_to_date_tensor([(2022, 1, 15)])
  valuation_date = dates.convert_to_date_tensor([(2021, 1, 15)])
  period_3m = dates.periods.months(3)
  flt_spec = instruments.FloatCouponSpecs(
      coupon_frequency=period_3m,
      reference_rate_term=period_3m,
      reset_frequency=period_3m,
      currency='usd',
      notional=100.,
      businessday_rule=dates.BusinessDayConvention.NONE,
      coupon_basis=0.,
      coupon_multiplier=1.,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)

  frn = instruments.FloatingRateNote(settlement_date, maturity_date,
                                     [flt_spec],
                                     dtype=dtype)

  curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([0.0, 0.005, 0.007, 0.015], dtype=dtype),
      valuation_date=valuation_date,
      dtype=dtype)
  market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                          reference_curve=reference_curve)

  price = frn.price(valuation_date, market)
  # Expected result: 100.
  ```

  #### References:
  [1]: Tomas Bjork. Arbitrage theory in continuous time, Second edition.
      Chapter 20. 2004.
  """

  def __init__(self,
               settlement_date,
               maturity_date,
               coupon_spec,
               start_date=None,
               first_coupon_date=None,
               penultimate_coupon_date=None,
               holiday_calendar=None,
               dtype=None,
               name=None):
    """Initialize a batch of floating rate notes (FRNs).

    Args:
      settlement_date: A rank 1 `DateTensor` specifying the settlement date of
        the FRNs.
      maturity_date: A rank 1 `DateTensor` specifying the maturity dates of the
        FRNs. The shape of the input should be the same as that of
        `settlement_date`.
      coupon_spec: A list of `FloatCouponSpecs` specifying the coupon payments.
        The length of the list should be the same as the number of FRNs
        being created.
      start_date: An optional `DateTensor` specifying the dates when the
        interest starts to accrue for the coupons. The input can be used to
        specify a forward start date for the coupons. The shape of the input
        correspond to the numbercof instruments being created.
        Default value: None in which case the coupons start to accrue from the
        `settlement_date`.
      first_coupon_date: An optional rank 1 `DateTensor` specifying the dates
        when first coupon will be paid for FRNs with irregular first coupon.
      penultimate_coupon_date: An optional rank 1 `DateTensor` specifying the
        dates when the penultimate coupon (or last regular coupon) will be paid
        for FRNs with irregular last coupon.
      holiday_calendar: An instance of `dates.HolidayCalendar` to specify
        weekends and holidays.
        Default value: None in which case a holiday calendar would be created
        with Saturday and Sunday being the holidays.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the bond object or created by the bond object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'floating_rate_note'.
    """
    self._name = name or 'floating_rate_note'

    if holiday_calendar is None:
      holiday_calendar = dates.create_holiday_calendar(
          weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY)

    with tf.name_scope(self._name):
      self._dtype = dtype
      self._settlement_date = dates.convert_to_date_tensor(settlement_date)
      self._maturity_date = dates.convert_to_date_tensor(maturity_date)
      self._holiday_calendar = holiday_calendar
      self._setup(coupon_spec, start_date, first_coupon_date,
                  penultimate_coupon_date)

  def price(self, valuation_date, market, model=None, name=None):
    """Returns the price of the FRNs on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the FRNs.
      model: Reserved for future use.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real dtype containing the price of each FRN
      based on the input market data.
    """

    name = name or (self._name + '_price')
    with tf.name_scope(name):
      discount_curve = market.discount_curve
      coupon_cf = self._cashflows.price(valuation_date, market, model)
      principal_cf = (
          self._notional * discount_curve.get_discount_factor(
              self._maturity_date)
          )
      return coupon_cf + principal_cf

  def _setup(self, coupon_spec, start_date, first_coupon_date,
             penultimate_coupon_date):
    """Setup bond cashflows."""
    if start_date is None:
      start_date = self._settlement_date
    self._cashflows = cs.FloatingCashflowStream(
        start_date,
        self._maturity_date,
        coupon_spec,
        first_coupon_date=first_coupon_date,
        penultimate_coupon_date=penultimate_coupon_date,
        dtype=self._dtype)

    self._notional = tf.convert_to_tensor([x.notional for x in coupon_spec],
                                          dtype=self._dtype)
