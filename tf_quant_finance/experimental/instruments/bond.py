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

"""Fixed rate bond."""

import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import cashflow_stream as cs


class Bond:
  """Represents a batch of fixed coupon bonds.

  Bonds are fixed income securities where the issuer makes periodic payments
  (or coupons) on a principal amount (also known as the face value) based on a
  fixed annualized interest rate. The payments are made periodically (for
  example quarterly or semi-annually) where the last payment is typically made
  at the maturity (or termination) of the contract at which time the principal
  is also paid back.

  For example, consider a fixed rate bond with settlement date T_0 and maturity
  date T_n and equally spaced coupon payment dates T_1, T_2, ..., T_n such that

  T_0 < T_1 < T_2 < ... < T_n and dt_i = T_(i+1) - T_i    (A)

  The coupon accrual begins on T_0, T_1, ..., T_(n-1) and the payments are made
  on T_1, T_2, ..., T_n (payment dates). The principal is also paid at T_n.

  The Bond class can be used to create and price multiple bond securities
  simultaneously. However all bonds within a Bond object must be priced using
  a common reference and discount curve.

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
  start_date = dates.convert_to_date_tensor([(2020, 2, 8)])
  maturity_date = dates.convert_to_date_tensor([(2022, 2, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
  period_3m = dates.periods.months(3)
  period_6m = dates.periods.months(6)
  fix_spec = instruments.FixedCouponSpecs(
              coupon_frequency=period_6m, currency='usd',
              notional=1., coupon_rate=0.03134,
              daycount_convention=rc.DayCountConvention.ACTUAL_365,
              businessday_rule=dates.BusinessDayConvention.NONE)

  flt_spec = instruments.FloatCouponSpecs(
              coupon_frequency=periods_3m, reference_rate_term=periods_3m,
              reset_frequency=periods_3m, currency='usd', notional=1.,
              businessday_rule=dates.BusinessDayConvention.NONE,
              coupon_basis=0., coupon_multiplier=1.,
              daycount_convention=rc.DayCountConvention.ACTUAL_365)

  swap = instruments.InterestRateSwap([(2020,2,2)], [(2023,2,2)], [fix_spec],
                                      [flt_spec], dtype=np.float64)

  curve_dates = valuation_date + dates.periods.years(
        [1, 2, 3, 5, 7, 10, 30])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
        0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
        0.03213901, 0.03257991
        ], dtype=dtype),
      dtype=dtype)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = swap.price(valuation_date, market)
  # Expected result: 1e-7
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
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
    """Initialize a batch of fixed coupon bonds.

    Args:
      settlement_date: A rank 1 `DateTensor` specifying the settlement date of
        the bonds.
      maturity_date: A rank 1 `DateTensor` specifying the maturity dates of the
        bonds. The shape of the input should be the same as that of
        `settlement_date`.
      coupon_spec: A list of `FixedCouponSpecs` specifying the coupon payments.
        The length of the list should be the same as the number of bonds
        being created.
      start_date: An optional `DateTensor` specifying the dates when the
        interest starts to accrue for the coupons. The input can be used to
        specify a forward start date for the coupons. The shape of the input
        correspond to the numbercof instruments being created.
        Default value: None in which case the coupons start to accrue from the
        `settlement_date`.
      first_coupon_date: An optional rank 1 `DateTensor` specifying the dates
        when first coupon will be paid for bonds with irregular first coupon.
      penultimate_coupon_date: An optional rank 1 `DateTensor` specifying the
        dates when the penultimate coupon (or last regular coupon) will be paid
        for bonds with irregular last coupon.
      holiday_calendar: An instance of `dates.HolidayCalendar` to specify
        weekends and holidays.
        Default value: None in which case a holiday calendar would be created
        with Saturday and Sunday being the holidays.
      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops
        either supplied to the bond object or created by the bond object.
        Default value: None which maps to the default dtype inferred by
        TensorFlow.
      name: Python str. The name to give to the ops created by this class.
        Default value: `None` which maps to 'bond'.
    """
    self._name = name or 'bond'

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
    """Returns the dirty price of the bonds on the valuation date.

    Args:
      valuation_date: A scalar `DateTensor` specifying the date on which
        valuation is being desired.
      market: A namedtuple of type `InterestRateMarket` which contains the
        necessary information for pricing the bonds.
      model: Reserved for future use.
      name: Python str. The name to give to the ops created by this function.
        Default value: `None` which maps to 'price'.

    Returns:
      A Rank 1 `Tensor` of real dtype containing the dirty price of each bond
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
      self._cashflows = cs.FixedCashflowStream(self._settlement_date,
                                               self._maturity_date,
                                               coupon_spec,
                                               first_coupon_date,
                                               penultimate_coupon_date,
                                               dtype=self._dtype)
    else:
      self._cashflows = cs.FixedCashflowStream(start_date,
                                               self._maturity_date,
                                               coupon_spec,
                                               first_coupon_date,
                                               penultimate_coupon_date,
                                               dtype=self._dtype)

    self._notional = tf.convert_to_tensor([x.notional for x in coupon_spec],
                                          dtype=self._dtype)
