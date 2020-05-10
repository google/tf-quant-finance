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

"""Common rates related utilities."""

import collections
import enum
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates


InterestRateMarket = collections.namedtuple(
    'InterestRateMarket',
    [
        # Instance of class RateCurve. The curve used for computing the forward
        # expectation of Libor rate.
        'reference_curve',
        # Instance of class RateCurve. The curve used for discounting cashflows.
        'discount_curve',
        # Scalar of real dtype containing the past fixing of libor rate
        'libor_rate',
        # Scalar of real dtype containing the past fixing of swap rate
        'swap_rate',
        # Instance of class VolatiltyCube. Market implied black volatilities.
        'volatility_curve'
    ])
InterestRateMarket.__new__.__defaults__ = (None, None, None, None, None)

# TODO(b/151954834): Change to `attrs` or `dataclasses`
FixedCouponSpecs = collections.namedtuple(
    'FixedCouponSpecs',
    [
        # Scalar or rank 1 `dates.PeriodTensor` specifying the frequency of
        # the cashflow payments
        'coupon_frequency',
        # String specifying the currency of cashflows
        'currency',
        # Scalar or rank 1 `Tensor` of real dtype specifying the notional for
        # the payments
        'notional',
        # Scalar or rank 1 `Tensor` of real dtype specifying the coupon rate
        'coupon_rate',
        # Scalar of type `DayCountConvention` specifying the applicable
        # daycount convention
        'daycount_convention',
        # Scalar of type `BusinessDayConvention` specifying how dates are rolled
        # if they fall on holidays
        'businessday_rule'
    ])

FloatCouponSpecs = collections.namedtuple(
    'FloatCouponSpecs',
    [
        # Scalar or rank 1 `dates.PeriodTensor` specifying the frequency of
        # the cashflow payments
        'coupon_frequency',
        # Scalar or rank 1 `dates.PeriodTensor` specifying the term of the
        # underlying rate which determines the coupon payment
        'reference_rate_term',
        # Scalar or rank 1 `dates.PeriodTensor` specifying the frequency with
        # which the underlying rate resets
        'reset_frequency',
        # String specifying the currency of cashflows
        'currency',
        # Scalar or rank 1 `Tensor` of real dtype specifying the notional for
        # the payments
        'notional',
        # Scalar of type `DayCountConvention` specifying the daycount
        # convention of the underlying rate
        'daycount_convention',
        # Scalar of type `BusinessDayConvention` specifying how dates are rolled
        # if they fall on holidays
        'businessday_rule',
        # Scalar of real dtype specifying the fixed basis (in decimals)
        'coupon_basis',
        # Scalar of real dtype
        'coupon_multiplier'
    ])

CMSCouponSpecs = collections.namedtuple(
    'CMSCouponSpecs',
    [
        # Scalar of type `dates.PeriodTensor` specifying the frequency of
        # the cashflow payments
        'coupon_frequency',
        # Scalar `dates.PeriodTensor` specifying the tenor of the CMS rate
        'tenor',
        # Scalar of type `instruments.FloatCouponSpecs` specifying the floating
        # leg of the CMS
        'float_leg',
        # Scalar of type `instruments.FixedCouponSpecs` specifying the fixed
        # leg of the CMS
        'fixed_leg',
        # Scalar of real dtype specifying the notional for the payments
        'notional',
        # Scalar of type `DayCountConvention` specifying the daycount
        # convention of the underlying rate
        'daycount_convention',
        # Scalar of real dtype specifying the fixed basis (in decimals)
        'coupon_basis',
        # Scalar of real dtype
        'coupon_multiplier',
        # Scalar of type `BusinessDayConvention` specifying how dates are rolled
        # if they fall on holidays
        'businessday_rule'
    ])


class AverageType(enum.Enum):
  """Averaging types."""
  # Componded rate
  COMPOUNDING = 1

  # Arthmatic average
  ARITHMETIC_AVERAGE = 2


class DayCountConvention(enum.Enum):
  """Day count conventions for accrual."""
  # Actual/360 day count basis
  ACTUAL_360 = 1

  # Acutal/365 day count basis
  ACTUAL_365 = 2

  # 30/360 ISDA day count basis
  THIRTY_360_ISDA = 3


class RateIndexType(enum.Enum):
  """Interest rate indexes."""
  # LIBOR rates
  LIBOR = 1

  # Swap rates
  SWAP = 2


class InterestRateModelType(enum.Enum):
  """Models for pricing interest rate derivatives."""
  # Lognormal model for the underlying rate
  LOGNORMAL_RATE = 1

  # Normal model for the underlying rate
  NORMAL_RATE = 2

  # Smile consistent replication (lognormal vols)
  LOGNORMAL_SMILE_CONSISTENT_REPLICATION = 3

  # Smile consistent replication (normal vols)
  NORMAL_SMILE_CONSISTENT_REPLICATION = 4


def elapsed_time(date_1, date_2, dtype):
  """Computes elapsed time between two date tensors."""
  days_in_year = 365.
  return tf.cast(date_1.days_until(date_2), dtype=dtype) / (
      days_in_year)


# TODO(b/149644030): Use daycounts.py for this.
def get_daycount_fraction(date_start, date_end, convention, dtype):
  """Return the day count fraction between two dates."""
  if convention == DayCountConvention.ACTUAL_365:
    return dates.daycount_actual_365_fixed(
        start_date=date_start, end_date=date_end, dtype=dtype)
  elif convention == DayCountConvention.ACTUAL_360:
    return dates.daycount_actual_360(
        start_date=date_start, end_date=date_end, dtype=dtype)
  elif convention == DayCountConvention.THIRTY_360_ISDA:
    return dates.daycount_thirty_360_isda(
        start_date=date_start, end_date=date_end, dtype=dtype)
  else:
    raise ValueError('Daycount convention not implemented.')


def get_rate_index(market,
                   valuation_date,
                   rate_type=None,
                   currency=None,
                   dtype=None):
  """Return the relevant rate from the market data."""
  del currency
  if rate_type == RateIndexType.LIBOR:
    rate = market.libor_rate or tf.zeros(valuation_date.shape, dtype=dtype)
  elif rate_type == RateIndexType.SWAP:
    rate = market.swap_rate or tf.zeros(valuation_date.shape, dtype=dtype)
  else:
    raise ValueError('Unrecognized rate type.')
  return rate


def get_implied_volatility_data(market,
                                valuation_date=None,
                                volatility_type=None,
                                currency=None):
  """Return the implied colatility date from the market data."""
  del valuation_date, volatility_type, currency
  vol_date = market.volatility_curve
  return vol_date
