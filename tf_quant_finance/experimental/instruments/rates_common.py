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
from tf_quant_finance.experimental import dates


InterestRateMarket = collections.namedtuple(
    'InterestRateMarket',
    [
        # Instance of class RateCurve. The curve used for computing the forward
        # expectation of Libor rate.
        'reference_curve',
        # Instance of class RateCurve. The curve used for discounting cashflows.
        'discount_curve'
    ])

# TODO(b/151954834): Change to `attrs` or `dataclasses`
FixedCouponSpecs = collections.namedtuple(
    'FixedCouponSpecs',
    [
        # Scalar of type `dates.PeriodTensor` specifying the frequency of
        # the cashflow payments
        'coupon_frequency',
        # String specifying the currency of cashflows
        'currency',
        # Scalar of real dtype specifying the notional for the payments
        'notional',
        # Scalar of real dtype specifying the coupon rate
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
        # Scalar of type `dates.PeriodTensor` specifying the frequency of
        # the cashflow payments
        'coupon_frequency',
        # Scalar of type `dates.PeriodTensor` specifying the term of the
        # underlying rate which determines the coupon payment
        'reference_rate_term',
        # Scalar of type `dates.PeriodTensor` specifying the frequency with
        # which the underlying rate resets
        'reset_frequency',
        # String specifying the currency of cashflows
        'currency',
        # Scalar of real dtype specifying the notional for the payments
        'notional',
        # Scalar of type `DayCountConvention` specifying the daycount
        # convention of the underlying rate
        'daycount_convention',
        # Scalar of type `BusinessDayConvention` specifying how dates are rolled
        # if they fall on holidays
        'businessday_rule',
        # Scalar of real dtype
        'coupon_basis',
        # Scalar of real dtype
        'coupon_multiplier'
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


def elapsed_time(date_1, date_2, dtype):
  """Computes elapsed time between two date tensors."""
  days_in_year = 365.
  return tf.cast(date_1.days_until(date_2), dtype=dtype) / (
      days_in_year)


# TODO(b/149644030): Use daycounts.py for this.
def get_daycount_fraction(date_start, date_end, convention, dtype):
  """Return the day count fraction between two dates."""
  if convention == DayCountConvention.ACTUAL_365:
    return dates.daycounts.actual_365_fixed(
        start_date=date_start, end_date=date_end, dtype=dtype)
  elif convention == DayCountConvention.ACTUAL_360:
    return dates.daycounts.actual_360(
        start_date=date_start, end_date=date_end, dtype=dtype)
  else:
    raise ValueError('Daycount convention not implemented.')
