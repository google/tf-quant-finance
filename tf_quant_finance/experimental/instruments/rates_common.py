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


# TODO(b/149644030): Use daycounts.py for this.
class AverageType(enum.Enum):
  """Averaging types."""
  # Componded rate
  COMPOUNDING = 1

  # Arthmatic average
  ARITHMATIC_AVERAGE = 2


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


def get_daycount_fraction(date_start, date_end, convention, dtype):
  """Return the day count fraction between two dates using the input convention."""
  if convention == DayCountConvention.ACTUAL_365:
    return dates.daycounts.actual_365_fixed(
        start_date=date_start, end_date=date_end, dtype=dtype)
  elif convention == DayCountConvention.ACTUAL_360:
    return dates.daycounts.actual_360(
        start_date=date_start, end_date=date_end, dtype=dtype)
  else:
    raise ValueError('Daycount convention not implemented.')
