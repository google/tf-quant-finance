# Lint as: python3
# Copyright 2020 Google LLC
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
"""Day Count Conventions.

Day count conventions are a system for determining how a coupon accumulates over
a coupon period. They can also be seen as a method for converting date
differences to elapsed time. For example, suppose we need to calculate the total
interest accrued over a period of 5 months starting from 6 Jan, 2020 to
8 June, 2020 given that the interest rate is quoted at 4% annually on a
principal of $100. Without the day count convention, we do not know how to
divide the total annual interest of $4 for the five month period. As an example
of the ambiguity, should the pro-rating be done by the total number of days
or by total number of months (or by some other metric)? The answer to this is
provided by assigning a specific day count convention to the quoted rate. For
example, one could use the Money market basis (Actual/360) which states that the
elapsed period for interest accrual between two dates D1 and D2 is the ratio
of the actual number of days between D1 and D2 and 360. For our example, it
leads to `154 / 360 = 0.4278`. Hence the accumulated interest is
`100 * 0.04 * 0.4278 = $1.71. For more details on the many conventions used, see
Ref. [1] and [2].

The functions in this module provide implementations of the commonly used day
count conventions. Some of the conventions also require a knowledge of the
payment schedule to be specified (e.g. Actual/Actual ISMA as in Ref [3] below.).

## References

[1] Wikipedia Contributors. Day Count Conventions. Available at:
  https://en.wikipedia.org/wiki/Day_count_convention
[2] ISDA, ISDA Definitions 2006.
  https://www.isda.org/book/2006-isda-definitions/
[3] ISDA, EMU and Market Conventions: Recent Developments,
  https://www.isda.org/a/AIJEE/1998-ISDA-memo-%E2%80%9CEMU-and-Market-Conventions-Recent-Developments%E2%80%9D.pdf
"""

# TODO(b/149382857): Move these implementations to use an interface.

import tensorflow.compat.v2 as tf
from tf_quant_finance.datetime import date_tensor as dt
from tf_quant_finance.datetime import date_utils as du
from tf_quant_finance.datetime import periods


def actual_360(*,
               start_date,
               end_date,
               schedule_info=None,
               dtype=None,
               name=None):
  """Computes the year fraction between the specified dates.

  The actual/360 convention specifies the year fraction between the start and
  end date as the actual number of days between the two dates divided by 360.

  Note that the schedule info is not needed for this convention and is ignored
  if supplied.

  For more details see:
  https://en.wikipedia.org/wiki/Day_count_convention#Actual/360

  Args:
    start_date: A `DateTensor` object of any shape.
    end_date: A `DateTensor` object of compatible shape with `start_date`.
    schedule_info: The schedule info. Ignored for this convention.
    dtype: The dtype of the result. Either `tf.float32` or `tf.float64`. If not
      supplied, `tf.float32` is returned.
    name: Python `str` name prefixed to ops created by this function. If not
      supplied, `actual_360` is used.

  Returns:
    A real `Tensor` of supplied `dtype` and shape of `start_date`. The year
    fraction between the start and end date as computed by Actual/360
    convention.
  """
  del schedule_info
  with tf.name_scope(name or 'actual_360'):
    end_date = dt.convert_to_date_tensor(end_date)
    start_date = dt.convert_to_date_tensor(start_date)
    dtype = dtype or tf.constant(0.).dtype
    actual_days = tf.cast(start_date.days_until(end_date), dtype=dtype)
    return actual_days / 360


def actual_365_fixed(*,
                     start_date,
                     end_date,
                     schedule_info=None,
                     dtype=None,
                     name=None):
  """Computes the year fraction between the specified dates.

  The actual/365 convention specifies the year fraction between the start and
  end date as the actual number of days between the two dates divided by 365.

  Note that the schedule info is not needed for this convention and is ignored
  if supplied.

  For more details see:
  https://en.wikipedia.org/wiki/Day_count_convention#Actual/365_Fixed

  Args:
    start_date: A `DateTensor` object of any shape.
    end_date: A `DateTensor` object of compatible shape with `start_date`.
    schedule_info: The schedule info. Ignored for this convention.
    dtype: The dtype of the result. Either `tf.float32` or `tf.float64`. If not
      supplied, `tf.float32` is returned.
    name: Python `str` name prefixed to ops created by this function. If not
      supplied, `actual_365_fixed` is used.

  Returns:
    A real `Tensor` of supplied `dtype` and shape of `start_date`. The year
    fraction between the start and end date as computed by Actual/365 fixed
    convention.
  """
  del schedule_info
  with tf.name_scope(name or 'actual_365_fixed'):
    end_date = dt.convert_to_date_tensor(end_date)
    start_date = dt.convert_to_date_tensor(start_date)
    dtype = dtype or tf.constant(0.).dtype
    actual_days = tf.cast(start_date.days_until(end_date), dtype=dtype)
    return actual_days / 365


def actual_365_actual(*,
                      start_date,
                      end_date,
                      schedule_info=None,
                      dtype=None,
                      name=None):
  """Computes the year fraction between the specified dates.

  The actual/365 actual convention specifies the year fraction between the
  start and end date as the actual number of days between the two dates divided
  365 if no leap day is contained in the date range and 366 otherwise.

  When determining whether a leap day is contained in the date range,
  `start_date` is excluded and `end_date` is included.

  Note that the schedule info is not needed for this convention and is ignored
  if supplied.

  Args:
    start_date: A `DateTensor` object of any shape.
    end_date: A `DateTensor` object of compatible shape with `start_date`.
    schedule_info: The schedule info. Ignored for this convention.
    dtype: The dtype of the result. Either `tf.float32` or `tf.float64`. If not
      supplied, `tf.float32` is returned.
    name: Python `str` name prefixed to ops created by this function. If not
      supplied, `actual_365_actual` is used.

  Returns:
    A real `Tensor` of supplied `dtype` and shape of `start_date`. The year
    fraction between the start and end date as computed by Actual/365 Actual
    convention.
  """
  del schedule_info
  with tf.name_scope(name or 'actual_365_actual'):
    end_date = dt.convert_to_date_tensor(end_date)
    start_date = dt.convert_to_date_tensor(start_date)
    dtype = dtype or tf.constant(0.).dtype
    actual_days = tf.cast(start_date.days_until(end_date), dtype=dtype)
    # Add a day to start_date and end_date so that start_date is excluded and
    # end_date is included.
    day = periods.day()
    leap_days_between = du.leap_days_between(
        start_date=start_date + day, end_date=end_date + day)
    denominator = tf.cast(tf.where(leap_days_between > 0, 366, 365),
                          dtype=dtype)
    return actual_days / denominator


def thirty_360_isda(*,
                    start_date,
                    end_date,
                    schedule_info=None,
                    dtype=None,
                    name=None):
  """Computes the year fraction between the specified dates.

  The 30/360 (ISDA / Bond Basis) convention specifies the year fraction
  between the start and end date as the number of days by the following
  formula between the two dates divided by 360.

    day difference = (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)

  where

    Y1 is the year, expressed as a number, of the start date;

    Y2 is the year, expressed as a number, of the end date;

    M1 is the calendar month, expressed as a number, of the start date;

    M2 is the calendar month, expressed as a number of the last date;

    D1 is the start date calendar day, unless such number would be 31, in
    which case D1 will be 30;

    D2 is the last date calendar day, unless such number would be 31 and D1
    is either 30 or 31, in which case D2 will be 30

  Note that the schedule info is not needed for this convention and is ignored
  if supplied.

  For more details see:
  https://en.wikipedia.org/wiki/Day_count_convention#30/360_Bond_Basis
  https://www.isda.org/2008/12/22/30-360-day-count-conventions

  Args:
    start_date: A `DateTensor` object of any shape.
    end_date: A `DateTensor` object of compatible shape with `start_date`.
    schedule_info: The schedule info. Ignored for this convention.
    dtype: The dtype of the result. Either `tf.float32` or `tf.float64`. If not
      supplied, `tf.float32` is returned.
    name: Python `str` name prefixed to ops created by this function. If not
      supplied, `thirty_360_isda` is used.

  Returns:
    A real `Tensor` of supplied `dtype` and shape of `start_date`. The year
    fraction between the start and end date as computed by 30/360 convention.
  """
  del schedule_info
  with tf.name_scope(name or 'thirty_360_isda'):
    d1_days = tf.minimum(start_date.day(), 30)
    d2_days = tf.where(
        tf.equal(d1_days, 30) & tf.equal(end_date.day(), 31),
        30,
        end_date.day()
    )

    day_difference = (d2_days - d1_days)
    month_difference = (end_date.month() - start_date.month()) * 30
    year_difference = (end_date.year() - start_date.year()) * 360
    dtype = dtype or tf.constant(0.).dtype
    total_day_difference = tf.cast(
        day_difference + month_difference + year_difference,
        dtype=dtype
    )

    return total_day_difference / 360


def actual_actual_isda(*,
                       start_date,
                       end_date,
                       schedule_info=None,
                       dtype=None,
                       name=None):
  """Computes the year fraction between the specified dates.

  Computes the year fraction between the dates by dividing the actual number of
  days in a leap year by 366 and the actual number of days in a standard year by
  365.

  When determining whether a leap day is contained in the date range,
  'start_date' is excluded and 'end_date' is included.

  Note that the schedule info is not needed for this convention and is ignored
  if supplied.

  https://en.wikipedia.org/wiki/Day_count_convention#Actual/Actual_ISDA

  Args:
    start_date: A `DateTensor` object of any shape.
    end_date: A `DateTensor` object of compatible shape with `start_date`.
    schedule_info: The schedule info. Ignored for this convention.
    dtype: The dtype of the result. Either `tf.float32` or `tf.float64`. If not
      supplied, `tf.float32` is returned.
    name: Python `str` name prefixed to ops created by this function. If not
      supplied, `actual_actual_isda` is used.

  Returns:
    A real `Tensor` of supplied `dtype` and shape of `start_date`. The year
    fraction between the start and end date as computed by Actual/Actual ISDA
    convention.
  """
  del schedule_info
  with tf.name_scope(name or 'actual_actual_isda'):
    end_date = dt.convert_to_date_tensor(end_date)
    start_date = dt.convert_to_date_tensor(start_date)
    dtype = dtype or tf.float32
    (
        days_in_leap_years,
        days_in_nonleap_years
    ) = du.days_in_leap_and_nonleap_years_between(start_date, end_date)
    # Cast to the target dtype
    days_in_leap_years = tf.cast(days_in_leap_years, dtype=dtype)
    days_in_nonleap_years = tf.cast(days_in_nonleap_years, dtype=dtype)
    return days_in_leap_years / 366 + days_in_nonleap_years / 365


__all__ = [
    'actual_actual_isda',
    'actual_360',
    'actual_365_actual',
    'actual_365_fixed',
    'thirty_360_isda',
]
