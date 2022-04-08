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
"""Utilities for working with dates."""

import tensorflow.compat.v2 as tf

_DAYS_IN_ERA = 146097  # Era is 400 years.
_YEARS_IN_ERA = 400
_DAYS_IN_YEAR = 365
_DAYS_IN_4_YEARS = 4 * _DAYS_IN_YEAR + 1
_DAYS_IN_100_YEARS = 100 * _DAYS_IN_YEAR + 24
_ORDINAL_OF_1_3_0000 = -305


def _day_of_year_to_month(day_of_year):
  # Converts day of year to month, when the year is counted from 1st of March,
  # and both days and months are zero-based (i.e. for 1st of March,
  # day_of_year = 0, and month = 0).
  # The numbers 5, 2 and 153 are "magic" numbers: this linear function just
  # happens to always give correct results, both for leap and non-leap years.
  return (5 * day_of_year + 2) // 153


def _days_in_year_before_month(month):
  # Calculates the number of days that there are between given month and 1st of
  # March. Again, 153, 2, 5 are "magic" numbers.
  return (153 * month + 2) // 5


def ordinal_to_year_month_day(ordinals):
  """Calculates years, months and dates Tensor given ordinals Tensor.

  Args:
    ordinals: Tensor of int32 type. Each element is number of days since 1 Jan
     0001. 1 Jan 0001 has `ordinal = 1`.

  Returns:
    Tuple (years, months, days), each element is an int32 Tensor of the same
    shape as `ordinals`. `months` and `days` are one-based.
  """
  with tf.compat.v1.name_scope(None, "o2ymd", [ordinals]):
    ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32, name="ordinals")

    # The algorithm is adapted from
    # http://howardhinnant.github.io/date_algorithms.html

    # Take the fictional date of 1 March 0000 as reference and consider 1 March
    # as start of the year. This simplifies computations.
    ordinals -= _ORDINAL_OF_1_3_0000
    era = ordinals // _DAYS_IN_ERA
    day_of_era = ordinals % _DAYS_IN_ERA
    year_of_era = (day_of_era - day_of_era // (_DAYS_IN_4_YEARS - 1)
                   + day_of_era // _DAYS_IN_100_YEARS
                   - day_of_era // (_DAYS_IN_ERA - 1)) // _DAYS_IN_YEAR
    year = year_of_era + era * _YEARS_IN_ERA
    day_of_year = day_of_era - (_DAYS_IN_YEAR * year_of_era + year_of_era // 4 -
                                year_of_era // 100)
    months = _day_of_year_to_month(day_of_year)
    days = day_of_year - _days_in_year_before_month(months) + 1

    # Go back from 1 March to 1 January as start of year.
    months = months + tf.compat.v2.where(months < 10, 3, -9)
    year += tf.compat.v2.where(months <= 2, 1, 0)
    return year, months, days


def year_month_day_to_ordinal(year, month, day):
  """Calculates ordinals Tensor given years, months and dates.

  Args:
    year: Tensor of int32 type. Elements should be positive.
    month: Tensor of int32 type of same shape as `year`. Elements should be in
      range `[1, 12]`.
    day: Tensor of int32 type of same shape as `year`. Elements should be in
      range `[1, 31]` and represent valid dates together with corresponding
      elements of `month` and `year` Tensors.

  Returns:
    Tensor of int32 type. Each element is number of days since 1 Jan 0001. 1 Jan
    0001 has `ordinal = 1`.
  """
  with tf.compat.v1.name_scope(None, "ymd2o", [year, month, day]):
    year = tf.convert_to_tensor(year, tf.int32, name="year")
    month = tf.convert_to_tensor(month, tf.int32, name="month")
    day = tf.convert_to_tensor(day, tf.int32, name="day")

    # The algorithm is adapted from
    # http://howardhinnant.github.io/date_algorithms.html

    # Take the fictional date of 1 March 0000 as reference and consider 1 March
    # as start of the year. This simplifies computations.
    year -= tf.compat.v2.where(month <= 2, 1, 0)
    month += tf.compat.v2.where(month > 2, -3, 9)

    era = year // _YEARS_IN_ERA
    year_of_era = year % _YEARS_IN_ERA
    day_of_year = _days_in_year_before_month(month) + day - 1
    day_of_era = (year_of_era * _DAYS_IN_YEAR + year_of_era // 4
                  - year_of_era // 100 + day_of_year)
    return era * _DAYS_IN_ERA + day_of_era + _ORDINAL_OF_1_3_0000


def is_leap_year(years):
  """Calculates whether years are leap years.

  Args:
    years: Tensor of int32 type. Elements should be positive.

  Returns:
    Tensor of bool type.
  """
  years = tf.convert_to_tensor(years, tf.int32)
  def divides_by(n):
    return tf.math.equal(years % n, 0)
  return tf.math.logical_and(
      divides_by(4), tf.math.logical_or(~divides_by(100), divides_by(400)))


def days_in_leap_years_between(start_date, end_date):
  """Calculates number of days between two dates that fall on leap years.

  'start_date' is included and 'end_date' is excluded from the period.

  For example, for dates `2019-12-24` and `2024-2-10` the result is
  406: 366 days in 2020, 31 in Jan 2024 and 9 in Feb 2024.

  If `end_date` is earlier than `start_date`, the result will be negative or
  zero.

  Args:
    start_date: DateTensor.
    end_date: DateTensor compatible with `start_date`.

  Returns:
    Tensor of type 'int32'.
  """
  def days_in_leap_years_since_1jan0001(date):
    prev_year = date.year() - 1
    leap_years_before = prev_year // 4 - prev_year // 100 + prev_year // 400
    n_leap_days = leap_years_before * 366

    days_in_cur_year = date.day_of_year() - 1  # exclude current day.
    n_leap_days += tf.where(is_leap_year(date.year()), days_in_cur_year, 0)
    return n_leap_days

  return (days_in_leap_years_since_1jan0001(end_date) -
          days_in_leap_years_since_1jan0001(start_date))


def days_in_leap_and_nonleap_years_between(start_date, end_date):
  """Calculates number of days that fall on leap and non-leap years.

  Calculates a tuple '(days_in_leap_years, days_in_nonleap_years)'.
  'start_date' is included and 'end_date' is excluded from the period.

  For example, for dates `2019-12-24` and `2024-2-10` the result is
  (406, 1103):
  406 = 366 days in 2020 + 31 in Jan 2024 + 9 in Feb 2024,
  1103 = 8 in 2019 + 365 in 2021 + 365 in 2022 + 365 in 2023.

  If `end_date` is earlier than `start_date`, the result will be negative or
  zero.

  Args:
    start_date: DateTensor.
    end_date: DateTensor compatible with `start_date`.

  Returns:
    Tuple of two Tensors of type 'int32'.
  """
  days_between = end_date.ordinal() - start_date.ordinal()
  days_in_leap_years = days_in_leap_years_between(start_date, end_date)
  return days_in_leap_years, days_between - days_in_leap_years


def leap_days_between(start_date, end_date):
  """Calculates number of leap days (29 Feb) between two dates.

  'start_date' is included and 'end_date' is excluded from the period.

  For example, for dates `2019-12-24` and `2024-3-10` the result is
  2: there is 29 Feb 2020 and 29 Feb 2024 between 24 Dec 2019 (inclusive) and
  10 Mar 2024 (exclusive).

  If `end_date` is earlier than `start_date`, the result will be negative or
  zero.

  Args:
    start_date: DateTensor.
    end_date: DateTensor compatible with `start_date`.

  Returns:
    Tensor of type 'int32'.
  """
  def leap_days_since_year_0(date_tensor):
    year = date_tensor.year()
    month = date_tensor.month()
    leap_years_since_0 = year // 4 - year // 100 + year // 400
    needs_adjustment = (is_leap_year(year) & (month <= 2))
    return leap_years_since_0 - tf.where(needs_adjustment, 1, 0)
  return leap_days_since_year_0(end_date) - leap_days_since_year_0(start_date)


__all__ = [
    "ordinal_to_year_month_day",
    "year_month_day_to_ordinal",
    "is_leap_year",
    "days_in_leap_years_between",
    "days_in_leap_and_nonleap_years_between",
    "leap_days_between",
]
