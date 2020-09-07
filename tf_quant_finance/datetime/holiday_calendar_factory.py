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
"""Factory for HolidayCalendar implementations."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.datetime import bounded_holiday_calendar
from tf_quant_finance.datetime import unbounded_holiday_calendar


def create_holiday_calendar(
    weekend_mask=None,
    holidays=None,
    start_year=None,
    end_year=None):
  """Creates a holiday calendar.

  Each instance should be used in the context of only one graph. E.g. one can't
  create a HolidayCalendar in one tf.function and reuse it in another.

  Note: providing bounds for the calendar, i.e. `holidays` and/or `start_year`,
  `end_year` yields a better-performing calendar.

  Args:
    weekend_mask: Boolean `Tensor` of 7 elements one for each day of the week
      starting with Monday at index 0. A `True` value indicates the day is
      considered a weekend day and a `False` value implies a week day.
      Default value: None which means no weekends are applied.
    holidays: Defines the holidays that are added to the weekends defined by
      `weekend_mask`. An instance of `dates.DateTensor` or an object
      convertible to `DateTensor`.
      Default value: None which means no holidays other than those implied by
      the weekends (if any).
      Note that it is necessary to provide holidays for each year, and also
      adjust the holidays that fall on the weekends if required, e.g.
      2021-12-25 to 2021-12-24. To avoid doing this manually one can use
      AbstractHolidayCalendar from Pandas:

      ```python
      from pandas.tseries.holiday import AbstractHolidayCalendar
      from pandas.tseries.holiday import Holiday
      from pandas.tseries.holiday import nearest_workday

      class MyCalendar(AbstractHolidayCalendar):
          rules = [
              Holiday('NewYear', month=1, day=1, observance=nearest_workday),
              Holiday('Christmas', month=12, day=25,
                       observance=nearest_workday)
          ]

      calendar = MyCalendar()
      holidays_index = calendar.holidays(
          start=datetime.date(2020, 1, 1),
          end=datetime.date(2030, 12, 31))
      holidays = np.array(holidays_index.to_pydatetime(), dtype="<M8[D]")
      ```

    start_year: Integer giving the earliest year this calendar includes. If
      `holidays` is specified, then `start_year` and `end_year` are ignored,
      and the boundaries are derived from `holidays`.
      Default value: None which means start year is inferred from `holidays`, if
      present.
    end_year: Integer giving the latest year this calendar includes. If
      `holidays` is specified, then `start_year` and `end_year` are ignored,
      and the boundaries are derived from `holidays`.
      Default value: None which means start year is inferred from `holidays`, if
      present.

  Returns:
    A HolidayCalendar instance.
  """
  # Choose BoundedHolidayCalendar if possible, for better performance, otherwise
  # choose UnboundedHolidayCalendar.
  is_bounded = (_tensor_is_not_empty(holidays) or
                (start_year is not None and end_year is not None))
  if is_bounded:
    return bounded_holiday_calendar.BoundedHolidayCalendar(
        weekend_mask, holidays, start_year, end_year)
  return unbounded_holiday_calendar.UnboundedHolidayCalendar(
      weekend_mask, holidays)


def _tensor_is_not_empty(t):
  """Returns whether t is definitely not empty."""
  # False means either empty or unknown.
  if t is None:
    return False
  if isinstance(t, np.ndarray):
    return t.size > 0
  if isinstance(t, tf.Tensor):
    num_elem = t.shape.num_elements
    return num_elem is not None and num_elem > 0  # None means shape is unknown.
  return bool(t)
