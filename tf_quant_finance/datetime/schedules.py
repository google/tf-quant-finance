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
"""Functions for creating schedules."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import date_tensor


_MIN_DAYS_IN_PERIOD = {
    constants.PeriodType.DAY: 1,
    constants.PeriodType.WEEK: 7,
    constants.PeriodType.MONTH: 28,
    constants.PeriodType.YEAR: 365
}


class PeriodicSchedule:
  """Defines an array of dates specified by a regular schedule."""

  def __init__(self,
               *,
               start_date,
               end_date,
               tenor,
               holiday_calendar=None,
               roll_convention=constants.BusinessDayConvention.NONE,
               backward=False,
               end_of_month=False):
    """Initializes the schedule.

    Initializes a schedule with a given tenor, date range and holiday calendar.

    A schedule is an increasing sequence of dates at a regular interval subject
    to holiday adjustments.

    The rules for schedule generation (accessed via the `dates()` method)
    are as follows.

    (a) If `backward=False`, take `start_date` and add `tenor` multiplied by
      0, 1, 2, etc. until the resulting date is greater than `end_date`.
    (b) If `backward=True`, take `end_date` and subtract `tenor` multiplied by
      0, 1, 2, etc. until the resulting date is smaller than `start_date`.
      Ensure that the result is in ascending order.
    (c) Both `start_date` and `end_date` are included, even if the distance
      between then is not an integer number of tenor periods.
    (d) If `holiday_calendar` is specified, roll all the dates according to
      `roll_convention`. The rolling includes `start_date` and `end_date` when
      they are part of resulting schedule. Thus if `start_date` or `end_date`
      fall on holidays, they will change and may go out of the
      [`start_date`, `end_date`] interval.

    Note that `tenor = PeriodType.DAY` is treated as an actual day, not as
    a business day. So a schedule with `tenor = days(7)` is the same
    as one with `tenor = week()`.

    The `dates()` can create multiple schedules simultaneously.
    The start and end dates may have any (compatible) shape.
    The `DateTensor` returned by `dates()` has the shape
    `start_date.shape + (n,)`, where `n` is the maximum length of schedules in
    the batch. If schedules have different lengths, the extra elements will be
    padded with extra `end_date` elements at the end, if `backward=False`
    and with extra `start_date` elements in the beginning if
    `backward=True`. In all cases each schedule in the batch is monotonic.

    The following examples demonstrate the batch and non-batch usage.

    #### Example Usage (Non-batch)

    ```python
      start_date = datetime.dates_from_tuples([(2020, 1, 18)])
      end_date = datetime.dates_from_tuples([(2021, 3, 25)])
      tenor = datetime.months(3)
      backward = False
      holiday_calendar = dates.HolidayCalendar(start_year=2020, end_year=2021)
      roll_convention = dates.BusinessDayConvention.FOLLOWING
      schedule = datetime.PeriodicSchedule(
          start_date=start_date,
          end_date=end_date,
          tenor=tenor,
          holiday_calendar=holiday_calendar,
          roll_convention=datetime.BusinessDayConvention.FOLLOWING,
          backward=backward).dates()
      # schedule is a DateTensor of
      # [[(2020, 1, 18), (2020, 4, 20), (2020, 7, 20), (2020, 10, 19),
      #   (2021, 1, 18), (2021, 3, 25)]] for backward = False and
      # [[(2020, 1, 18), (2020, 3, 25), (2020, 6, 25), (2020, 9, 25),
      #   (2020, 12, 25), (2021, 3, 25)]] for backward = True.
    ```

    The following example demonstrates this batching property.

    #### Example Usage (Batch)

    ```python
      start_date = datetime.dates_from_tuples([(2020, 1, 15), (2020, 4, 15)])
      end_date = datetime.dates_from_tuples([(2021, 3, 31), (2021, 1, 1)])
      tenor = datetime.months([4, 3])
      schedule = datetime.PeriodicSchedule(
          start_dates,
          end_dates,
          tenors,
          dates.HolidayCalendar(start_year=2020, end_year=2021),
          roll_convention=datetime.BusinessDayConvention.FOLLOWING,
          backward=False).dates()
      # Returns DateTensor of
      # [[(2020, 1, 15), (2020, 5, 15), (2020, 9, 15), (2021, 1, 15),
      #   (2021, 3, 31)],
      # [(2020, 4, 15), (2020, 7, 15), (2020, 10, 15), (2021, 1, 1),
      #  (2021, 1, 1)]].
    ```

    Args:
      start_date: `DateTensor`. Defines the lower boundary of schedule. If
        `backward=True` must be broadcastable to `end_date`, otherwise has
        arbitrary shape.
      end_date: `DateTensor`. Defines the upper boundary of the schedule.
        If `backward=False` must be broadcastable to `start_date`, otherwise has
        arbitrary shape.
      tenor: `PeriodTensor`. Defines the frequency of the schedule. Must
        be broadcastable to `start_date` if `backward=False`, and to `end_date`
        if `backward=True`.
      holiday_calendar: `dates.HolidayCalendar`. If `None`, the dates in the
        schedule will not be rolled to business days.
      roll_convention: BusinessDayConvention. Defines how dates in the schedule
        should be rolled to business days if they fall on holidays. Ignored if
        `holiday_calendar = None`.
        Default value: BusinessDayConvention.NONE (i.e. no rolling).
      backward: Python `bool`. Whether to build the schedule from the
        `start_date` moving forwards or from the `end_date` and moving
        backwards.
      end_of_month: Python `bool`. If `True`, shifts all dates in schedule to
        the ends of corresponding months, if `start_date` or `end_date` (
        depending on `backward`) is at the end of a month. The shift is applied
        before applying `roll_convention`. In the batched case, only those
        schedules in a batch, whose corresponding `start_date` (or `end_date`)
        are at ends of months, will be shifted.
    """
    if end_of_month and tenor.period_type() not in [constants.PeriodType.MONTH,
                                                    constants.PeriodType.YEAR]:
      raise ValueError(
          "end_of_month may only be used with tenors of PeriodType.MONTH or "
          "PeriodType.YEAR"
      )

    self._start_date = start_date
    self._end_date = end_date
    self._tenor = tenor
    self._holiday_calendar = holiday_calendar
    self._roll_convention = roll_convention
    self._backward = backward
    self._end_of_month = end_of_month

  def dates(self):
    """Returns the dates as computed from the schedule as a DateTensor.

    Constructs the date schedule from the supplied data. For more details see
    the initializer docstring.

    Returns:
      `DateTensor` of rank one more than `start_date` or `end_date`
      (depending on `backwards`), representing schedules for each element
      of the input.
    """
    return _gen_periodic_schedule(
        self._start_date,
        self._end_date,
        self._tenor,
        holiday_calendar=self._holiday_calendar,
        roll_convention=self._roll_convention,
        backward=self._backward,
        end_of_month=self._end_of_month)

  @property
  def start_date(self):
    return self._start_date

  @property
  def end_date(self):
    return self._end_date

  @property
  def tenor(self):
    return self._tenor

  @property
  def holiday_calendar(self):
    return self._holiday_calendar

  @property
  def roll_convention(self):
    return self._roll_convention

  @property
  def generate_backwards(self):
    """Returns whether the schedule is generated from the end date."""
    return self._backward

  @property
  def end_of_month(self):
    return self._end_of_month


class BusinessDaySchedule:
  """Generates schedules containing every business day in a period."""

  def __init__(self,
               *,
               start_date,
               end_date,
               holiday_calendar,
               backward=False):
    """Initializes the schedule.

    Initializes a schedule with a given date range and holiday calendar.

    The schedule includes all business days between and including `start_date`
    and `end_date`.

    Can create multiple schedules simultaneously. The start and end dates may
    have any (compatible) shape. The `DateTensor` returned by `dates()` has the
    shape `start_date.shape + (n,)`, where `n` is the maximum length of
    schedules in the batch. If schedules have different lengths, the extra
    elements will be padded with extra `end_date` elements at the end, if
    `backward=False` and with extra `start_date` elements in the beginning if
    `backward=True`. In all cases each schedule in the batch is monotonic.

    #### Example Usage (Non-batch)

    ```python
      start_date = datetime.dates_from_tuples([(2020, 3, 19)])
      end_date = datetime.dates_from_tuples([(2021, 3, 25)])
      holiday_calendar = datetime.HolidayCalendar(start_year=2020,
                                                  end_year=2021)
      schedule = datetime.BusinessDaysSchedule(
          start_date=start_date,
          end_date=end_date,
          holiday_calendar=holiday_calendar,
          roll_convention=datetime.BusinessDayConvention.FOLLOWING,
          backward=False).dates()
      # schedule is a DateTensor of
      # [[(2020, 3, 19), (2020, 3, 20), (2020, 3, 23), (2020, 3, 24),
      #   (2021, 3, 25)]] regardless of `backward`.
    ```

    #### Example Usage (Batch)

    ```python
      start_date = datetime.dates_from_tuples([(2020, 3, 19), (2020, 4, 15)])
      end_date = datetime.dates_from_tuples([(2021, 3, 13), (2021, 3, 17)])
      schedule = datetime.BusinessDaysSchedule(
          start_dates,
          end_dates,
          datetime.HolidayCalendar(start_year=2020, end_year=2021),
          backward=False).dates()
      # Returns DateTensor of
      # [[(2020, 3, 19), (2020, 3, 20), (2020, 3, 23), (2020, 3, 24),
      #   (2021, 3, 25)],
      # [(2020, 3, 13), (2020, 3, 16), (2020, 3, 17), (2020, 3, 17),
      #  (2021, 3, 17)]], if `backward` is True.
      # [[(2020, 3, 19), (2020, 3, 20), (2020, 3, 23), (2020, 3, 24),
      #   (2021, 3, 25)],
      # [(2020, 3, 13), (2020, 3, 13), (2020, 3, 13), (2020, 3, 16),
      #  (2021, 3, 17)]], if `backward` is True.
    ```
    Args:
      start_date: `dates.DateTensor`. Defines the lower boundary of schedule. If
        `backward=True` must be broadcastable to `end_date`, otherwise has
        arbitrary shape.
      end_date: `dates.DateTensor`. Defines the upper boundary of the schedule.
        If `backward=False` must be broadcastable to `start_date`, otherwise has
        arbitrary shape.
      holiday_calendar: `dates.HolidayCalendar` that defines which days will be
        included.
      backward: Python `bool`. Defines the way padding is applied in case of
        batching. If schedules in a batch have different lengths, the extra
        elements will be padded with extra `end_date` elements at the end, if
        `backward=False` and with extra `start_date` elements in the beginning
        if `backward=True`.
    """
    self._start_date = start_date
    self._end_date = end_date
    self._holiday_calendar = holiday_calendar
    self._backward = backward

  def dates(self):
    """Returns the dates as computed from the schedule as a DateTensor.

    Constructs the date schedule from the supplied data. For more details see
    the initializer docstring.

    Returns:
      `DateTensor` of rank one more than `start_date` or `end_date`
      (depending on `backwards`), representing schedules for each element
      of the input.
    """
    return _gen_business_days(self._start_date,
                              self._end_date,
                              self._holiday_calendar,
                              self._backward)

  @property
  def holiday_calendar(self):
    return self._holiday_calendar

  @property
  def start_date(self):
    return self._start_date

  @property
  def end_date(self):
    return self._end_date

  @property
  def generate_backwards(self):
    return self._backward


def _gen_periodic_schedule(start_date,
                           end_date,
                           tenor,
                           holiday_calendar=None,
                           roll_convention=constants.BusinessDayConvention.NONE,
                           backward=False,
                           end_of_month=False):
  """Generates a periodic schedule, see PeriodicSchedule."""

  # Validate inputs.
  control_deps = [
      tf.debugging.assert_greater_equal(end_date.ordinal(),
                                        start_date.ordinal()),
      tf.debugging.assert_positive(tenor.quantity())
  ]

  with tf.compat.v1.control_dependencies(control_deps):
    # Reshape the input Tensors.
    if backward:
      start_date = start_date.broadcast_to(end_date.shape)
      tenor = tenor.broadcast_to(end_date.shape)
    else:
      end_date = end_date.broadcast_to(start_date.shape)
      tenor = tenor.broadcast_to(start_date.shape)
    start_date = start_date.expand_dims(axis=-1)
    end_date = end_date.expand_dims(axis=-1)
    tenor = tenor.expand_dims(axis=-1)

    # Figure out the upper bound of the schedule length.
    min_days_in_period = _MIN_DAYS_IN_PERIOD[tenor.period_type()]
    days_between = end_date.ordinal() - start_date.ordinal() + 1
    schedule_len_upper_bound = tf.cast(
        tf.math.ceil(tf.math.reduce_max(
            days_between / (tenor.quantity() * min_days_in_period))),
        dtype=tf.int32)

    # Add the periods.
    if backward:
      # Subtract tenor * n, where n = n_max, ..., 2, 1, 0.
      tenors_expanded = tenor * tf.range(schedule_len_upper_bound - 1, -1, -1,
                                         dtype=tf.int32)
      schedules = end_date - tenors_expanded
      # Prepend start_date to ensure we always include it.
      schedules = date_tensor.DateTensor.concat((start_date, schedules),
                                                axis=-1)
      in_bounds = schedules.ordinal() >= start_date.ordinal()

      # Pad with start_date.
      schedules = date_tensor.DateTensor.where(in_bounds, schedules, start_date)

      # Find how much we overestimated max schedule length and trim the extras.
      not_start_date = tf.math.not_equal(schedules.ordinal(),
                                         start_date.ordinal())
      max_schedule_len_error = (
          tf.math.reduce_min(tf.where(not_start_date)[..., -1]) - 1)
      schedules = schedules[..., max_schedule_len_error:]
    else:
      # Add tenor * n, where n = 0, 1, 2, ..., n_max.
      tenors_expanded = tenor * tf.range(schedule_len_upper_bound,
                                         dtype=tf.int32)
      schedules = start_date + tenors_expanded
      # Append end_date to ensure we always include it.
      schedules = date_tensor.DateTensor.concat((schedules, end_date), axis=-1)

      in_bounds = schedules.ordinal() <= end_date.ordinal()

      # Pad with end_date.
      schedules = date_tensor.DateTensor.where(in_bounds, schedules, end_date)

      # Find the actual schedule length and trim the extras.
      not_end_date = tf.math.not_equal(schedules.ordinal(), end_date.ordinal())
      max_schedule_len = tf.math.reduce_max(tf.where(not_end_date)[..., -1]) + 2
      schedules = schedules[..., :max_schedule_len]

    # Move to the end of month where necessary.
    if end_of_month:
      where_cond = (end_date if backward else start_date).is_end_of_month()
      schedules = date_tensor.DateTensor.where(where_cond,
                                               schedules.to_end_of_month(),
                                               schedules)

    # Roll to business days.
    if holiday_calendar is not None:
      schedules = holiday_calendar.roll_to_business_day(schedules,
                                                        roll_convention)

    return schedules


def _gen_business_days(start_date, end_date, holiday_calendar, backward=False):
  """Generates business days between given dates, see BusinessDaySchedule."""
  # Handle the case when start_date or end_date fall on holidays.
  start_date = holiday_calendar.roll_to_business_day(
      start_date, roll_convention=constants.BusinessDayConvention.FOLLOWING)
  end_date = holiday_calendar.roll_to_business_day(
      end_date, roll_convention=constants.BusinessDayConvention.PRECEDING)

  # Validate inputs.
  control_deps = [
      tf.debugging.assert_greater_equal(end_date.ordinal(),
                                        start_date.ordinal()),
  ]
  with tf.compat.v1.control_dependencies(control_deps):
    # Reshape the input Tensors.
    if backward:
      start_date = start_date.broadcast_to(end_date.shape)
    else:
      end_date = end_date.broadcast_to(start_date.shape)
    start_date = start_date.expand_dims(axis=-1)
    end_date = end_date.expand_dims(axis=-1)

    # Find the longest schedule in the batch.
    max_len = tf.math.abs(tf.math.reduce_max(
        holiday_calendar.business_days_between(start_date, end_date))) + 1

    if backward:
      # Subtract n days, where n = max_len-1, ..., 2, 1, 0.
      days = tf.range(-max_len + 1, 1, dtype=tf.int32)
      schedules = holiday_calendar.add_business_days(end_date, days)
      in_bounds = schedules.ordinal() >= start_date.ordinal()
      # Pad with start_date.
      schedules = date_tensor.DateTensor.where(in_bounds, schedules, start_date)
    else:
      # Add n days, where n = 0, 1, 2, ..., max_len-1.
      days = tf.range(max_len, dtype=tf.int32)
      schedules = holiday_calendar.add_business_days(start_date, days)
      in_bounds = schedules.ordinal() <= end_date.ordinal()
      # Pad with end_date.
      schedules = date_tensor.DateTensor.where(in_bounds, schedules, end_date)

    return schedules
