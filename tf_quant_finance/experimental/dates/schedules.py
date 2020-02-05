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

import tensorflow as tf

from tf_quant_finance.experimental.dates import constants
from tf_quant_finance.experimental.dates import date_tensor


_MIN_DAYS_IN_PERIOD = {
    constants.PeriodType.DAY: 1,
    constants.PeriodType.WEEK: 7,
    constants.PeriodType.MONTH: 28,
    constants.PeriodType.YEAR: 365
}


def schedule(start_date,
             end_date,
             tenor,
             holiday_calendar=None,
             roll_convention=constants.BusinessDayConvention.NONE,
             backward=False):
  """Makes a schedule with a given tenor, date range and holiday calendar.

  A schedule is an increasing sequence of dates at a regular interval subject to
  holiday adjustments.

  The rules for schedule generation are as follows.

  - If `backward=False`, take `start_date` and add `tenor` multiplied by 0, 1,
  2, etc. until the resulting date is greater than `end_date`.

  - If `backward=True`, take `end_date` and subtract `tenor` multiplied by 0, 1,
  2, etc. until the resulting date is smaller than `start_date`. Ensure the
  result is in increasing order.

  - Both `start_date` and `end_date` are included, even if the distance between
  then is not an integer number of tenor periods.

  - If `holiday_calendar` is specified, roll all the dates according to
  `roll_convention`. The rolling includes `start_date` and `end_date` when they
  are part of resulting schedule. Thus if `start_date` or `end_date` fall on
  holidays, they will change and may go out of [`start_date`, `end_date`]
  interval.

  Example:

  ```python
    start_date = dates.DateTensor.from_tuples([(2020, 1, 18)])
    end_date = dates.DateTensor.from_tuples([(2021, 3, 25)])
    tenor = dates.months(3)
    backward = False
    schedule = dates.schedule(
        start_dates,
        end_dates,
        tenors,
        dates.HolidayCalendar(start_year=2020, end_year=2021),
        roll_convention=dates.BusinessDayConvention.FOLLOWING,
        backward=backward)
    # schedule is a DateTensor of
    # [[(2020, 1, 18), (2020, 4, 20), (2020, 7, 20), (2020, 10, 19),
    #   (2021, 1, 18), (2021, 3, 25)]] for backward = False and
    # [[(2020, 1, 18), (2020, 3, 25), (2020, 6, 25), (2020, 9, 25),
    #   (2020, 12, 25), (2021, 3, 25)]] for backward = True.
  ```

  Note that PeriodType.DAY is treated as actual day, not as business day.
  So schedule with `tenor = dates.days(7)` is the same as one with `tenor =
  dates.week()`.

  This method can create multiple schedules simultaneously. If `backward=False`,
  `start_date` can have arbitrary shape, and the returned DateTensor has shape
  `start_date.shape + (n,)`, where `n` is the maximum length of schedules in the
  batch. Similarly for `backward=True`. If schedules have different lengths, the
  extra elements will be padded with extra `end_date` elements at the end, if
  `backward=False`, or extra `start_date` elements in the beginning if
  `backward=True`. In all cases each schedule in the batch is monotonic.

  Example:

  ```python
    start_date = dates.DateTensor.from_tuples([(2020, 1, 15), (2020, 4, 15)])
    end_date = dates.DateTensor.from_tuples([(2021, 3, 31), (2021, 1, 1)])
    tenor = dates.months([4, 3])
    actual_schedule = dates.schedule(
        start_dates,
        end_dates,
        tenors,
        dates.HolidayCalendar(start_year=2020, end_year=2021),
        roll_convention=dates.BusinessDayConvention.FOLLOWING,
        backward=False)
    # Returns DateTensor of
    # [[(2020, 1, 15), (2020, 5, 15), (2020, 9, 15), (2021, 1, 15),
    #   (2021, 3, 31)],
    # [(2020, 4, 15), (2020, 7, 15), (2020, 10, 15), (2021, 1, 1),
    #  (2021, 1, 1)]].
  ```

  Args:
    start_date: DateTensor. Defines the lower boundary of schedule. If
      `backward=True` must be broadcastable to `end_date`, otherwise has
      arbitrary shape.
    end_date: DateTensor. Defines the upper boundary of the schedule. If
      `backward=False` must be broadcastable to `start_date`, otherwise has
      arbitrary shape.
    tenor: PeriodTensor. Defines the step of the schedule. Must be broadcastable
      to `start_date` if `backward=False`, and to `end_date` if `backward=True`.
    holiday_calendar: HolidayCalendar. If `None`, the dates in the schedule will
      not be rolled to business days.
    roll_convention: BusinessDayConvention. Defines how dates in the schedule
      should be rolled to business days if they fall on holidays. Ignored if
      `holiday_calendar = None`.
    backward: Boolean. Whether to build the schedule from `start_date` forward
      or from `end_date` backward.

  Returns:
    DateTensor of one rank higher than `start_date` or `end_date` (depending on
     `backwards`), representing schedules for each element of the input.
  """

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

    # Roll to business days.
    if holiday_calendar is not None:
      schedules = holiday_calendar.roll_to_business_day(schedules,
                                                        roll_convention)

    return schedules
