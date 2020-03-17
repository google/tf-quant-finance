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
"""HolidayCalendar definition."""

import collections
import datetime

import attr
import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.experimental.dates import constants
from tf_quant_finance.experimental.dates import date_tensor as dt
from tf_quant_finance.experimental.dates import periods


_ORDINAL_OF_1_1_1970 = 719163


class HolidayCalendar(object):
  """Represents a holiday calendar.

  Provides methods for manipulating the dates taking into account the holidays,
  and the business day roll conventions. Weekends are treated as holidays.
  """

  def __init__(
      self,
      weekend_mask=None,
      holidays=None,
      start_year=None,
      end_year=None):
    """Initializer.

    Args:
      weekend_mask: Sequence of 7 elements, where "0" means work day and "1" -
        day off. The first element is Monday. By default, no weekends are
        applied. Some of the common weekend patterns are defined in
        `dates.WeekendMask`.
        Default value: None which maps to no weekend days.
      holidays: Defines the holidays that are added to the weekends defined by
        `weekend_mask`. Can be provided in following forms:
        - Iterable of tuples containing dates in (year, month, day) format:
          ```python
          holidays = [(2020, 1, 1), (2020, 12, 25),
                      (2021, 1, 1), (2021, 12, 24)]
          ```
        - Iterable of datetime.date objects:
          ```python
          holidays = [datetime.date(2020, 1, 1), datetime.date(2020, 12, 25),
                      datetime.date(2021, 1, 1), datetime.date(2021, 12, 24)]
          ```
        - A numpy array of type np.datetime64:

          ```python
          holidays = np.array(['2020-01-01', '2020-12-25', '2021-01-01',
                               '2020-12-24'], dtype=np.datetime64)
          ```

        Note that it is necessary to provide holidays for each year, and also
        adjust the holidays that fall on the weekends if required, like
        2021-12-25 to 2021-12-24 in the example above. To avoid doing this
        manually one can use AbstractHolidayCalendar from Pandas:

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
        holidays_index = holidays.holidays(
            start=datetime.date(2020, 1, 1),
            end=datetime.date(2030, 12, 31))
        holidays = np.array(holidays_index.to_pydatetime(), dtype="<M8[D]")
        ```
      start_year: Integer giving the earliest year this calendar includes. If
        `holidays` is specified, then `start_year` and `end_year` are ignored,
        and the boundaries are derived from `holidays`. If `holidays` is `None`,
        both `start_year` and `end_year` must be specified.
      end_year: Integer giving the latest year this calendar includes. If
        `holidays` is specified, then `start_year` and `end_year` are ignored,
        and the boundaries are derived from `holidays`. If `holidays` is `None`,
        both `start_year` and `end_year` must be specified.
    """
    self._weekend_mask = np.array(weekend_mask or constants.WeekendMask.NONE)
    self._holidays_np = _to_np_holidays_array(holidays)
    start_year, end_year = _resolve_calendar_boundaries(self._holidays_np,
                                                        start_year, end_year)
    self._dates_np = np.arange(
        datetime.date(start_year, 1, 1), datetime.date(end_year + 1, 1, 1),
        datetime.timedelta(days=1)).astype("<M8[D]")
    self._ordinal_offset = datetime.date(start_year, 1, 1).toordinal()

    # Precomputed tables. These are constant 1D Tensors, mapping each day in the
    # [start_year, end_year] period to some quantity of interest, e.g. next
    # business day. The tables should be indexed with
    # `date.ordinal - self._offset`. All tables are computed lazily.
    self._table_cache = _TableCache()

  def is_business_day(self, date_tensor):
    """Returns a tensor of bools for whether given dates are business days."""
    is_bus_day_table = self._compute_is_bus_day_table()
    is_bus_day_int32 = self._gather(
        is_bus_day_table,
        date_tensor.ordinal() - self._ordinal_offset)
    return tf.cast(is_bus_day_int32, dtype=tf.bool)

  def roll_to_business_day(self, date_tensor, roll_convention):
    """Rolls the given dates to business dates according to given convention.

    Args:
      date_tensor: DateTensor of dates to roll from.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    if roll_convention == constants.BusinessDayConvention.NONE:
      return date_tensor
    adjusted_ordinals_table = self._compute_rolled_dates_table(roll_convention)
    ordinals_with_offset = date_tensor.ordinal() - self._ordinal_offset
    adjusted_ordinals = self._gather(adjusted_ordinals_table,
                                     ordinals_with_offset)
    return dt.from_ordinals(adjusted_ordinals, validate=False)

  def add_period_and_roll(self,
                          date_tensor,
                          period_tensor,
                          roll_convention=constants.BusinessDayConvention.NONE):
    """Adds given periods to given dates and rolls to business days.

    The original dates are not rolled prior to addition.

    Args:
      date_tensor: DateTensor of dates to add to.
      period_tensor: PeriodTensor broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    return self.roll_to_business_day(date_tensor + period_tensor,
                                     roll_convention)

  def add_business_days(self,
                        date_tensor,
                        num_days,
                        roll_convention=constants.BusinessDayConvention.NONE):
    """Adds given number of business days to given dates.

    Note that this is different from calling `add_period_and_roll` with
    PeriodType.DAY. For example, adding 5 business days to Monday gives the next
    Monday (unless there are holidays on this week or next Monday). Adding 5
    days and rolling means landing on Saturday and then rolling either to next
    Monday or to Friday of the same week, depending on the roll convention.

    If any of the dates in `date_tensor` are not business days, they will be
    rolled to business days before doing the addition. If `roll_convention` is
    `NONE`, and any dates are not business days, an exception is raised.

    Args:
      date_tensor: DateTensor of dates to advance from.
      num_days: Tensor of int32 type broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    control_deps = []
    if roll_convention == constants.BusinessDayConvention.NONE:
      message = ("Some dates in date_tensor are not business days. "
                 "Please specify the roll_convention argument.")
      is_bus_day = self.is_business_day(date_tensor)
      control_deps.append(
          tf.debugging.assert_equal(is_bus_day, True, message=message))
    else:
      date_tensor = self.roll_to_business_day(date_tensor, roll_convention)

    with tf.compat.v1.control_dependencies(control_deps):
      cumul_bus_days_table = self._compute_cumul_bus_days_table()
      cumul_bus_days = self._gather(
          cumul_bus_days_table,
          date_tensor.ordinal() - self._ordinal_offset)
      target_cumul_bus_days = cumul_bus_days + num_days

      bus_day_ordinals_table = self._compute_bus_day_ordinals_table()
      ordinals = self._gather(bus_day_ordinals_table, target_cumul_bus_days)
      return dt.from_ordinals(ordinals, validate=False)

  def subtract_period_and_roll(
      self,
      date_tensor,
      period_tensor,
      roll_convention=constants.BusinessDayConvention.NONE):
    """Subtracts given periods from given dates and rolls to business days.

    The original dates are not rolled prior to subtraction.

    Args:
      date_tensor: DateTensor of dates to subtract from.
      period_tensor: PeriodTensor broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    minus_period_tensor = periods.PeriodTensor(-period_tensor.quantity(),
                                               period_tensor.period_type())
    return self.add_period_and_roll(date_tensor, minus_period_tensor,
                                    roll_convention)

  def subtract_business_days(
      self,
      date_tensor,
      num_days,
      roll_convention=constants.BusinessDayConvention.NONE):
    """Adds given number of business days to given dates.

    Note that this is different from calling `subtract_period_and_roll` with
    PeriodType.DAY. For example, subtracting 5 business days from Friday gives
    the previous Friday (unless there are holidays on this week or previous
    Friday). Subtracting 5 days and rolling means landing on Sunday and then
    rolling either to Monday or to Friday, depending on the roll convention.

    If any of the dates in `date_tensor` are not business days, they will be
    rolled to business days before doing the subtraction. If `roll_convention`
    is `NONE`, and any dates are not business days, an exception is raised.

    Args:
      date_tensor: DateTensor of dates to advance from.
      num_days: Tensor of int32 type broadcastable to `date_tensor`.
      roll_convention: BusinessDayConvention. Determines how to roll a date that
        falls on a holiday.

    Returns:
      The resulting DateTensor.
    """
    return self.add_business_days(date_tensor, -num_days, roll_convention)

  def business_days_in_period(self, date_tensor, period_tensor):
    """Calculates number of business days in a period.

    Includes the dates in `date_tensor`, but excludes final dates resulting from
    addition of `period_tensor`.

    Args:
      date_tensor: DateTensor of starting dates.
      period_tensor: PeriodTensor, should be broadcastable to `date_tensor`.

    Returns:
       An int32 Tensor with the number of business days in given periods that
       start at given dates.

    """
    return self.business_days_between(date_tensor, date_tensor + period_tensor)

  def business_days_between(self, from_dates, to_dates):
    """Calculates number of business between pairs of dates.

    For each pair, the initial date is included in the difference, and the final
    date is excluded. If the final date is the same or earlier than the initial
    date, zero is returned.

    Args:
      from_dates: DateTensor of initial dates.
      to_dates: DateTensor of final dates, should be broadcastable to
        `from_dates`.

    Returns:
       An int32 Tensor with the number of business days between the
       corresponding pairs of dates.
    """
    cumul_bus_days_table = self._compute_cumul_bus_days_table()
    ordinals_1, ordinals_2 = from_dates.ordinal(), to_dates.ordinal()
    ordinals_2 = tf.broadcast_to(ordinals_2, ordinals_1.shape)
    cumul_bus_days_1 = self._gather(cumul_bus_days_table,
                                    ordinals_1 - self._ordinal_offset)
    cumul_bus_days_2 = self._gather(cumul_bus_days_table,
                                    ordinals_2 - self._ordinal_offset)
    return tf.math.maximum(cumul_bus_days_2 - cumul_bus_days_1, 0)

  def _compute_rolled_dates_table(self, roll_convention):
    """Computes and caches rolled dates table."""
    already_computed = self._table_cache.rolled_dates.get(roll_convention, None)
    if already_computed is not None:
      return already_computed

    roll_convention_np = _to_np_roll_convention(roll_convention)
    holidays_arg = self._holidays_np
    if holidays_arg is None:
      holidays_arg = []  # np.busday_offset doesn't accept None
    adjusted_np = np.busday_offset(
        dates=self._dates_np,
        offsets=0,
        roll=roll_convention_np,
        weekmask=1 - self._weekend_mask,
        holidays=holidays_arg)
    rolled_date_table = adjusted_np.astype(np.int32) + _ORDINAL_OF_1_1_1970

    # To make tensor caching safe, lift the ops out of the current scope using
    # tf.init_scope(). This allows e.g. to cache these tensors in one
    # tf.function and reuse them in another tf.function.
    with tf.init_scope():
      rolled_date_table = tf.convert_to_tensor(rolled_date_table,
                                               name="rolled_date_table")
    self._table_cache.rolled_dates[roll_convention] = rolled_date_table
    return rolled_date_table

  def _compute_is_bus_day_table(self):
    """Computes and caches "is business day" table."""
    if self._table_cache.is_bus_day is not None:
      return self._table_cache.is_bus_day

    is_bus_day_table = np.ones_like(self._dates_np, dtype=np.int32)

    ordinals = np.arange(self._ordinal_offset,
                         self._ordinal_offset + len(is_bus_day_table))
    # Apply week mask
    week_days = (ordinals - 1) % 7
    is_bus_day_table[self._weekend_mask[week_days] == 1] = 0

    # Apply holidays
    if self._holidays_np is not None:
      holiday_ordinals = (
          np.array(self._holidays_np, dtype=np.int32) + _ORDINAL_OF_1_1_1970)
      is_bus_day_table[holiday_ordinals - self._ordinal_offset] = 0

    with tf.init_scope():
      is_bus_day_table = tf.convert_to_tensor(is_bus_day_table,
                                              name="is_bus_day_table")
    self._table_cache.is_bus_day = is_bus_day_table
    return is_bus_day_table

  def _compute_cumul_bus_days_table(self):
    """Computes and caches cumulative business days table."""
    if self._table_cache.cumul_bus_days is not None:
      return self._table_cache.cumul_bus_days

    is_bus_day_table = self._compute_is_bus_day_table()
    with tf.init_scope():
      cumul_bus_days_table = tf.math.cumsum(is_bus_day_table, exclusive=True,
                                            name="cumul_bus_days_table")
      self._table_cache.cumul_bus_days = cumul_bus_days_table
    return cumul_bus_days_table

  def _compute_bus_day_ordinals_table(self):
    """Computes and caches rolled business day ordinals table."""
    if self._table_cache.bus_day_ordinals is not None:
      return self._table_cache.bus_day_ordinals

    is_bus_day_table = self._compute_is_bus_day_table()
    with tf.init_scope():
      bus_day_ordinals_table = tf.cast(
          tf.compat.v2.where(is_bus_day_table)[:, 0] + self._ordinal_offset,
          tf.int32, name="bus_day_ordinals_table")
      self._table_cache.bus_day_ordinals = bus_day_ordinals_table
    return bus_day_ordinals_table

  def _gather(self, table, indices):
    message = "Went out of calendar boundaries!"
    assert1 = tf.debugging.assert_greater_equal(indices, 0, message=message)
    assert2 = tf.debugging.assert_less(indices, len(self._dates_np),
                                       message=message)
    with tf.compat.v1.control_dependencies([assert1, assert2]):
      return tf.gather(table, indices)


def _to_np_holidays_array(holidays):
  """Converts holidays from any acceptable format to np.datetime64 array."""
  if holidays is None:
    return None
  if isinstance(holidays, collections.Iterable):
    if all(isinstance(h, datetime.date) for h in holidays):
      return np.array(list(holidays), "<M8[D]")
    if all(isinstance(h, tuple) for h in holidays):
      datetimes = [datetime.date(*t) for t in holidays]
      return np.array(datetimes, "<M8[D]")
  if isinstance(holidays, np.ndarray):
    return holidays.astype("<M8[D]")
  raise ValueError("Unrecognized format of holidays")


def _to_np_roll_convention(convention):
  if convention == constants.BusinessDayConvention.FOLLOWING:
    return "following"
  if convention == constants.BusinessDayConvention.PRECEDING:
    return "preceding"
  if convention == constants.BusinessDayConvention.MODIFIED_FOLLOWING:
    return "modifiedfollowing"
  if convention == constants. BusinessDayConvention.MODIFIED_PRECEDING:
    return "modifiedpreceding"
  raise ValueError("Unrecognized convention: {}".format(convention))


def _resolve_calendar_boundaries(holidays_np, start_year, end_year):
  if holidays_np is None or holidays_np.size == 0:
    if start_year is None or end_year is None:
      raise ValueError("Please specify either holidays or both start_year and"
                       "end_year arguments")
    return start_year, end_year

  years = [date.year for date in holidays_np.astype(object)]
  return np.min(years), np.max(years)


@attr.s
class _TableCache(object):
  """Cache of pre-computed tables."""

  # Tables of rolled date ordinals keyed by BusinessDayConvention.
  rolled_dates = attr.ib(factory=dict)

  # Table with "1" on business days and "0" otherwise.
  is_bus_day = attr.ib(default=None)

  # Table with number of business days before each date. Starts with 0.
  cumul_bus_days = attr.ib(default=None)

  # Table with ordinals of each business day in the [start_year, end_year],
  # in order.
  bus_day_ordinals = attr.ib(default=None)
