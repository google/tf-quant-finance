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

import attr
import tensorflow.compat.v2 as tf

from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import date_tensor as dt
from tf_quant_finance.datetime import holiday_calendar
from tf_quant_finance.datetime import periods


_ORDINAL_OF_1_1_1970 = 719163
_OUT_OF_BOUNDS_MSG = "Went out of calendar boundaries!"


class BoundedHolidayCalendar(holiday_calendar.HolidayCalendar):
  """HolidayCalendar implementation.

  Requires the calendar to be bounded. Constructs tables of precomputed values
  of interest for each day in the calendar, which enables better performance
  compared to a more flexible UnboundedHolidayCalendar.
  """

  def __init__(
      self,
      weekend_mask=None,
      holidays=None,
      start_year=None,
      end_year=None):
    """Initializer.

    Args:
      weekend_mask: Tensor of 7 elements, where "0" means work day and "1" -
        day off. The first element is Monday. By default, no weekends are
        applied. Some of the common weekend patterns are defined in
        `dates.WeekendMask`.
        Default value: None which maps to no weekend days.
      holidays: Defines the holidays that are added to the weekends defined by
      `weekend_mask`. An instance of `dates.DateTensor` or an object
       convertible to `DateTensor`.
       Default value: None which means no holidays other than those implied by
       the weekends (if any).
      start_year: Integer giving the earliest year this calendar includes. If
        `holidays` is specified, then `start_year` and `end_year` are ignored,
        and the boundaries are derived from `holidays`. If `holidays` is `None`,
        both `start_year` and `end_year` must be specified.
      end_year: Integer giving the latest year this calendar includes. If
        `holidays` is specified, then `start_year` and `end_year` are ignored,
        and the boundaries are derived from `holidays`. If `holidays` is `None`,
        both `start_year` and `end_year` must be specified.
    """
    self._weekend_mask = tf.convert_to_tensor(weekend_mask or
                                              constants.WeekendMask.NONE)
    if holidays is None:
      self._holidays = None
    else:
      self._holidays = dt.convert_to_date_tensor(holidays)
    start_year, end_year = _resolve_calendar_boundaries(self._holidays,
                                                        start_year, end_year)
    self._ordinal_offset = dt.from_year_month_day(start_year, 1, 1).ordinal()
    self._calendar_size = (
        dt.from_year_month_day(end_year + 1, 1, 1).ordinal() -
        self._ordinal_offset)

    # Precomputed tables. These are constant 1D Tensors, mapping each day in the
    # [start_year, end_year] period to some quantity of interest, e.g. next
    # business day. The tables should be indexed with
    # `date.ordinal - self._offset`. All tables are computed lazily.
    # All tables have an extra element at the beginning and the end, see comment
    # in _compute_is_bus_day_table().
    self._table_cache = _TableCache()

  def is_business_day(self, date_tensor):
    """Returns a tensor of bools for whether given dates are business days."""
    is_bus_day_table = self._compute_is_bus_day_table()
    is_bus_day_int32 = self._gather(
        is_bus_day_table,
        date_tensor.ordinal() - self._ordinal_offset + 1)
    with tf.control_dependencies(
        self._assert_ordinals_in_bounds(date_tensor.ordinal())):
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
    rolled_ordinals_table = self._compute_rolled_dates_table(roll_convention)
    ordinals_with_offset = date_tensor.ordinal() - self._ordinal_offset + 1
    rolled_ordinals = self._gather(rolled_ordinals_table, ordinals_with_offset)
    with tf.control_dependencies(
        self._assert_ordinals_in_bounds(rolled_ordinals)):
      return dt.from_ordinals(rolled_ordinals, validate=False)

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

    with tf.control_dependencies(control_deps):
      cumul_bus_days_table = self._compute_cumul_bus_days_table()
      cumul_bus_days = self._gather(
          cumul_bus_days_table,
          date_tensor.ordinal() - self._ordinal_offset + 1)
      target_cumul_bus_days = cumul_bus_days + num_days

      bus_day_ordinals_table = self._compute_bus_day_ordinals_table()
      ordinals = self._gather(bus_day_ordinals_table, target_cumul_bus_days)
      with tf.control_dependencies(self._assert_ordinals_in_bounds(ordinals)):
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
    with tf.control_dependencies(
        self._assert_ordinals_in_bounds(ordinals_1) +
        self._assert_ordinals_in_bounds(ordinals_2)):
      ordinals_2 = tf.broadcast_to(ordinals_2, ordinals_1.shape)
      cumul_bus_days_1 = self._gather(cumul_bus_days_table,
                                      ordinals_1 - self._ordinal_offset + 1)
      cumul_bus_days_2 = self._gather(cumul_bus_days_table,
                                      ordinals_2 - self._ordinal_offset + 1)
      return tf.math.maximum(cumul_bus_days_2 - cumul_bus_days_1, 0)

  def _compute_rolled_dates_table(self, convention):
    """Computes and caches rolled dates table."""
    already_computed = self._table_cache.rolled_dates.get(convention, None)
    if already_computed is not None:
      return already_computed

    # To make tensor caching safe, lift the ops out of the current scope using
    # tf.init_scope(). This allows e.g. to cache these tensors in one
    # tf.function and reuse them in another tf.function.
    with tf.init_scope():
      rolled_date_table = (
          self._compute_rolled_dates_table_without_cache(convention))
      self._table_cache.rolled_dates[convention] = rolled_date_table
    return rolled_date_table

  def _compute_rolled_dates_table_without_cache(self, convention):
    is_bus_day = self._compute_is_bus_day_table()
    cumul_bus_days = self._compute_cumul_bus_days_table()
    bus_day_ordinals = self._compute_bus_day_ordinals_table()

    if convention == constants.BusinessDayConvention.FOLLOWING:
      return tf.gather(bus_day_ordinals, cumul_bus_days)

    if convention == constants.BusinessDayConvention.PRECEDING:
      return tf.gather(bus_day_ordinals,
                       cumul_bus_days - (1 - is_bus_day))

    following = self._compute_rolled_dates_table(
        constants.BusinessDayConvention.FOLLOWING)
    preceding = self._compute_rolled_dates_table(
        constants.BusinessDayConvention.PRECEDING)
    dates_following = dt.from_ordinals(following)
    dates_preceding = dt.from_ordinals(preceding)
    original_dates = dt.from_ordinals(
        tf.range(self._ordinal_offset - 1,
                 self._ordinal_offset + self._calendar_size + 1))

    if convention == constants.BusinessDayConvention.MODIFIED_FOLLOWING:
      return tf.where(tf.equal(dates_following.month(), original_dates.month()),
                      following, preceding)

    if convention == constants. BusinessDayConvention.MODIFIED_PRECEDING:
      return tf.where(tf.equal(dates_preceding.month(), original_dates.month()),
                      preceding, following)

    raise ValueError("Unrecognized convention: {}".format(convention))

  def _compute_is_bus_day_table(self):
    """Computes and caches "is business day" table."""
    if self._table_cache.is_bus_day is not None:
      return self._table_cache.is_bus_day

    with tf.init_scope():
      ordinals = tf.range(self._ordinal_offset,
                          self._ordinal_offset + self._calendar_size)
      # Apply weekend mask
      week_days = (ordinals - 1) % 7
      is_holiday = tf.gather(self._weekend_mask, week_days)

      # Apply holidays
      if self._holidays is not None:
        indices = self._holidays.ordinal() - self._ordinal_offset
        ones_at_indices = tf.scatter_nd(
            tf.expand_dims(indices, axis=-1), tf.ones_like(indices),
            is_holiday.shape)
        is_holiday = tf.bitwise.bitwise_or(is_holiday, ones_at_indices)

      # Add a business day at the beginning and at the end, i.e. at 31 Dec of
      # start_year-1 and at 1 Jan of end_year+1. This trick is to avoid dealing
      # with special cases on boundaries.
      # For example, for Following and Preceding conventions we'd need a special
      # value that means "unknown" in the tables. More complicated conventions
      # then combine the Following and Preceding tables, and would need special
      # treatment of the "unknown" values.
      # With these "fake" business days, all computations are automatically
      # correct, unless we land on those extra days - for this reason we add
      # assertions in all API calls before returning.
      is_bus_day_table = tf.concat([[1], 1 - is_holiday, [1]], axis=0)
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
      bus_day_ordinals_table = (
          tf.cast(tf.where(is_bus_day_table)[:, 0], tf.int32) +
          self._ordinal_offset - 1)
      self._table_cache.bus_day_ordinals = bus_day_ordinals_table
    return bus_day_ordinals_table

  def _gather(self, table, indices):
    table_size = self._calendar_size + 2
    assert1 = tf.debugging.assert_greater_equal(
        indices, 0, message=_OUT_OF_BOUNDS_MSG)
    assert2 = tf.debugging.assert_less(
        indices, table_size, message=_OUT_OF_BOUNDS_MSG)
    with tf.control_dependencies([assert1, assert2]):
      return tf.gather(table, indices)

  def _assert_ordinals_in_bounds(self, ordinals):
    assert1 = tf.debugging.assert_greater_equal(
        ordinals, self._ordinal_offset, message=_OUT_OF_BOUNDS_MSG)
    assert2 = tf.debugging.assert_less(
        ordinals,
        self._ordinal_offset + self._calendar_size,
        message=_OUT_OF_BOUNDS_MSG)
    return [assert1, assert2]


def _resolve_calendar_boundaries(holidays, start_year, end_year):
  if holidays is None or holidays.shape.num_elements() in [None, 0]:
    if start_year is None or end_year is None:
      raise ValueError("Please specify either holidays or both start_year and "
                       "end_year arguments")
    return start_year, end_year

  return tf.math.reduce_min(holidays.year()), tf.math.reduce_max(
      holidays.year())


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
