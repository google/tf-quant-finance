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
"""DateTensor definition."""

import collections
import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental.dates import date_utils
from tf_quant_finance.experimental.dates.constants import Month
from tf_quant_finance.experimental.dates.constants import PeriodType
from tf_quant_finance.experimental.dates.periods import PeriodTensor
from tf_quant_finance.experimental.dates.tensor_wrapper import TensorWrapper

_ORDINAL_OF_1_1_1970 = 719163

_days_in_months = [0,  # For easier indexing
                   31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                   31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def convert_to_date_tensor(date_inputs):
  """Converts supplied data to a `DateTensor` if possible.

  Args:
    date_inputs: One of the supported types that can be converted to a
      DateTensor. The following input formats are supported. 1. Sequence of
      `datetime.datetime`, `datetime.date`, or any other structure with
      data attributes called 'year', 'month' and 'day'. 2. A numpy array of
      `datetime64` type. 3. Sequence of (year, month, day) Tuples.
      Months are 1-based (with January as 1) and constants.Months enum may be
      used instead of ints. Days are also 1-based. 4. A tuple of three
      int32 `Tensor`s containing year,
      month and date as positive integers in that order. 5. A single int32
      `Tensor` containing ordinals (i.e. number of days since 31 Dec 0 with 1
      being 1 Jan 1.)

  Returns:
    A `DateTensor` object representing the supplied dates.

  Raises:
    ValueError: If conversion fails for any reason.
  """
  if isinstance(date_inputs, DateTensor):
    return date_inputs

  if isinstance(date_inputs, np.ndarray):  # Case 2.
    date_inputs = date_inputs.astype("datetime64[D]")
    return DateTensor.from_np_datetimes(date_inputs)

  if tf.is_tensor(date_inputs):  # Case 5
    return DateTensor.from_ordinals(date_inputs)

  if isinstance(date_inputs, collections.Sequence):
    if not date_inputs:
      return DateTensor.from_ordinals([])
    test_element = date_inputs[0]
    if hasattr(test_element, "year"):  # Case 1.
      return DateTensor.from_datetimes(date_inputs)
    # Case 3
    if isinstance(test_element, collections.Sequence):
      return DateTensor.from_tuples(date_inputs)
    if len(date_inputs) == 3:  # Case 4.
      return DateTensor.from_year_month_day_tensors(date_inputs[0],
                                                    date_inputs[1],
                                                    date_inputs[2])
  # As a last ditch effort, try to convert the sequence to a Tensor to see if
  # that can work
  try:
    as_ordinals = tf.convert_to_tensor(date_inputs, dtype=tf.int32)
    return DateTensor.from_ordinals(as_ordinals)
  except ValueError as e:
    raise ValueError("Failed to convert inputs to DateTensor. "
                     "Unrecognized format. Error: " + e)


class DateTensor(TensorWrapper):
  """Represents a tensor of dates."""

  @classmethod
  def from_datetimes(cls, datetimes):
    """Creates DateTensor from a sequence of Python datetime objects.

    Args:
      datetimes: Sequence of Python datetime objects.

    Returns:
      DateTensor object.

    Example:
    '''python
    import datetime

    dates = [datetime.date(2015, 4, 15), datetime.date(2017, 12, 30)]
    date_tensor = DateTensor.from_datetimes(dates)
    '''

    """
    years = tf.constant([dt.year for dt in datetimes], dtype=tf.int32)
    months = tf.constant([dt.month for dt in datetimes], dtype=tf.int32)
    days = tf.constant([dt.day for dt in datetimes], dtype=tf.int32)

    # datetime stores year, month and day internally, and datetime.toordinal()
    # performs calculations. We use a tf routine to perform these calculations
    # instead.
    return DateTensor.from_year_month_day_tensors(years, months, days,
                                                  validate=False)

  @classmethod
  def from_np_datetimes(cls, np_datetimes):
    """Creates DateTensor from a Numpy array of dtype datetime64.

    Args:
      np_datetimes: Numpy array of dtype datetime64.

    Returns:
      DateTensor object.

    Example:
    '''python
    import datetime
    import numpy as np

    date_tensor_np = np.array(
      [[datetime.date(2019, 3, 25), datetime.date(2020, 6, 2)],
       [datetime.date(2020, 9, 15), datetime.date(2020, 12, 27)]],
       dtype=np.datetime64)

    date_tensor = DateTensor.from_np_datetimes(date_tensor_np)
    '''
    """

    # There's no easy way to extract year, month, day from numpy datetime, so
    # we start with ordinals.
    ordinals = tf.constant(np_datetimes, dtype=tf.int32) + _ORDINAL_OF_1_1_1970
    return DateTensor.from_ordinals(ordinals, validate=False)

  @classmethod
  def from_tuples(cls, year_month_day_tuples, validate=True):
    """Creates DateTensor from a sequence of year-month-day Tuples.

    Args:
      year_month_day_tuples: Sequence of (year, month, day) Tuples. Months
        are 1-based; constants from Months enum can be used instead of ints.
        Days are also 1-based.
      validate: Whether to validate the dates.

    Returns:
      DateTensor object.

    Example:
    '''python
    date_tensor = DateTensor.from_tuples([(2015, 4, 15), (2017, 12, 30)])
    '''

    """
    years, months, days = [], [], []
    for t in year_month_day_tuples:
      years.append(t[0])
      months.append(t[1])
      days.append(t[2])
    years = tf.constant(years, dtype=tf.int32)
    months = tf.constant(months, dtype=tf.int32)
    days = tf.constant(days, dtype=tf.int32)
    return DateTensor.from_year_month_day_tensors(years, months, days, validate)

  @classmethod
  def from_year_month_day_tensors(cls,
                                  years,
                                  months,
                                  days,
                                  validate=True):
    """Creates DateTensor from tensors of years, months and days.

    Args:
      years: Tensor of int32 type. Elements should be positive.
      months: Tensor of int32 type of same shape as `year`. Elements should be
        in range `[1, 12]`.
      days: Tensor of int32 type of same shape as `year`. Elements should be in
        range `[1, 31]` and represent valid dates together with corresponding
        elements of `month` and `year` Tensors.
      validate: Whether to validate the dates.

    Returns:
      DateTensor object.

    Example:
    ```python
    years = tf.constant([2015, 2017], dtype=tf.int32)
    months = tf.constant([4, 12], dtype=tf.int32)
    days = tf.constant([15, 30], dtype=tf.int32)
    date_tensor = DateTensor.from_year_month_day_tensors(years, months, days)
    ```
    """
    years = tf.convert_to_tensor(years, tf.int32)
    months = tf.convert_to_tensor(months, tf.int32)
    days = tf.convert_to_tensor(days, tf.int32)

    control_deps = []
    if validate:
      control_deps.append(tf.debugging.assert_positive(years))
      control_deps.append(
          tf.debugging.assert_greater_equal(months, Month.JANUARY.value))
      control_deps.append(
          tf.debugging.assert_less_equal(months, Month.DECEMBER.value))
      control_deps.append(tf.debugging.assert_positive(days))
      is_leap = date_utils.is_leap_year(years)
      days_in_months = tf.constant(_days_in_months, tf.int32)
      max_days = tf.gather(days_in_months,
                           months + 12 * tf.dtypes.cast(is_leap, np.int32))
      control_deps.append(tf.debugging.assert_less_equal(days, max_days))
      with tf.compat.v1.control_dependencies(control_deps):
        # Ensure years, months, days themselves are under control_deps.
        years = tf.identity(years)
        months = tf.identity(months)
        days = tf.identity(days)

    with tf.compat.v1.control_dependencies(control_deps):
      ordinals = date_utils.year_month_day_to_ordinal(years, months, days)
      return DateTensor(ordinals, years, months, days)

  @classmethod
  def from_ordinals(cls, ordinals, validate=True):
    """Creates DateTensor from tensors of ordinals.

    Args:
      ordinals: Tensor of type int32. Each value is number of days since
        1 Jan 0001. 1 Jan 0001 has `ordinal=1`.
      validate: Whether to validate the dates.

    Returns:
      DateTensor object.

    Example:
    ```python

    ordinals = tf.constant([
      735703,  # 2015-4-12
      736693   # 2017-12-30
    ], dtype=tf.int32)

    date_tensor = DateTensor.from_ordinals(ordinals)
    ```
    """
    ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32)

    control_deps = []
    if validate:
      control_deps.append(tf.debugging.assert_positive(ordinals))
      with tf.compat.v1.control_dependencies(control_deps):
        ordinals = tf.identity(ordinals)

    with tf.compat.v1.control_dependencies(control_deps):
      years, months, days = date_utils.ordinal_to_year_month_day(ordinals)
      return DateTensor(ordinals, years, months, days)

  def __init__(self, ordinals, years, months, days):
    """Initializer.

    This initializer is for internal use. Consider using one of the static
    builder methods above.

    Args:
      ordinals: Tensor of type int32. Each value is number of days since
        1 Jan 0001. 1 Jan 0001 has `ordinal=1`. `years`, `months` and `days`
        must represent the same dates as `ordinals`.
      years: Tensor of type int32, of same shape as `ordinals`.
      months: Tensor of type int32, of same shape as `ordinals`
      days: Tensor of type int32, of same shape as `ordinals`.
    """
    # The internal representation of a DateTensor is all four int32 Tensors
    # (ordinals, years, months, days). Why do we need such redundancy?
    #
    # Imagine we kept only ordinals, and consider the following example:
    # User creates a DateTensor, adds a certain number of months, and then
    # calls .days() on resulting DateTensor. The transformations would be as
    # follows: o -> y, m, d -> y', m', d' -> o' -> y', m', d'.
    # The first transformation is required for adding months.
    # The second is actually adding months. Third - for creating a new
    # DateTensor object that is backed by o'. Last - to yield a result from
    # new_date_tensor.days(). The last transformation is clearly unnecessary and
    # it's expensive.
    #
    # With a "redundant" representation we have:
    # o -> y, m, d -> y', m', d' -> o' or o <- y, m, d -> y', m', d' -> o',
    # depending on how the first DateTensor is created. new_date_tensor.days()
    # yields m', which we didn't discard, and if o and o' are never needed,
    # they'll be eliminated (in graph mode).
    #
    # A similar argument shows why (y, m, d) is not an optimal representation
    # either - for e.g. adding days instead of months.

    self._ordinals = tf.identity(ordinals, name="dt_ordinals")
    self._years = tf.identity(years, name="dt_years")
    self._months = tf.identity(months, name="dt_months")
    self._days = tf.identity(days, name="dt_days")

  def days(self):
    """Returns an int32 tensor of days since the beginning the month.

    The result is one-based, i.e. yields 1 for first days of the month.

    Example:
    ```python
    dates = DateTensor([(2019, 1, 25), (2020, 3, 2)])
    dates.days_of_month()  # [25, 2]
    ```

    """
    return self._days

  def days_of_week(self):
    """Returns an int32 tensor of weekdays.

    The result is zero-based according to Python datetime convention, i.e.
    Monday is "0".

    Example:
    ```python
    dates = DateTensor([(2019, 1, 25), (2020, 3, 2)])
    dates.days_of_week()  # [5, 1]
    ```

    """
    # 1 Jan 0001 was Monday according to the proleptic Gregorian calendar.
    # So, 1 Jan 0001 has ordinal 1, and the weekday is 0.
    return (self._ordinals - 1) % 7

  def months(self):
    """Returns an int32 tensor of months.

    Example:
    ```python
    dates = DateTensor([(2019, 1, 25), (2020, 3, 2)])
    dates.months()  # [1, 3]
    ```

    """
    return self._months

  def years(self):
    """Returns an int32 tensor of years.

    Example:
    ```python
    dates = DateTensor([(2019, 1, 25), (2020, 3, 2)])
    dates.years()  # [2019, 2020]
    ```

    """
    return self._years

  def ordinals(self):
    """Returns an int32 tensor of ordinals.

    Ordinal is the number of days since 1st Jan 0001.

    Example:
    ```python
    dates = DateTensor([(2019, 3, 25), (1, 1, 1)])
    dates.ordinals()  # [737143, 1]
    ```
    """
    return self._ordinals

  def days_until(self, target_date_tensor):
    """Returns an int32 tensor with numbers of days until the target dates.

    Args:
      target_date_tensor: a DateTensor object broadcastable to the shape of
        "self".

    Example:
    ```python
    dates = DateTensor([(2020, 1, 25), (2020, 3, 2)])
    target = DateTensor([(2020, 3, 5)])
    dates.days_until(target)  # [40, 3]

    targets = DateTensor([(2020, 2, 5), (2020, 3, 5)])
    dates.days_until(targets)  # [11, 3]
    ```
    """
    return target_date_tensor.ordinals() - self._ordinals

  def period_length_in_days(self, period_tensor):
    """Returns an int32 tensor with numbers of days each period takes.

    Args:
      period_tensor: a PeriodTensor object broadcastable to the shape of
        "self".

    Example:
    ```python
    dates = DateTensor([(2020, 2, 25), (2020, 3, 2)])
    dates.period_length_in_days(period.month())  # [29, 31]

    periods = periods.months([1, 2])
    dates.period_length_in_days(period.month())  # [29, 61]
    ```
    """
    return (self + period_tensor).ordinals() - self._ordinals

  @property
  def shape(self):
    return self._ordinals.shape

  @property
  def rank(self):
    return tf.rank(self._ordinals)

  def __add__(self, period_tensor):
    """Adds a tensor of periods.

    Args:
      period_tensor: a PeriodTensor object broadcastable to the shape of
        "self".
    Returns:
      The new instance of DateTensor.

    When adding months or years, the resulting day of the month is decreased
    to the largest valid value if necessary. E.g. 31.03.2020 + 1 month =
    30.04.2020, 29.02.2020 + 1 year = 28.02.2021.

    Example:
    ```python
    dates = DateTensor([(2020, 2, 25), (2020, 3, 31)])
    new_dates = dates + period.month()
    # DateTensor([(2020, 3, 25), (2020, 4, 30)])

    new_dates = dates + periods.months([1, 2])
    # DateTensor([(2020, 3, 25), (2020, 5, 31)])
    ```
    """
    period_type = period_tensor.period_type()

    if period_type == PeriodType.DAY:
      ordinals = self._ordinals + period_tensor.quantity()
      return DateTensor.from_ordinals(ordinals)

    if period_type == PeriodType.WEEK:
      return self + PeriodTensor(period_tensor.quantity() * 7, PeriodType.DAY)

    def adjust_day(year, month, day):
      is_leap = date_utils.is_leap_year(year)
      days_in_months = tf.constant(_days_in_months, tf.int32)
      max_days = tf.gather(days_in_months,
                           month + 12 * tf.dtypes.cast(is_leap, np.int32))
      return tf.math.minimum(day, max_days)

    if period_type == PeriodType.MONTH:
      m = self._months - 1 + period_tensor.quantity()
      y = self._years + m // 12
      m = m % 12 + 1
      d = adjust_day(y, m, self._days)
      return DateTensor.from_year_month_day_tensors(
          y, m, d, validate=False)

    if period_type == PeriodType.YEAR:
      y = self._years + period_tensor.quantity()
      m = tf.broadcast_to(self._months, y.shape)
      d = adjust_day(y, m, self._days)
      return DateTensor.from_year_month_day_tensors(
          y, m, d, validate=False)

    raise ValueError("Unrecognized period type: {}".format(period_type))

  def __sub__(self, period_tensor):
    """Subtracts a tensor of periods.

    Args:
      period_tensor: a PeriodTensor object broadcastable to the shape of
        "self".
    Returns:
      The new instance of DateTensor.

    When subtracting months or years, the resulting day of the month is
    decreased to the largest valid value if necessary. E.g. 31.03.2020 - 1 month
    = 29.02.2020, 29.02.2020 - 1 year = 28.02.2019.
    """
    return self + PeriodTensor(-period_tensor.quantity(),
                               period_tensor.period_type())

  def __eq__(self, other):
    """Compares two DateTensors by "==", returning a Tensor of bools."""
    # Note that tf doesn't override "==" and  "!=", unlike numpy.
    return tf.math.equal(self._ordinals, other.ordinals())

  def __ne__(self, other):
    """Compares two DateTensors by "!=", returning a Tensor of bools."""
    return tf.math.not_equal(self._ordinals, other.ordinals())

  def __gt__(self, other):
    """Compares two DateTensors by ">", returning a Tensor of bools."""
    return self._ordinals > other.ordinals()

  def __ge__(self, other):
    """Compares two DateTensors by ">=", returning a Tensor of bools."""
    return self._ordinals >= other.ordinals()

  def __lt__(self, other):
    """Compares two DateTensors by "<", returning a Tensor of bools."""

    return self._ordinals < other.ordinals()

  def __le__(self, other):
    """Compares two DateTensors by "<=", returning a Tensor of bools."""
    return self._ordinals <= other.ordinals()

  def __repr__(self):
    output = "DateTensor: shape={}".format(self.shape)
    if tf.executing_eagerly():
      contents_np = np.stack(
          (self._years.numpy(), self._months.numpy(), self._days.numpy()),
          axis=-1)
      return output + ", contents={}".format(repr(contents_np))
    return output

  @classmethod
  def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
    o = op_fn([t.ordinals() for t in tensor_wrappers])
    y = op_fn([t.years() for t in tensor_wrappers])
    m = op_fn([t.months() for t in tensor_wrappers])
    d = op_fn([t.days() for t in tensor_wrappers])
    return DateTensor(o, y, m, d)

  def _apply_op(self, op_fn):
    o, y, m, d = (
        op_fn(t)
        for t in (self._ordinals, self._years, self._months, self._days))
    return DateTensor(o, y, m, d)
