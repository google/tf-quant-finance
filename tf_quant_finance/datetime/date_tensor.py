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
"""DateTensor definition."""
import collections.abc
import datetime
import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance.datetime import constants
from tf_quant_finance.datetime import date_utils
from tf_quant_finance.datetime import periods
from tf_quant_finance.datetime import tensor_wrapper

# Days in each month of a non-leap year.
_DAYS_IN_MONTHS_NON_LEAP = [
    31,  # January.
    28,  # February.
    31,  # March.
    30,  # April.
    31,  # May.
    30,  # June.
    31,  # July.
    31,  # August.
    30,  # September.
    31,  # October.
    30,  # November.
    31,  # December.
]

# Days in each month of a leap year.
_DAYS_IN_MONTHS_LEAP = [
    31,  # January.
    29,  # February.
    31,  # March.
    30,  # April.
    31,  # May.
    30,  # June.
    31,  # July.
    31,  # August.
    30,  # September.
    31,  # October.
    30,  # November.
    31,  # December.
]

# Combined list of days per month. A sentinel value of 0 is added to the top of
# the array so indexing is easier.
_DAYS_IN_MONTHS_COMBINED = [0] + _DAYS_IN_MONTHS_NON_LEAP + _DAYS_IN_MONTHS_LEAP

_ORDINAL_OF_1_1_1970 = 719163


class DateTensor(tensor_wrapper.TensorWrapper):
  """Represents a tensor of dates."""

  def __init__(self, ordinals, years, months, days):
    """Initializer.

    This initializer is primarily for internal use. More convenient construction
    methods are available via 'dates_from_*' functions.

    Args:
      ordinals: Tensor of type int32. Each value is number of days since 1 Jan
        0001. 1 Jan 0001 has `ordinal=1`. `years`, `months` and `days` must
        represent the same dates as `ordinals`.
      years: Tensor of type int32, of same shape as `ordinals`.
      months: Tensor of type int32, of same shape as `ordinals`
      days: Tensor of type int32, of same shape as `ordinals`.
    """
    # The internal representation of a DateTensor is all four int32 Tensors
    # (ordinals, years, months, days). Why do we need such redundancy?
    #
    # Imagine we kept only ordinals, and consider the following example:
    # User creates a DateTensor, adds a certain number of months, and then
    # calls .day() on resulting DateTensor. The transformations would be as
    # follows: o -> y, m, d -> y', m', d' -> o' -> y', m', d'.
    # The first transformation is required for adding months.
    # The second is actually adding months. Third - for creating a new
    # DateTensor object that is backed by o'. Last - to yield a result from
    # new_date_tensor.day(). The last transformation is clearly unnecessary and
    # it's expensive.
    #
    # With a "redundant" representation we have:
    # o -> y, m, d -> y', m', d' -> o' or o <- y, m, d -> y', m', d' -> o',
    # depending on how the first DateTensor is created. new_date_tensor.day()
    # yields m', which we didn't discard, and if o and o' are never needed,
    # they'll be eliminated (in graph mode).
    #
    # A similar argument shows why (y, m, d) is not an optimal representation
    # either - for e.g. adding days instead of months.

    self._ordinals = tf.convert_to_tensor(
        ordinals, dtype=tf.int32, name="dt_ordinals")
    self._years = tf.convert_to_tensor(years, dtype=tf.int32, name="dt_years")
    self._months = tf.convert_to_tensor(
        months, dtype=tf.int32, name="dt_months")
    self._days = tf.convert_to_tensor(days, dtype=tf.int32, name="dt_days")
    self._day_of_year = None  # Computed lazily.

  def day(self):
    """Returns an int32 tensor of days since the beginning the month.

    The result is one-based, i.e. yields 1 for first day of the month.

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
    dates.day()  # [25, 2]
    ```
    """
    return self._days

  def day_of_week(self):
    """Returns an int32 tensor of weekdays.

    The result is zero-based according to Python datetime convention, i.e.
    Monday is "0".

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
    dates.day_of_week()  # [5, 1]
    ```
    """
    # 1 Jan 0001 was Monday according to the proleptic Gregorian calendar.
    # So, 1 Jan 0001 has ordinal 1, and the weekday is 0.
    return (self._ordinals - 1) % 7

  def month(self):
    """Returns an int32 tensor of months.

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
    dates.month()  # [1, 3]
    ```
    """
    return self._months

  def year(self):
    """Returns an int32 tensor of years.

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
    dates.year()  # [2019, 2020]
    ```
    """
    return self._years

  def ordinal(self):
    """Returns an int32 tensor of ordinals.

    Ordinal is the number of days since 1st Jan 0001.

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2019, 3, 25), (1, 1, 1)])
    dates.ordinal()  # [737143, 1]
    ```
    """
    return self._ordinals

  def to_tensor(self):
    """Packs the dates into a single Tensor.

    The Tensor has shape `date_tensor.shape() + (3,)`, where the last dimension
    represents years, months and days, in this order.

    This can be convenient when the dates are the final result of a computation
    in the graph mode: a `tf.function` can return `date_tensor.to_tensor()`, or,
    if one uses `tf.compat.v1.Session`, they can call
    `session.run(date_tensor.to_tensor())`.

    Returns:
      A Tensor of shape `date_tensor.shape() + (3,)`.

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
    dates.to_tensor()  # tf.Tensor with contents [[2019, 1, 25], [2020, 3, 2]].
    ```
    """
    return tf.stack((self.year(), self.month(), self.day()), axis=-1)

  def day_of_year(self):
    """Calculates the number of days since the beginning of the year.

    Returns:
      Tensor of int32 type with elements in range [1, 366]. January 1st yields
      "1".

    #### Example

    ```python
    dt = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
    dt.day_of_year()  # [25, 62]
    ```
    """
    if self._day_of_year is None:
      cumul_days_in_month_nonleap = tf.math.cumsum(
          _DAYS_IN_MONTHS_NON_LEAP, exclusive=True)
      cumul_days_in_month_leap = tf.math.cumsum(
          _DAYS_IN_MONTHS_LEAP, exclusive=True)
      days_before_month_non_leap = tf.gather(cumul_days_in_month_nonleap,
                                             self.month() - 1)
      days_before_month_leap = tf.gather(cumul_days_in_month_leap,
                                         self.month() - 1)
      days_before_month = tf.where(
          date_utils.is_leap_year(self.year()), days_before_month_leap,
          days_before_month_non_leap)
      self._day_of_year = days_before_month + self.day()
    return self._day_of_year

  def days_until(self, target_date_tensor):
    """Computes the number of days until the target dates.

    Args:
      target_date_tensor: A DateTensor object broadcastable to the shape of
        "self".

    Returns:
      An int32 tensor with numbers of days until the target dates.

     #### Example

     ```python
    dates = tff.datetime.dates_from_tuples([(2020, 1, 25), (2020, 3, 2)])
    target = tff.datetime.dates_from_tuples([(2020, 3, 5)])
    dates.days_until(target) # [40, 3]

    targets = tff.datetime.dates_from_tuples([(2020, 2, 5), (2020, 3, 5)])
    dates.days_until(targets)  # [11, 3]
    ```
    """
    return target_date_tensor.ordinal() - self._ordinals

  def period_length_in_days(self, period_tensor):
    """Computes the number of days in each period.

    Args:
      period_tensor: A PeriodTensor object broadcastable to the shape of "self".

    Returns:
      An int32 tensor with numbers of days each period takes.

    #### Example

    ```python
    dates = tff.datetime.dates_from_tuples([(2020, 2, 25), (2020, 3, 2)])
    dates.period_length_in_days(month())  # [29, 31]

    periods = tff.datetime.months([1, 2])
    dates.period_length_in_days(periods)  # [29, 61]
    ```
    """
    return (self + period_tensor).ordinal() - self._ordinals

  def is_end_of_month(self):
    """Returns a bool Tensor indicating whether dates are at ends of months."""
    return tf.math.equal(self._days,
                         _num_days_in_month(self._months, self._years))

  def to_end_of_month(self):
    """Returns a new DateTensor with each date shifted to the end of month."""
    days = _num_days_in_month(self._months, self._years)
    return from_year_month_day(self._years, self._months, days, validate=False)

  @property
  def shape(self):
    return self._ordinals.shape

  @property
  def rank(self):
    return tf.rank(self._ordinals)

  def __add__(self, period_tensor):
    """Adds a tensor of periods.

    When adding months or years, the resulting day of the month is decreased
    to the largest valid value if necessary. E.g. 31.03.2020 + 1 month =
    30.04.2020, 29.02.2020 + 1 year = 28.02.2021.

    Args:
      period_tensor: A `PeriodTensor` object broadcastable to the shape of
      "self".

    Returns:
      The new instance of DateTensor.

    #### Example
    ```python
    dates = tff.datetime.dates_from_tuples([(2020, 2, 25), (2020, 3, 31)])
    new_dates = dates + tff.datetime.month()
    # DateTensor([(2020, 3, 25), (2020, 4, 30)])

    new_dates = dates + tff.datetime.month([1, 2])
    # DateTensor([(2020, 3, 25), (2020, 5, 31)])
    ```
    """
    period_type = period_tensor.period_type()

    if period_type == constants.PeriodType.DAY:
      ordinals = self._ordinals + period_tensor.quantity()
      return from_ordinals(ordinals)

    if period_type == constants.PeriodType.WEEK:
      return self + periods.PeriodTensor(period_tensor.quantity() * 7,
                                         constants.PeriodType.DAY)

    def adjust_day(year, month, day):
      return tf.math.minimum(day, _num_days_in_month(month, year))

    if period_type == constants.PeriodType.MONTH:
      m = self._months - 1 + period_tensor.quantity()
      y = self._years + m // 12
      m = m % 12 + 1
      d = adjust_day(y, m, self._days)
      return from_year_month_day(y, m, d, validate=False)

    if period_type == constants.PeriodType.YEAR:
      y = self._years + period_tensor.quantity()
      # Use tf.shape to handle the case of dynamically shaped `y`
      m = tf.broadcast_to(self._months, tf.shape(y))
      d = adjust_day(y, m, self._days)
      return from_year_month_day(y, m, d, validate=False)

    raise ValueError("Unrecognized period type: {}".format(period_type))

  def __sub__(self, period_tensor):
    """Subtracts a tensor of periods.

    When subtracting months or years, the resulting day of the month is
    decreased to the largest valid value if necessary. E.g. 31.03.2020 - 1 month
    = 29.02.2020, 29.02.2020 - 1 year = 28.02.2019.

    Args:
      period_tensor: a PeriodTensor object broadcastable to the shape of "self".

    Returns:
      The new instance of DateTensor.
    """
    return self + periods.PeriodTensor(-period_tensor.quantity(),
                                       period_tensor.period_type())

  def __eq__(self, other):
    """Compares two DateTensors by "==", returning a Tensor of bools."""
    # Note that tf doesn't override "==" and  "!=", unlike numpy.
    return tf.math.equal(self._ordinals, other.ordinal())

  def __ne__(self, other):
    """Compares two DateTensors by "!=", returning a Tensor of bools."""
    return tf.math.not_equal(self._ordinals, other.ordinal())

  def __gt__(self, other):
    """Compares two DateTensors by ">", returning a Tensor of bools."""
    return self._ordinals > other.ordinal()

  def __ge__(self, other):
    """Compares two DateTensors by ">=", returning a Tensor of bools."""
    return self._ordinals >= other.ordinal()

  def __lt__(self, other):
    """Compares two DateTensors by "<", returning a Tensor of bools."""

    return self._ordinals < other.ordinal()

  def __le__(self, other):
    """Compares two DateTensors by "<=", returning a Tensor of bools."""
    return self._ordinals <= other.ordinal()

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
    o = op_fn([t.ordinal() for t in tensor_wrappers])
    y = op_fn([t.year() for t in tensor_wrappers])
    m = op_fn([t.month() for t in tensor_wrappers])
    d = op_fn([t.day() for t in tensor_wrappers])
    return DateTensor(o, y, m, d)

  def _apply_op(self, op_fn):
    o, y, m, d = (
        op_fn(t)
        for t in (self._ordinals, self._years, self._months, self._days))
    return DateTensor(o, y, m, d)


def _num_days_in_month(month, year):
  """Returns number of days in a given month of a given year."""
  days_in_months = tf.constant(_DAYS_IN_MONTHS_COMBINED, tf.int32)
  is_leap = date_utils.is_leap_year(year)
  return tf.gather(days_in_months,
                   month + 12 * tf.dtypes.cast(is_leap, tf.int32))


def convert_to_date_tensor(date_inputs):
  """Converts supplied data to a `DateTensor` if possible.

  Args:
    date_inputs: One of the supported types that can be converted to a
      DateTensor. The following input formats are supported. 1. Sequence of
      `datetime.datetime`, `datetime.date`, or any other structure with data
      attributes called 'year', 'month' and 'day'. 2. A numpy array of
      `datetime64` type. 3. Sequence of (year, month, day) Tuples. Months are
      1-based (with January as 1) and constants.Months enum may be used instead
      of ints. Days are also 1-based. 4. A tuple of three int32 `Tensor`s
      containing year, month and date as positive integers in that order. 5. A
      single int32 `Tensor` containing ordinals (i.e. number of days since 31
      Dec 0 with 1 being 1 Jan 1.)

  Returns:
    A `DateTensor` object representing the supplied dates.

  Raises:
    ValueError: If conversion fails for any reason.
  """
  if isinstance(date_inputs, DateTensor):
    return date_inputs

  if hasattr(date_inputs, "year"):  # Case 1.
    return from_datetimes(date_inputs)

  if isinstance(date_inputs, np.ndarray):  # Case 2.
    date_inputs = date_inputs.astype("datetime64[D]")
    return from_np_datetimes(date_inputs)

  if tf.is_tensor(date_inputs):  # Case 5
    return from_ordinals(date_inputs)

  if isinstance(date_inputs, collections.abc.Sequence):
    if not date_inputs:
      return from_ordinals([])
    test_element = date_inputs[0]
    if hasattr(test_element, "year"):  # Case 1.
      return from_datetimes(date_inputs)
    # Case 3
    if isinstance(test_element, collections.abc.Sequence):
      return from_tuples(date_inputs)
    if len(date_inputs) == 3:  # Case 4.
      return from_year_month_day(date_inputs[0], date_inputs[1], date_inputs[2])
  # As a last ditch effort, try to convert the sequence to a Tensor to see if
  # that can work
  try:
    as_ordinals = tf.convert_to_tensor(date_inputs, dtype=tf.int32)
    return from_ordinals(as_ordinals)
  except ValueError as e:
    raise ValueError("Failed to convert inputs to DateTensor. "
                     "Unrecognized format. Error: " + e)


def from_datetimes(datetimes):
  """Creates DateTensor from a sequence of Python datetime objects.

  Args:
    datetimes: Sequence of Python datetime objects.

  Returns:
    DateTensor object.

  #### Example

  ```python
  import datetime

  dates = [datetime.date(2015, 4, 15), datetime.date(2017, 12, 30)]
  date_tensor = tff.datetime.dates_from_datetimes(dates)
  ```
  """
  if isinstance(datetimes, (datetime.date, datetime.datetime)):
    return from_year_month_day(datetimes.year, datetimes.month, datetimes.day,
                               validate=False)
  years = tf.constant([dt.year for dt in datetimes], dtype=tf.int32)
  months = tf.constant([dt.month for dt in datetimes], dtype=tf.int32)
  days = tf.constant([dt.day for dt in datetimes], dtype=tf.int32)

  # datetime stores year, month and day internally, and datetime.toordinal()
  # performs calculations. We use a tf routine to perform these calculations
  # instead.
  return from_year_month_day(years, months, days, validate=False)


def from_np_datetimes(np_datetimes):
  """Creates DateTensor from a Numpy array of dtype datetime64.

  Args:
    np_datetimes: Numpy array of dtype datetime64.

  Returns:
    DateTensor object.

  #### Example

  ```python
  import datetime
  import numpy as np

  date_tensor_np = np.array(
    [[datetime.date(2019, 3, 25), datetime.date(2020, 6, 2)],
     [datetime.date(2020, 9, 15), datetime.date(2020, 12, 27)]],
     dtype=np.datetime64)

  date_tensor = tff.datetime.dates_from_np_datetimes(date_tensor_np)
  ```
  """

  # There's no easy way to extract year, month, day from numpy datetime, so
  # we start with ordinals.
  ordinals = tf.constant(np_datetimes, dtype=tf.int32) + _ORDINAL_OF_1_1_1970
  return from_ordinals(ordinals, validate=False)


def from_tuples(year_month_day_tuples, validate=True):
  """Creates DateTensor from a sequence of year-month-day Tuples.

  Args:
    year_month_day_tuples: Sequence of (year, month, day) Tuples. Months are
      1-based; constants from Months enum can be used instead of ints. Days are
      also 1-based.
    validate: Whether to validate the dates.

  Returns:
    DateTensor object.

  #### Example

  ```python
  date_tensor = tff.datetime.dates_from_tuples([(2015, 4, 15), (2017, 12, 30)])
  ```
  """
  years, months, days = [], [], []
  for t in year_month_day_tuples:
    years.append(t[0])
    months.append(t[1])
    days.append(t[2])
  years = tf.constant(years, dtype=tf.int32)
  months = tf.constant(months, dtype=tf.int32)
  days = tf.constant(days, dtype=tf.int32)
  return from_year_month_day(years, months, days, validate)


def from_year_month_day(year, month, day, validate=True):
  """Creates DateTensor from tensors of years, months and days.

  Args:
    year: Tensor of int32 type. Elements should be positive.
    month: Tensor of int32 type of same shape as `year`. Elements should be in
      range `[1, 12]`.
    day: Tensor of int32 type of same shape as `year`. Elements should be in
      range `[1, 31]` and represent valid dates together with corresponding
      elements of `month` and `year` Tensors.
    validate: Whether to validate the dates.

  Returns:
    DateTensor object.

  #### Example

  ```python
  year = tf.constant([2015, 2017], dtype=tf.int32)
  month = tf.constant([4, 12], dtype=tf.int32)
  day = tf.constant([15, 30], dtype=tf.int32)
  date_tensor = tff.datetime.dates_from_year_month_day(year, month, day)
  ```
  """
  year = tf.convert_to_tensor(year, tf.int32)
  month = tf.convert_to_tensor(month, tf.int32)
  day = tf.convert_to_tensor(day, tf.int32)

  control_deps = []
  if validate:
    control_deps.append(
        tf.debugging.assert_positive(year, message="Year must be positive."))
    control_deps.append(
        tf.debugging.assert_greater_equal(
            month,
            constants.Month.JANUARY.value,
            message=f"Month must be >= {constants.Month.JANUARY.value}"))
    control_deps.append(
        tf.debugging.assert_less_equal(
            month,
            constants.Month.DECEMBER.value,
            message="Month must be <= {constants.Month.JANUARY.value}"))
    control_deps.append(
        tf.debugging.assert_positive(day, message="Day must be positive."))
    is_leap = date_utils.is_leap_year(year)
    days_in_months = tf.constant(_DAYS_IN_MONTHS_COMBINED, tf.int32)
    max_days = tf.gather(days_in_months,
                         month + 12 * tf.dtypes.cast(is_leap, np.int32))
    control_deps.append(
        tf.debugging.assert_less_equal(
            day, max_days, message="Invalid day-month pairing."))
    with tf.compat.v1.control_dependencies(control_deps):
      # Ensure years, months, days themselves are under control_deps.
      year = tf.identity(year)
      month = tf.identity(month)
      day = tf.identity(day)

  with tf.compat.v1.control_dependencies(control_deps):
    ordinal = date_utils.year_month_day_to_ordinal(year, month, day)
    return DateTensor(ordinal, year, month, day)


def from_ordinals(ordinals, validate=True):
  """Creates DateTensor from tensors of ordinals.

  Args:
    ordinals: Tensor of type int32. Each value is number of days since 1 Jan
      0001. 1 Jan 0001 has `ordinal=1`.
    validate: Whether to validate the dates.

  Returns:
    DateTensor object.

  #### Example

  ```python
  ordinals = tf.constant([
    735703,  # 2015-4-12
    736693   # 2017-12-30
  ], dtype=tf.int32)

  date_tensor = tff.datetime.dates_from_ordinals(ordinals)
  ```
  """
  ordinals = tf.convert_to_tensor(ordinals, dtype=tf.int32)

  control_deps = []
  if validate:
    control_deps.append(
        tf.debugging.assert_positive(
            ordinals, message="Ordinals must be positive."))
    with tf.compat.v1.control_dependencies(control_deps):
      ordinals = tf.identity(ordinals)

  with tf.compat.v1.control_dependencies(control_deps):
    years, months, days = date_utils.ordinal_to_year_month_day(ordinals)
    return DateTensor(ordinals, years, months, days)


def from_tensor(tensor, validate=True):
  """Creates DateTensor from a single tensor containing years, months and days.

  This function is complementary to DateTensor.to_tensor: given an int32 Tensor
  of shape (..., 3), creates a DateTensor. The three elements of the last
  dimension are years, months and days, in this order.

  Args:
    tensor: Tensor of type int32 and shape (..., 3).
    validate: Whether to validate the dates.

  Returns:
    DateTensor object.

  #### Example

  ```python
  tensor = tf.constant([[2015, 4, 15], [2017, 12, 30]], dtype=tf.int32)
  date_tensor = tff.datetime.dates_from_tensor(tensor)
  ```
  """
  tensor = tf.convert_to_tensor(tensor, dtype=tf.int32)
  return from_year_month_day(
      year=tensor[..., 0],
      month=tensor[..., 1],
      day=tensor[..., 2],
      validate=validate)


def random_dates(*, start_date, end_date, size=1, seed=None, name=None):
  """Generates random dates between the supplied start and end dates.

  Generates specified number of random dates between the given start and end
  dates. The start and end dates are supplied as `DateTensor` objects. The dates
  uniformly distributed between the start date (inclusive) and end date
  (exclusive). Note that the dates are uniformly distributed over the calendar
  range, i.e. no holiday calendar is taken into account.

  Args:
    start_date: DateTensor of arbitrary shape. The start dates of the range from
      which to sample. The start dates are themselves included in the range.
    end_date: DateTensor of shape compatible with the `start_date`. The end date
      of the range from which to sample. The end dates are excluded from the
      range.
    size: Positive scalar int32 Tensor. The number of dates to draw between the
      start and end date.
      Default value: 1.
    seed: Optional seed for the random generation.
    name: Optional str. The name to give to the ops created by this function.
      Default value: 'random_dates'.

  Returns:
    A DateTensor of shape [size] + dates_shape where dates_shape is the common
    broadcast shape for (start_date, end_date).

  #### Example

  ```python
  # Note that the start and end dates need to be of broadcastable shape (though
  # not necessarily the same shape).
  # In this example, the start dates are of shape [2] and the end dates are
  # of a compatible but non-identical shape [1].
  start_dates = tff.datetime.dates_from_tuples([
    (2020, 5, 16),
    (2020, 6, 13)
  ])
  end_dates = tff.datetime.dates_from_tuples([(2021, 5, 21)])
  size = 3  # Generate 3 dates for each pair of (start, end date).
  sample = tff.datetime.random_dates(start_date=start_dates, end_date=end_dates,
                              size=size)
  # sample is a DateTensor of shape [3, 2]. The [3] is from the size and [2] is
  # the common broadcast shape of start and end date.
  ```
  """
  with tf.name_scope(name or "random_dates"):
    size = tf.reshape(
        tf.convert_to_tensor(size, dtype=tf.int32, name="size"), [-1])
    start_date = convert_to_date_tensor(start_date)
    end_date = convert_to_date_tensor(end_date)
    # Note that tf.random.uniform cannot deal with non scalar max value with
    # int dtypes. So we do this in float64 space and then floor. This incurs
    # some non-uniformity of the distribution but for practical purposes this
    # will be negligible.
    ordinal_range = tf.cast(
        end_date.ordinal() - start_date.ordinal(), dtype=tf.float64)
    sample_shape = tf.concat((size, tf.shape(ordinal_range)), axis=0)
    ordinal_sample = tf.cast(
        tf.random.uniform(
            sample_shape,
            maxval=ordinal_range,
            seed=seed,
            name="ordinal_sample",
            dtype=tf.float64),
        dtype=tf.int32)
    return from_ordinals(start_date.ordinal() + ordinal_sample, validate=False)
