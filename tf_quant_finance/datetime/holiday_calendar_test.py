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
"""Tests for HolidayCalendar implementations."""

import datetime
import functools

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tf_quant_finance.datetime import bounded_holiday_calendar
from tf_quant_finance.datetime import test_data
from tf_quant_finance.datetime import unbounded_holiday_calendar
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

dates = tff.datetime


def test_both_impls(test_fn):
  # Decorator to run the test with both BoundedHolidayCalendar and
  # UnboundedHolidayCalendar.
  # Create the calendar as `self.impl(args)`.
  def create_unbounded_calendar(**kwargs):
    kwargs.pop("start_year", None)
    kwargs.pop("end_year", None)
    return unbounded_holiday_calendar.UnboundedHolidayCalendar(**kwargs)

  @functools.wraps(test_fn)
  def wrapped(*args, **kwargs):
    self = args[0]
    with self.subTest("Bounded"):
      self.impl = bounded_holiday_calendar.BoundedHolidayCalendar
      test_fn(*args, **kwargs)
    with self.subTest("Unbounded"):
      self.impl = create_unbounded_calendar
      test_fn(*args, **kwargs)
  return wrapped


@test_util.run_all_in_graph_and_eager_modes
class HolidayCalendarTest(tf.test.TestCase, parameterized.TestCase):

  rolling_test_parameters = {
      "testcase_name": "unadjusted",
      "rolling_enum_value": dates.BusinessDayConvention.NONE,
      "data_key": "unadjusted"
  }, {
      "testcase_name": "following",
      "rolling_enum_value": dates.BusinessDayConvention.FOLLOWING,
      "data_key": "following"  # See test_data.adjusted_dates_data
  }, {
      "testcase_name": "modified_following",
      "rolling_enum_value": dates.BusinessDayConvention.MODIFIED_FOLLOWING,
      "data_key": "modified_following"
  }, {
      "testcase_name": "preceding",
      "rolling_enum_value": dates.BusinessDayConvention.PRECEDING,
      "data_key": "preceding"
  }, {
      "testcase_name": "modified_preceding",
      "rolling_enum_value": dates.BusinessDayConvention.MODIFIED_PRECEDING,
      "data_key": "modified_preceding"
  }

  @parameterized.named_parameters({
      "testcase_name": "as_tuples",
      "holidays": [(2020, 1, 1), (2020, 12, 25), (2021, 1, 1)],
  }, {
      "testcase_name": "as_datetimes",
      "holidays": [datetime.date(2020, 1, 1),
                   datetime.date(2020, 12, 25),
                   datetime.date(2021, 1, 1)],
  }, {
      "testcase_name": "as_numpy_array",
      "holidays": np.array(["2020-01-01", "2020-12-25", "2021-01-01"],
                           dtype=np.datetime64),
  }, {
      "testcase_name": "as_date_tensors",
      "holidays": [(2020, 1, 1), (2020, 12, 25), (2021, 1, 1)],
      "convert_to_date_tensor": True,  # Can't do this in parameter definition.
  })
  @test_both_impls
  def test_providing_holidays(self, holidays, convert_to_date_tensor=False):
    if convert_to_date_tensor:
      holidays = dates.convert_to_date_tensor(holidays)
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY, holidays=holidays)
    date_tensor = dates.dates_from_tuples([(2020, 1, 1), (2020, 5, 1),
                                           (2020, 12, 25), (2021, 3, 8),
                                           (2021, 1, 1)])
    self.assertAllEqual([False, True, False, True, False],
                        cal.is_business_day(date_tensor))

  @test_both_impls
  def test_custom_weekend_mask(self):
    weekend_mask = [0, 0, 0, 0, 1, 0, 1]  # Work Saturdays instead of Fridays.
    cal = self.impl(start_year=2020, end_year=2021, weekend_mask=weekend_mask)
    date_tensor = dates.dates_from_tuples([(2020, 1, 2), (2020, 1, 3),
                                           (2020, 1, 4), (2020, 1, 5),
                                           (2020, 1, 6), (2020, 5, 1),
                                           (2020, 5, 2)])
    self.assertAllEqual([True, False, True, False, True, False, True],
                        cal.is_business_day(date_tensor))

  @test_both_impls
  def test_holidays_intersect_with_weekends(self):
    holidays = [(2020, 1, 4)]  # Saturday.
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY, holidays=holidays)
    date_tensor = dates.dates_from_tuples([(2020, 1, 3), (2020, 1, 4),
                                           (2020, 1, 5), (2020, 1, 6)])
    self.assertAllEqual([True, False, False, True],
                        cal.is_business_day(date_tensor))

  @test_both_impls
  def test_no_holidays_specified(self):
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        start_year=2020,
        end_year=2021)
    date_tensor = dates.dates_from_tuples([(2020, 1, 3), (2020, 1, 4),
                                           (2021, 12, 24), (2021, 12, 25)])
    self.assertAllEqual([True, False, True, False],
                        cal.is_business_day(date_tensor))

  def test_tf_function(self):
    # Check that tensor caching doesn't mess with tf.function.
    @tf.function
    def foo():
      cal = bounded_holiday_calendar.BoundedHolidayCalendar(
          weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
          holidays=test_data.holidays)
      date_tensor = dates.dates_from_tuples([(2020, 1, 3), (2020, 1, 4),
                                             (2021, 12, 24), (2021, 12, 25)])
      return cal.is_business_day(date_tensor)

    self.assertAllEqual([True, False, False, False], foo())

  @parameterized.named_parameters(*rolling_test_parameters)
  @test_both_impls
  def test_roll_to_business_days(self, rolling_enum_value, data_key):
    data = test_data.adjusted_dates_data
    date_tensor = dates.dates_from_tuples([item["date"] for item in data])
    expected_dates = dates.dates_from_tuples([item[data_key] for item in data])

    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=test_data.holidays)
    actual_dates = cal.roll_to_business_day(date_tensor,
                                            roll_convention=rolling_enum_value)
    self.assertAllEqual(expected_dates.ordinal(), actual_dates.ordinal())

  @parameterized.named_parameters(*rolling_test_parameters)
  @test_both_impls
  def test_add_months_and_roll(self, rolling_enum_value, data_key):
    data = test_data.add_months_data
    date_tensor = dates.dates_from_tuples([item["date"] for item in data])
    periods = dates.periods.months([item["months"] for item in data])
    expected_dates = dates.dates_from_tuples([item[data_key] for item in data])
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=test_data.holidays)
    actual_dates = cal.add_period_and_roll(date_tensor, periods,
                                           roll_convention=rolling_enum_value)
    self.assertAllEqual(expected_dates.ordinal(), actual_dates.ordinal())

  @test_both_impls
  def test_add_business_days(self):
    data = test_data.add_days_data
    date_tensor = dates.dates_from_tuples([item["date"] for item in data])
    days = tf.constant([item["days"] for item in data])
    expected_dates = dates.dates_from_tuples([item["shifted_date"]
                                              for item in data])
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=test_data.holidays)
    actual_dates = cal.add_business_days(
        date_tensor, days,
        roll_convention=dates.BusinessDayConvention.MODIFIED_FOLLOWING)
    self.assertAllEqual(expected_dates.ordinal(), actual_dates.ordinal())

  @test_both_impls
  def test_add_business_days_raises_on_invalid_input(self):
    data = test_data.add_days_data  # Contains some holidays.
    date_tensor = dates.dates_from_tuples([item["date"] for item in data])
    days = tf.constant([item["days"] for item in data])
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=test_data.holidays)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      new_dates = cal.add_business_days(
          date_tensor, days,
          roll_convention=dates.BusinessDayConvention.NONE)
      self.evaluate(new_dates.ordinal())

  @test_both_impls
  def test_business_days_between(self):
    data = test_data.days_between_data
    date_tensor1 = dates.dates_from_tuples([item["date1"] for item in data])
    date_tensor2 = dates.dates_from_tuples([item["date2"] for item in data])
    expected_days_between = [item["days"] for item in data]
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=test_data.holidays)
    actual_days_between = cal.business_days_between(date_tensor1, date_tensor2)
    self.assertAllEqual(expected_days_between, actual_days_between)

  @test_both_impls
  def test_is_business_day(self):
    data = test_data.is_business_day_data
    date_tensor = dates.dates_from_tuples([item[0] for item in data])
    expected = [item[1] for item in data]
    cal = self.impl(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=test_data.holidays)
    actual = cal.is_business_day(date_tensor)
    self.assertEqual(tf.bool, actual.dtype)
    self.assertAllEqual(expected, actual)

  def test_bounded_impl_near_boundaries(self):
    cal = bounded_holiday_calendar.BoundedHolidayCalendar(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        start_year=2017,
        end_year=2022)

    def assert_roll_raises(roll_convention, date):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(cal.roll_to_business_day(date, roll_convention).year())

    def assert_rolls_to(roll_convention, date, expected_date):
      rolled = cal.roll_to_business_day(date, roll_convention)
      self.assertAllEqual(rolled.ordinal(), expected_date.ordinal())

    date = dates.dates_from_tuples([(2022, 12, 31)])  # Saturday
    preceding = dates.dates_from_tuples([(2022, 12, 30)])
    with self.subTest("following_upper_bound"):
      assert_roll_raises(dates.BusinessDayConvention.FOLLOWING, date)
    with self.subTest("preceding_upper_bound"):
      assert_rolls_to(dates.BusinessDayConvention.PRECEDING, date, preceding)
    with self.subTest("modified_following_upper_bound"):
      assert_rolls_to(dates.BusinessDayConvention.MODIFIED_FOLLOWING,
                      date, preceding)
    with self.subTest("modified_preceding_upper_bound"):
      assert_rolls_to(dates.BusinessDayConvention.MODIFIED_PRECEDING,
                      date, preceding)

    date = dates.dates_from_tuples([(2017, 1, 1)])  # Sunday
    following = dates.dates_from_tuples([(2017, 1, 2)])

    with self.subTest("following_lower_bound"):
      assert_rolls_to(dates.BusinessDayConvention.FOLLOWING, date, following)
    with self.subTest("preceding_lower_bound"):
      assert_roll_raises(dates.BusinessDayConvention.PRECEDING, date)
    with self.subTest("modified_following_lower_bound"):
      assert_rolls_to(dates.BusinessDayConvention.MODIFIED_FOLLOWING,
                      date, following)
    with self.subTest("modified_preceding_lower_bound"):
      assert_rolls_to(dates.BusinessDayConvention.MODIFIED_PRECEDING,
                      date, following)

if __name__ == "__main__":
  tf.test.main()
