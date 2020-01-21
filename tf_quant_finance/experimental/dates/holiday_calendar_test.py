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
"""Tests for holiday_calendar.py."""

import datetime

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_quant_finance.experimental import dates
from tf_quant_finance.experimental.dates import test_data
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


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
  })
  def test_providing_holidays(self, holidays):
    cal = dates.HolidayCalendar(holidays=holidays)
    date_tensor = dates.DateTensor.from_tuples([(2020, 1, 1), (2020, 5, 1),
                                                (2020, 12, 25), (2021, 3, 8),
                                                (2021, 1, 1)])
    self.assertAllEqual([False, True, False, True, False],
                        cal.is_business_day(date_tensor))

  def test_custom_weekend_mask(self):
    weekend_mask = [0, 0, 0, 0, 1, 0, 1]  # Work Saturdays instead of Fridays.
    cal = dates.HolidayCalendar(start_year=2020,
                                end_year=2021,
                                weekend_mask=weekend_mask)
    date_tensor = dates.DateTensor.from_tuples([(2020, 1, 2), (2020, 1, 3),
                                                (2020, 1, 4), (2020, 1, 5),
                                                (2020, 1, 6), (2020, 5, 1),
                                                (2020, 5, 2)])
    self.assertAllEqual([True, False, True, False, True, False, True],
                        cal.is_business_day(date_tensor))

  def test_holidays_intersect_with_weekends(self):
    holidays = [(2020, 1, 4)]  # Saturday.
    cal = dates.HolidayCalendar(holidays=holidays)
    date_tensor = dates.DateTensor.from_tuples([(2020, 1, 3), (2020, 1, 4),
                                                (2020, 1, 5), (2020, 1, 6)])
    self.assertAllEqual([True, False, False, True],
                        cal.is_business_day(date_tensor))

  def test_no_holidays_specified(self):
    cal = dates.HolidayCalendar(start_year=2020, end_year=2021)
    date_tensor = dates.DateTensor.from_tuples([(2020, 1, 3), (2020, 1, 4),
                                                (2021, 12, 24), (2021, 12, 25)])
    self.assertAllEqual([True, False, True, False],
                        cal.is_business_day(date_tensor))

  def test_skip_eager_reset(self):
    cal = dates.HolidayCalendar(start_year=2020, end_year=2021)
    cal.is_business_day(dates.DateTensor.from_tuples([]))  # Trigger caching.
    tf.compat.v1.reset_default_graph()
    cal.reset()
    date_tensor = dates.DateTensor.from_tuples([(2020, 1, 3), (2020, 1, 4),
                                                (2021, 12, 24), (2021, 12, 25)])
    self.assertAllEqual([True, False, True, False],
                        cal.is_business_day(date_tensor))

  @parameterized.named_parameters(*rolling_test_parameters)
  def test_roll_to_business_days(self, rolling_enum_value, data_key):
    data = test_data.adjusted_dates_data
    date_tensor = dates.DateTensor.from_tuples([item["date"] for item in data])
    expected_dates = dates.DateTensor.from_tuples(
        [item[data_key] for item in data])

    cal = dates.HolidayCalendar(holidays=test_data.holidays)
    actual_dates = cal.roll_to_business_day(date_tensor,
                                            roll_convention=rolling_enum_value)
    self.assertAllEqual(expected_dates.ordinals(), actual_dates.ordinals())

  @parameterized.named_parameters(*rolling_test_parameters)
  def test_add_months_and_roll(self, rolling_enum_value, data_key):
    data = test_data.add_months_data
    date_tensor = dates.DateTensor.from_tuples([item["date"] for item in data])
    periods = dates.PeriodTensor([item["months"] for item in data],
                                 dates.PeriodType.MONTH)
    expected_dates = dates.DateTensor.from_tuples(
        [item[data_key] for item in data])
    cal = dates.HolidayCalendar(holidays=test_data.holidays)
    actual_dates = cal.add_period_and_roll(date_tensor, periods,
                                           roll_convention=rolling_enum_value)
    self.assertAllEqual(expected_dates.ordinals(), actual_dates.ordinals())

  def test_add_business_days(self):
    data = test_data.add_days_data
    date_tensor = dates.DateTensor.from_tuples([item["date"] for item in data])
    days = tf.constant([item["days"] for item in data])
    expected_dates = dates.DateTensor.from_tuples(
        [item["shifted_date"] for item in data])
    cal = dates.HolidayCalendar(holidays=test_data.holidays)
    actual_dates = cal.add_business_days(
        date_tensor, days,
        roll_convention=dates.BusinessDayConvention.MODIFIED_FOLLOWING)
    self.assertAllEqual(expected_dates.ordinals(), actual_dates.ordinals())

  def test_add_business_days_raises_on_invalid_input(self):
    data = test_data.add_days_data  # Contains some holidays.
    date_tensor = dates.DateTensor.from_tuples([item["date"] for item in data])
    days = tf.constant([item["days"] for item in data])
    cal = dates.HolidayCalendar(holidays=test_data.holidays)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      new_dates = cal.add_business_days(
          date_tensor, days,
          roll_convention=dates.BusinessDayConvention.NONE)
      self.evaluate(new_dates.ordinals())

  def test_business_days_between(self):
    data = test_data.days_between_data
    date_tensor1 = dates.DateTensor.from_tuples(
        [item["date1"] for item in data])
    date_tensor2 = dates.DateTensor.from_tuples(
        [item["date2"] for item in data])
    expected_days_between = [item["days"] for item in data]
    cal = dates.HolidayCalendar(holidays=test_data.holidays)
    actual_days_between = cal.business_days_between(date_tensor1, date_tensor2)
    self.assertAllEqual(expected_days_between, actual_days_between)

  def test_is_business_day(self):
    data = test_data.is_business_day_data
    date_tensor = dates.DateTensor.from_tuples([item[0] for item in data])
    expected = [item[1] for item in data]
    cal = dates.HolidayCalendar(holidays=test_data.holidays)
    actual = cal.is_business_day(date_tensor)
    self.assertEqual(tf.bool, actual.dtype)
    self.assertAllEqual(expected, actual)

if __name__ == "__main__":
  tf.test.main()
