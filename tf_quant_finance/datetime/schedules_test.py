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
"""Tests for schedules.py."""

import datetime

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tf_quant_finance.datetime import test_data
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

dates = tff.datetime


@test_util.run_all_in_graph_and_eager_modes
class SchedulesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      *test_data.periodic_schedule_test_cases)
  def test_periodic_schedule(self, start_dates, end_dates, period_quantities,
                             period_type, backward, expected_schedule,
                             end_of_month=False):
    start_dates = dates.dates_from_np_datetimes(_to_np_datetimes(start_dates))
    end_dates = dates.dates_from_np_datetimes(_to_np_datetimes(end_dates))
    tenors = dates.PeriodTensor(period_quantities, period_type)
    expected_schedule = dates.dates_from_np_datetimes(
        _to_np_datetimes(expected_schedule))
    actual_schedule = dates.PeriodicSchedule(
        start_date=start_dates,
        end_date=end_dates,
        tenor=tenors,
        holiday_calendar=dates.create_holiday_calendar(
            weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
            start_year=2020,
            end_year=2028),
        roll_convention=dates.BusinessDayConvention.MODIFIED_FOLLOWING,
        backward=backward,
        end_of_month=end_of_month).dates()
    self.assertAllEqual(expected_schedule.ordinal(), actual_schedule.ordinal())

  @parameterized.named_parameters(
      *test_data.periodic_schedule_dynamic)
  def test_periodic_schedule_dynamic(
      self, start_dates, end_dates, period_quantities,
      period_type, backward, expected_schedule, end_of_month=False):
    start_dates = dates.dates_from_ordinals(
        [s + tf.compat.v1.placeholder_with_default(0, [])
         for s in start_dates])
    end_dates = dates.dates_from_ordinals(
        [s + tf.compat.v1.placeholder_with_default(0, [])
         for s in end_dates])
    tenors = dates.PeriodTensor(period_quantities, period_type)
    expected_schedule = dates.dates_from_np_datetimes(
        _to_np_datetimes(expected_schedule))
    actual_schedule = dates.PeriodicSchedule(
        start_date=start_dates,
        end_date=end_dates,
        tenor=tenors,
        holiday_calendar=dates.create_holiday_calendar(
            weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
            start_year=2020,
            end_year=2028),
        roll_convention=dates.BusinessDayConvention.MODIFIED_FOLLOWING,
        backward=backward,
        end_of_month=end_of_month).dates()
    self.assertAllEqual(expected_schedule.ordinal(), actual_schedule.ordinal())

  @parameterized.named_parameters(
      *test_data.periodic_schedule_dynamic)
  def test_periodic_schedule_dynamic_shape(
      self, start_dates, end_dates, period_quantities,
      period_type, backward, expected_schedule, end_of_month=False):
    # Use tf.function to postulate unknown shape of input tensors.
    @tf.function(input_signature=(
        tf.TensorSpec([None], tf.int32, name="start_date"),
        tf.TensorSpec([None], tf.int32, name="end_date"),
        tf.TensorSpec([None], tf.int32, name="period_quantities")),
                 autograph=False)
    def _schedule_uncknown_shape(start_dates, end_dates, period_quantities):
      start_dates = dates.dates_from_ordinals(start_dates)
      end_dates = dates.dates_from_ordinals(end_dates)
      tenors = dates.PeriodTensor(period_quantities, period_type)
      return dates.PeriodicSchedule(
          start_date=start_dates,
          end_date=end_dates,
          tenor=tenors,
          holiday_calendar=dates.create_holiday_calendar(
              weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
              start_year=2020,
              end_year=2028),
          roll_convention=dates.BusinessDayConvention.MODIFIED_FOLLOWING,
          backward=backward,
          end_of_month=end_of_month).dates().ordinal()
    actual_schedule = _schedule_uncknown_shape(
        start_dates, end_dates, [period_quantities])
    expected_schedule = dates.dates_from_np_datetimes(
        _to_np_datetimes(expected_schedule))
    self.assertAllEqual(expected_schedule.ordinal(), actual_schedule)

  @parameterized.named_parameters(*test_data.business_day_schedule_test_cases)
  def test_business_day_schedule(self, start_dates, end_dates, holidays,
                                 backward, expected_schedule):
    start_dates = dates.dates_from_np_datetimes(_to_np_datetimes(start_dates))
    end_dates = dates.dates_from_np_datetimes(_to_np_datetimes(end_dates))
    holiday_calendar = dates.create_holiday_calendar(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY,
        holidays=holidays,
        start_year=2020,
        end_year=2020)
    expected_schedule = dates.dates_from_np_datetimes(
        _to_np_datetimes(expected_schedule))
    actual_schedule = dates.BusinessDaySchedule(
        start_date=start_dates,
        end_date=end_dates,
        holiday_calendar=holiday_calendar,
        backward=backward).dates()
    self.assertAllEqual(expected_schedule.ordinal(), actual_schedule.ordinal())


def _to_np_datetimes(nested_date_tuples):

  def recursive_convert_to_datetimes(sequence):
    result = []
    for item in sequence:
      if isinstance(item, list):
        result.append(recursive_convert_to_datetimes(item))
      else:
        result.append(datetime.date(*item))
    return result

  return np.array(
      recursive_convert_to_datetimes(nested_date_tuples), dtype=np.datetime64)


if __name__ == "__main__":
  tf.test.main()
