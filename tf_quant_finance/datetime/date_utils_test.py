# Copyright 2019 Google LLC
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
"""Tests for date_utils.py."""

import datetime
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tf_quant_finance.datetime import test_data
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

dates = tff.datetime
date_utils = tff.datetime.utils


@test_util.run_all_in_graph_and_eager_modes
class DateUtilsTest(tf.test.TestCase):

  def test_ordinal_to_year_month_day(self):
    date_tuples = test_data.test_dates
    ordinals = np.array(
        [datetime.date(y, m, d).toordinal() for y, m, d in date_tuples],
        dtype=np.int32)
    y, m, d = date_utils.ordinal_to_year_month_day(ordinals)
    result = tf.stack((y, m, d), axis=1)
    self.assertAllEqual(date_tuples, result)

  def test_year_month_day_to_ordinal(self):
    date_tuples = test_data.test_dates
    expected = np.array(
        [datetime.date(y, m, d).toordinal() for y, m, d in date_tuples],
        dtype=np.int32)
    dates_np = np.array(date_tuples)
    years, months, days = dates_np[:, 0], dates_np[:, 1], dates_np[:, 2]
    actual = date_utils.year_month_day_to_ordinal(years, months, days)
    self.assertAllEqual(expected, actual)

  def test_is_leap_year(self):
    years = np.array([
        1600, 1700, 1800, 1900, 1901, 1903, 1904, 1953, 2000, 2020, 2025, 2100
    ])
    expected = np.array([
        True, False, False, False, False, False, True, False, True, True, False,
        False
    ])
    self.assertAllEqual(
        expected, date_utils.is_leap_year(years))

  def test_days_in_leap_years_between(self):
    test_cases = test_data.days_in_leap_years_test_cases
    date_tuples1, date_tuples2, expected_num_days = [], [], []
    for case in test_cases:
      date_tuples1.append(case["date1"])
      date_tuples2.append(case["date2"])
      expected_num_days.append(case["expected"])
    dates1 = dates.dates_from_tuples(date_tuples1)
    dates2 = dates.dates_from_tuples(date_tuples2)
    actual_num_days = date_utils.days_in_leap_years_between(dates1, dates2)
    self.assertAllEqual(expected_num_days, actual_num_days)

  def test_leap_days_between(self):
    test_cases = test_data.leap_days_between_dates_test_cases
    date_tuples1, date_tuples2, expected_num_days = [], [], []
    for case in test_cases:
      date_tuples1.append(case["date1"])
      date_tuples2.append(case["date2"])
      expected_num_days.append(case["expected"])
    dates1 = dates.dates_from_tuples(date_tuples1)
    dates2 = dates.dates_from_tuples(date_tuples2)
    actual_num_days = date_utils.leap_days_between(dates1, dates2)
    self.assertAllEqual(expected_num_days, actual_num_days)


if __name__ == "__main__":
  tf.test.main()
