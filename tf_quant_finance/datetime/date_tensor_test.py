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
"""Tests for date_tensor.py."""

import datetime
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff

from tf_quant_finance.datetime import test_data
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

dateslib = tff.datetime


@test_util.run_all_in_graph_and_eager_modes
class DateTensorTest(tf.test.TestCase):

  def test_convert_to_date_tensor_tuples(self):
    inputs = [(2018, 5, 4), (2042, 11, 22), (1947, 8, 15)]
    date_tensor = dateslib.convert_to_date_tensor(inputs)
    y, m, d = zip(*inputs)
    self.assert_date_tensor_components(date_tensor, y, m, d, None)

  def test_convert_to_date_tensor_datetimes(self):
    date = datetime.date(2020, 6, 15)
    date_tensor = dateslib.convert_to_date_tensor(date)
    (y, m, d) = (2020, 6, 15)
    self.assert_date_tensor_components(date_tensor, y, m, d, None)

  def test_convert_to_date_tensor_list_of_datetimes(self):
    inputs = [
        datetime.date(2018, 5, 4),
        datetime.date(2042, 11, 22),
        datetime.date(1947, 8, 15)
    ]
    date_tensor = dateslib.convert_to_date_tensor(inputs)
    y, m, d = [2018, 2042, 1947], [5, 11, 8], [4, 22, 15]
    self.assert_date_tensor_components(date_tensor, y, m, d, None)

  def test_convert_to_date_tensor_ordinals(self):
    inputs = [1, 2, 3, 4, 5]
    inputs2 = tf.constant(inputs)
    date_tensor = dateslib.convert_to_date_tensor(inputs)
    date_tensor2 = dateslib.convert_to_date_tensor(inputs2)
    self.assert_date_tensor_components(date_tensor, [1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1], [1, 2, 3, 4, 5], inputs)

    self.assert_date_tensor_components(date_tensor2, [1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1], [1, 2, 3, 4, 5], inputs)

  def test_convert_to_date_tensor_tensor_tuples(self):
    inputs = [
        tf.constant([2018, 2042, 1947]),
        tf.constant([5, 11, 8]),
        tf.constant([4, 22, 15])
    ]
    date_tensor = dateslib.convert_to_date_tensor(inputs)
    y, m, d = [2018, 2042, 1947], [5, 11, 8], [4, 22, 15]
    self.assert_date_tensor_components(date_tensor, y, m, d, None)

  def test_convert_to_date_tensor_npdatetime(self):
    inputs = np.array([
        datetime.date(2018, 5, 4),
        datetime.date(2042, 11, 22),
        datetime.date(1947, 8, 15)
    ],
                      dtype='datetime64')
    date_tensor = dateslib.convert_to_date_tensor(inputs)
    y, m, d = [2018, 2042, 1947], [5, 11, 8], [4, 22, 15]
    self.assert_date_tensor_components(date_tensor, y, m, d, None)

  def test_create_from_date_time_list(self):
    dates = test_data.test_dates
    y, m, d, o, datetimes = unpack_test_dates(dates)
    date_tensor = dateslib.dates_from_datetimes(datetimes)
    self.assert_date_tensor_components(date_tensor, y, m, d, o)

  def test_create_from_date_time_scalar(self):
    test_date = datetime.date(2018, 5, 4)
    date_tensor = dateslib.dates_from_datetimes(test_date)
    self.assertEqual(self.evaluate(date_tensor.year()), 2018)
    self.assertEqual(self.evaluate(date_tensor.month()), 5)
    self.assertEqual(self.evaluate(date_tensor.day()), 4)

  def test_create_from_np_datetimes(self):
    dates = test_data.test_dates
    y, m, d, o, datetimes = unpack_test_dates(dates)
    np_datetimes = np.array(datetimes, dtype=np.datetime64)
    date_tensor = dateslib.dates_from_np_datetimes(np_datetimes)
    self.assert_date_tensor_components(date_tensor, y, m, d, o)

  def test_create_from_tuples(self):
    dates = test_data.test_dates
    y, m, d, o, _ = unpack_test_dates(dates)
    date_tensor = dateslib.dates_from_tuples(dates)
    self.assert_date_tensor_components(date_tensor, y, m, d, o)

  def test_create_from_year_month_day(self):
    dates = test_data.test_dates
    y, m, d, o, _ = unpack_test_dates(dates)
    date_tensor = dateslib.dates_from_year_month_day(y, m, d)
    self.assert_date_tensor_components(date_tensor, y, m, d, o)

  def test_create_from_ordinals(self):
    dates = test_data.test_dates
    y, m, d, o, _ = unpack_test_dates(dates)
    date_tensor = dateslib.dates_from_ordinals(o)
    self.assert_date_tensor_components(date_tensor, y, m, d, o)

  def test_to_and_from_tensor(self):
    dates = [[[2020, 1, 21], [2021, 2, 22], [2022, 3, 23]],
             [[2023, 4, 24], [2024, 5, 25], [2025, 6, 26]]]
    date_tensor = dateslib.dates_from_tensor(dates)

    with self.subTest('from_tensor'):
      self.assert_date_tensor_components(
          date_tensor,
          [[2020, 2021, 2022], [2023, 2024, 2025]],
          [[1, 2, 3], [4, 5, 6]],
          [[21, 22, 23], [24, 25, 26]])

    with self.subTest('to_tensor'):
      self.assertAllEqual(dates, date_tensor.to_tensor())

  def test_validation(self):
    not_raised = []
    for y, m, d in test_data.invalid_dates:
      try:
        self.evaluate(dateslib.dates_from_tuples([(y, m, d)]).month())
        not_raised.append((y, m, d))
      except tf.errors.InvalidArgumentError:
        pass
    self.assertEmpty(not_raised)

    for invalid_ordinal in [-5, 0]:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(dateslib.dates_from_ordinals([invalid_ordinal]).month())

  def test_day_of_week(self):
    dates = test_data.test_dates
    datetimes = unpack_test_dates(dates)[-1]
    date_tensor = dateslib.dates_from_datetimes(datetimes)
    expected_day_of_week = np.array([dt.weekday() for dt in datetimes])
    self.assertAllEqual(expected_day_of_week, date_tensor.day_of_week())

  def test_days_until(self):
    dates = test_data.test_dates
    diffs = np.arange(0, len(dates))
    _, _, _, o, datetimes = unpack_test_dates(dates)
    date_tensor = dateslib.dates_from_datetimes(datetimes)

    target_ordinals = o + diffs
    target_datetimes = [datetime.date.fromordinal(o) for o in target_ordinals]
    target_date_tensor = dateslib.dates_from_datetimes(target_datetimes)
    self.assertAllEqual(diffs, date_tensor.days_until(target_date_tensor))

  def test_days_addition(self):
    self.perform_addition_test(test_data.day_addition_data,
                               dateslib.PeriodType.DAY)

  def test_week_addition(self):
    self.perform_addition_test(test_data.week_addition_data,
                               dateslib.PeriodType.WEEK)

  def test_month_addition(self):
    self.perform_addition_test(test_data.month_addition_data,
                               dateslib.PeriodType.MONTH)

  def test_year_addition(self):
    self.perform_addition_test(test_data.year_addition_data,
                               dateslib.PeriodType.YEAR)

  def perform_addition_test(self, data, period_type):
    dates_from, quantities, expected_dates = [], [], []
    for date_from, quantity, expected_date in data:
      dates_from.append(date_from)
      quantities.append(quantity)
      expected_dates.append(expected_date)

    datetimes = unpack_test_dates(dates_from)[-1]
    date_tensor = dateslib.dates_from_datetimes(datetimes)
    period_tensor = dateslib.PeriodTensor(quantities, period_type)
    result_date_tensor = date_tensor + period_tensor

    y, m, d, o, _ = unpack_test_dates(expected_dates)
    self.assert_date_tensor_components(result_date_tensor, y, m, d, o)

  def test_date_subtraction(self):
    # Subtraction trivially transforms to addition, so we don't test
    # extensively.
    dates_from = dateslib.dates_from_tuples([(2020, 3, 15), (2020, 3, 31)])
    period = dateslib.PeriodTensor([2, 1], dateslib.PeriodType.MONTH)
    expected_ordinals = np.array([datetime.date(2020, 1, 15).toordinal(),
                                  datetime.date(2020, 2, 29).toordinal()])
    self.assertAllEqual(expected_ordinals, (dates_from - period).ordinal())

  def test_comparisons(self):
    dates1 = dateslib.dates_from_tuples(
        [(2020, 3, 15), (2020, 3, 31), (2021, 2, 28)])
    dates2 = dateslib.dates_from_tuples(
        [(2020, 3, 18), (2020, 3, 31), (2019, 2, 28)])
    self.assertAllEqual(np.array([False, True, False]), dates1 == dates2)
    self.assertAllEqual(np.array([True, False, True]), dates1 != dates2)
    self.assertAllEqual(np.array([False, False, True]), dates1 > dates2)
    self.assertAllEqual(np.array([False, True, True]), dates1 >= dates2)
    self.assertAllEqual(np.array([True, False, False]), dates1 < dates2)
    self.assertAllEqual(np.array([True, True, False]), dates1 <= dates2)

  def test_tensor_wrapper_ops(self):
    dates1 = dateslib.dates_from_tuples(
        [(2019, 3, 25), (2020, 1, 2), (2019, 1, 2)])
    dates2 = dateslib.dates_from_tuples(
        [(2019, 4, 25), (2020, 5, 2), (2018, 1, 2)])
    dates = dateslib.DateTensor.stack((dates1, dates2), axis=-1)
    self.assertEqual((3, 2), dates.shape)
    self.assertEqual((2,), dates[0].shape)
    self.assertEqual((2, 2), dates[1:].shape)
    self.assertEqual((2, 1), dates[1:, :-1].shape)
    self.assertEqual((3, 1, 2), dates.expand_dims(axis=1).shape)
    self.assertEqual((3, 3, 2), dates.broadcast_to((3, 3, 2)).shape)

  def test_boolean_mask(self):
    dates = dateslib.dates_from_tuples(
        [(2019, 3, 25), (2020, 1, 2), (2019, 1, 2)])
    mask = [True, False, True]
    expected = dateslib.DateTensor.stack((dates[0], dates[2]))
    self.assert_date_tensor_equals(expected, dates.boolean_mask(mask))

  def test_day_of_year(self):
    data = test_data.day_of_year_data
    date_tuples, expected_days_of_year = zip(*data)
    dates = dateslib.dates_from_tuples(date_tuples)
    self.assertAllEqual(expected_days_of_year, dates.day_of_year())

  def test_random_dates(self):
    start_dates = dateslib.dates_from_tuples([(2020, 5, 16), (2020, 6, 13)])
    end_dates = dateslib.dates_from_tuples([(2021, 5, 21)])
    size = 3  # Generate 3 dates for each pair of (start, end date).
    sample = dateslib.random_dates(
        start_date=start_dates, end_date=end_dates, size=size, seed=42)
    self.assertEqual(sample.shape, (3, 2))
    self.assertTrue(self.evaluate(tf.reduce_all(sample < end_dates)))
    self.assertTrue(self.evaluate(tf.reduce_all(sample >= start_dates)))

  def test_is_end_of_month(self):
    cases = test_data.end_of_month_test_cases
    dates = dateslib.dates_from_tuples([case[0] for case in cases])
    expected = tf.constant([case[1] for case in cases])
    self.assertAllEqual(expected, dates.is_end_of_month())

  def test_to_end_of_month(self):
    cases = test_data.end_of_month_test_cases
    dates = dateslib.dates_from_tuples([case[0] for case in cases])
    expected = dateslib.dates_from_tuples([case[2] for case in cases])
    self.assert_date_tensor_equals(expected, dates.to_end_of_month())

  def assert_date_tensor_equals(self, expected_date_tensor, actual_date_tensor):
    """Asserts given two DateTensors are equal."""
    self.assertAllEqual(expected_date_tensor.ordinal(),
                        actual_date_tensor.ordinal())

  def assert_date_tensor_components(self, date_tensor, expected_years_np,
                                    expected_months_np, expected_days_np,
                                    expected_ordinals_np=None):
    """Asserts given DateTensor has expected components."""
    self.assertAllEqual(expected_years_np, date_tensor.year())
    self.assertAllEqual(expected_months_np, date_tensor.month())
    self.assertAllEqual(expected_days_np, date_tensor.day())
    if expected_ordinals_np is not None:
      self.assertAllEqual(expected_ordinals_np, date_tensor.ordinal())


def unpack_test_dates(dates):
  y, m, d = (np.array([d[i] for d in dates], dtype=np.int32) for i in range(3))
  datetimes = [datetime.date(y, m, d) for y, m, d in dates]
  o = np.array([datetime.date(y, m, d).toordinal() for y, m, d in dates],
               dtype=np.int32)
  return y, m, d, o, datetimes

if __name__ == '__main__':
  tf.test.main()
