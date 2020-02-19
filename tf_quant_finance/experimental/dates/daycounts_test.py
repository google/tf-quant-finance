# Lint as: python3
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
"""Tests for daycounts.py."""

import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

dateslib = tff.experimental.dates


@test_util.run_all_in_graph_and_eager_modes
class DayCountsTest(tf.test.TestCase):

  def test_actual_360(self):
    start_date = dateslib.from_tuples([(2019, 12, 17)])
    end_date = dateslib.from_tuples([(2021, 2, 11)])
    yf = self.evaluate(
        dateslib.daycounts.actual_360(start_date=start_date, end_date=end_date))
    self.assertAllClose(yf, [1.17222], atol=1e-5)

  def test_actual_365_fixed(self):
    start_date = dateslib.from_tuples([(2019, 12, 17)])
    end_date = dateslib.from_tuples([(2021, 2, 11)])
    yf = self.evaluate(
        dateslib.daycounts.actual_365_fixed(
            start_date=start_date, end_date=end_date))
    self.assertAllClose(yf, [1.156164], atol=1e-5)

  def test_actual_365_actual_with_leap_day(self):
    start_date = dateslib.from_tuples([(2019, 12, 17)])
    end_date = dateslib.from_tuples([(2021, 2, 11)])
    yf = self.evaluate(
        dateslib.daycounts.actual_365_actual(
            start_date=start_date, end_date=end_date))
    self.assertAllClose(yf, [1.153005], atol=1e-5)

  def test_actual_365_actual_no_leap_day(self):
    start_date = dateslib.from_tuples([(2020, 3, 17)])
    end_date = dateslib.from_tuples([(2021, 2, 17)])
    yf = self.evaluate(
        dateslib.daycounts.actual_365_actual(
            start_date=start_date, end_date=end_date))
    self.assertAllClose(yf, [0.923288], atol=1e-5)

  def test_actual_365_actual_with_leap_day_on_start(self):
    start_date = dateslib.from_tuples([(2020, 2, 29)])
    end_date = dateslib.from_tuples([(2021, 2, 11)])
    yf = self.evaluate(
        dateslib.daycounts.actual_365_actual(
            start_date=start_date, end_date=end_date))
    self.assertAllClose(yf, [0.953424], atol=1e-5)

  def test_actual_365_actual_with_leap_day_on_end(self):
    start_date = dateslib.from_tuples([(2019, 2, 11)])
    end_date = dateslib.from_tuples([(2020, 2, 29)])
    yf = self.evaluate(
        dateslib.daycounts.actual_365_actual(
            start_date=start_date, end_date=end_date))
    self.assertAllClose(yf, [1.046448], atol=1e-5)


if __name__ == '__main__':
  tf.test.main()
