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
"""Tests for overnight_index_linked_futures.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class OvernightIndexLinkedFuturesTest(tf.test.TestCase,
                                      parameterized.TestCase):

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fut_compounded(self, dtype):
    cal = dates.create_holiday_calendar(weekend_mask=dates.WeekendMask.NONE)

    start_date = dates.convert_to_date_tensor([(2020, 5, 1)])
    end_date = dates.convert_to_date_tensor([(2020, 5, 31)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    indexfuture = instruments.OvernightIndexLinkedFutures(
        start_date,
        end_date,
        holiday_calendar=cal,
        averaging_type=instruments.AverageType.COMPOUNDING,
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 6])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=None)

    price = self.evaluate(indexfuture.price(valuation_date, market))
    np.testing.assert_allclose(price, 98.64101997, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fut_averaged(self, dtype):
    cal = dates.create_holiday_calendar(weekend_mask=dates.WeekendMask.NONE)

    start_date = dates.convert_to_date_tensor([(2020, 5, 1)])
    end_date = dates.convert_to_date_tensor([(2020, 5, 31)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    indexfuture = instruments.OvernightIndexLinkedFutures(
        start_date,
        end_date,
        averaging_type=instruments.AverageType.ARITHMETIC_AVERAGE,
        holiday_calendar=cal,
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 6])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=None)

    price = self.evaluate(indexfuture.price(valuation_date, market))
    np.testing.assert_allclose(price, 98.6417886, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fut_compounded_calendar(self, dtype):
    cal = dates.create_holiday_calendar(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY)

    start_date = dates.convert_to_date_tensor([(2020, 5, 1)])
    end_date = dates.convert_to_date_tensor([(2020, 5, 31)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    indexfuture = instruments.OvernightIndexLinkedFutures(
        start_date,
        end_date,
        holiday_calendar=cal,
        averaging_type=instruments.AverageType.COMPOUNDING,
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 6])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=None)

    price = self.evaluate(indexfuture.price(valuation_date, market))
    np.testing.assert_allclose(price, 98.6332129, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fut_averaged_calendar(self, dtype):
    cal = dates.create_holiday_calendar(
        weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY)

    start_date = dates.convert_to_date_tensor([(2020, 5, 1)])
    end_date = dates.convert_to_date_tensor([(2020, 5, 31)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    indexfuture = instruments.OvernightIndexLinkedFutures(
        start_date,
        end_date,
        averaging_type=instruments.AverageType.ARITHMETIC_AVERAGE,
        holiday_calendar=cal,
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 6])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=None)

    price = self.evaluate(indexfuture.price(valuation_date, market))
    np.testing.assert_allclose(price, 98.63396465, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fut_many(self, dtype):
    cal = dates.create_holiday_calendar(weekend_mask=dates.WeekendMask.NONE)

    start_date = dates.convert_to_date_tensor([(2020, 5, 1), (2020, 5, 1)])
    end_date = dates.convert_to_date_tensor([(2020, 5, 31), (2020, 5, 31)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    indexfuture = instruments.OvernightIndexLinkedFutures(
        start_date,
        end_date,
        holiday_calendar=cal,
        averaging_type=instruments.AverageType.COMPOUNDING,
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 6])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=None)

    price = self.evaluate(indexfuture.price(valuation_date, market))
    np.testing.assert_allclose(price, [98.64101997, 98.64101997], atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
