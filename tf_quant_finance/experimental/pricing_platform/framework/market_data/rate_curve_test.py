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
"""Tests for rate_curve.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

rate_curve = tff.experimental.pricing_platform.framework.market_data.rate_curve
dateslib = tff.datetime
core = tff.experimental.pricing_platform.framework.core
InterpolationMethod = core.interpolation_method.InterpolationMethod


# This function can't be moved to SetUp since that would break graph mode
# execution
def build_cuve(
    interpolation_method=InterpolationMethod.CUBIC):
  valuation_date = dateslib.convert_to_date_tensor([(2020, 6, 15)])
  curve_dates = valuation_date + dateslib.periods.years([0, 1, 2])
  curve_disounts = [1.0, 0.95, 0.9]
  return rate_curve.RateCurve(
      curve_dates, curve_disounts, valuation_date,
      interpolator=interpolation_method, dtype=tf.float64)


@test_util.run_all_in_graph_and_eager_modes
class RateCurveTest(tf.test.TestCase, parameterized.TestCase):

  def test_rate_curve(self):
    curve = build_cuve()
    values = self.evaluate(
        curve.discount_rate(interpolation_dates=
                            [(2020, 6, 16), (2021, 6, 1), (2025, 1, 1)]))
    np.testing.assert_allclose(values,
                               [0.00017471, 0.05022863, 0.05268026], atol=1e-6)

  def test_discount_factor(self):
    curve = build_cuve()
    values = self.evaluate(
        curve.discount_factor(interpolation_dates=
                              [(2020, 6, 16), (2021, 6, 1), (2025, 1, 1)]))
    np.testing.assert_allclose(values,
                               [0.99999952, 0.95284594, 0.78683929], atol=1e-6)

  def test_rate_curve_time(self):
    curve = build_cuve()
    interpolation_times = curve._get_time(
        [(2020, 6, 16), (2021, 6, 1), (2025, 1, 1)])
    values = self.evaluate(
        curve.discount_rate(interpolation_times=interpolation_times))
    np.testing.assert_allclose(values,
                               [0.00017471, 0.05022863, 0.05268026], atol=1e-6)

  def test_discount_factor_time(self):
    curve = build_cuve()
    interpolation_times = curve._get_time(
        [(2020, 6, 16), (2021, 6, 1), (2025, 1, 1)])
    values = self.evaluate(
        curve.discount_factor(interpolation_times=interpolation_times))
    np.testing.assert_allclose(values,
                               [0.99999952, 0.95284594, 0.78683929], atol=1e-6)

  def test_fwd_rates(self):
    curve = build_cuve()
    start_dates = [(2020, 6, 16), (2021, 6, 1), (2025, 1, 1)]
    maturity_dates = [(2020, 7, 1), (2021, 8, 1), (2025, 3, 1)]
    day_count_fraction = dateslib.daycount_actual_actual_isda(
        start_date=start_dates, end_date=maturity_dates,
        dtype=tf.float64)
    values = self.evaluate(
        curve.forward_rate(
            start_date=start_dates,
            maturity_date=maturity_dates,
            day_count_fraction=day_count_fraction))
    np.testing.assert_allclose(values,
                               [0.0029773, 0.07680459, 0.05290519], atol=1e-6)

  def test_constant_fwd_interpolation(self):
    curve = build_cuve(
        interpolation_method=InterpolationMethod.CONSTANT_FORWARD)
    start_dates = [(2020, 7, 18), (2020, 8, 18), (2020, 8, 18), (2025, 1, 1)]
    maturity_dates = [(2020, 7, 19), (2020, 8, 19),
                      (2020, 12, 18), (2025, 3, 1)]
    values = self.evaluate(
        curve.forward_rate(
            start_date=start_dates,
            maturity_date=maturity_dates))
    np.testing.assert_allclose(
        values,
        [0.0512969, 0.0512969, 0.05173552, 0.05290519], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
