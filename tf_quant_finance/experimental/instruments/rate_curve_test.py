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
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
def get_curve(dtype, ext_discount):
  valuation_date = dates.convert_to_date_tensor([(2020, 1, 1)])

  curve_dates = valuation_date + dates.periods.years([0, 1, 2])
  curve_rates = np.array([0.0, 0.01, 0.02], dtype=np.float64)

  def my_discount_function(idates):
    idates = dates.convert_to_date_tensor(idates)
    curve_times = dates.daycount_actual_365_fixed(
        start_date=valuation_date, end_date=curve_dates, dtype=dtype)
    itimes = dates.daycount_actual_365_fixed(
        start_date=valuation_date, end_date=idates, dtype=dtype)
    irates = tff.math.interpolation.linear.interpolate(
        itimes, curve_times, curve_rates, dtype=dtype)
    return tf.math.exp(-irates * itimes)

  if ext_discount:
    fn = my_discount_function
    return instruments.ratecurve_from_discounting_function(fn, dtype)
  else:
    curve = instruments.RateCurve(
        curve_dates,
        curve_rates,
        valuation_date=valuation_date,
        dtype=np.float64)
    return curve


class RateCurveTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64, False),
  )
  def test_rate_curve(self, dtype, ext_discount):
    curve = get_curve(dtype, ext_discount)
    values = self.evaluate(
        curve.get_rates([(2020, 6, 1), (2021, 6, 1), (2025, 1, 1)]))
    np.testing.assert_allclose(values, [0.0041530054644809, 0.0141369863013699,
                                        0.02], atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64, False),
      ('DoublePrecisionExternalDiscount', np.float64, True),
  )
  def test_discount_factor(self, dtype, ext_discount):
    curve = get_curve(dtype, ext_discount)
    values = self.evaluate(
        curve.get_discount_factor([(2020, 6, 1), (2021, 6, 1), (2025, 1, 1)]))
    np.testing.assert_allclose(values, [0.9982720239040115, 0.9801749825461190,
                                        0.9047382632042100], atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64, False),
      ('DoublePrecisionExternalDiscount', np.float64, True),
  )
  def test_fwd_rates(self, dtype, ext_discount):
    start_dates = [(2020, 6, 1), (2021, 6, 1), (2025, 1, 1)]
    maturity_dates = [(2020, 7, 1), (2021, 8, 1), (2025, 3, 1)]
    curve = get_curve(dtype, ext_discount)
    values = self.evaluate(
        curve.get_forward_rate(start_dates, maturity_dates))
    np.testing.assert_allclose(values, [0.0091291063032444, 0.0300477964192536,
                                        0.0200323636336040], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
