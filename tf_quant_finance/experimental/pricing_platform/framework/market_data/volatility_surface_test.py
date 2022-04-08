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
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

volatility_surface = tff.experimental.pricing_platform.framework.market_data.volatility_surface
dateslib = tff.datetime
core = tff.experimental.pricing_platform.framework.core
InterpolationMethod = core.interpolation_method.InterpolationMethod


# This function can't be moved to SetUp since that would break graph mode
# execution
def build_surface(dim, default_interp=True):
  dtype = tf.float64
  year = dim * [[2021, 2022, 2023, 2025, 2050]]
  month = dim * [[2, 2, 2, 2, 2]]
  day = dim * [[8, 8, 8, 8, 8]]
  expiries = tff.datetime.dates_from_year_month_day(year, month, day)
  valuation_date = [(2020, 6, 24)]
  strikes = dim * [[[1500, 1550, 1510],
                    [1500, 1550, 1510],
                    [1500, 1550, 1510],
                    [1500, 1550, 1510],
                    [1500, 1550, 1510]]]
  volatilities = dim * [[[0.1, 0.12, 0.13],
                         [0.15, 0.2, 0.15],
                         [0.1, 0.2, 0.1],
                         [0.1, 0.2, 0.1],
                         [0.1, 0.1, 0.3]]]
  interpolator = None
  if not default_interp:
    expiry_times = tf.cast(
        tff.datetime.convert_to_date_tensor(
            valuation_date).days_until(expiries), dtype=dtype) / 365.0
    interpolator_obj = tff.math.interpolation.interpolation_2d.Interpolation2D(
        expiry_times, tf.convert_to_tensor(strikes, dtype=dtype),
        volatilities)
    interpolator = interpolator_obj.interpolate

  return volatility_surface.VolatilitySurface(
      valuation_date, expiries, strikes, volatilities,
      interpolator=interpolator, dtype=dtype)


@test_util.run_all_in_graph_and_eager_modes
class VolatilitySurfaceTest(tf.test.TestCase, parameterized.TestCase):

  def test_volatility_1d(self):
    vol_surface = build_surface(1)
    expiry = tff.datetime.dates_from_tuples(
        [(2020, 6, 16), (2021, 6, 1), (2025, 1, 1)])
    vols = vol_surface.volatility(
        strike=[[1525, 1400, 1570]], expiry_dates=expiry.expand_dims(axis=0))
    self.assertAllClose(
        self.evaluate(vols),
        [[0.14046875, 0.11547945, 0.1]], atol=1e-6)

  def test_volatility_2d(self):
    vol_surface = build_surface(2)
    expiry = tff.datetime.dates_from_ordinals(
        [[737592, 737942, 739252],
         [737592, 737942, 739252]])
    vols = vol_surface.volatility(
        strike=[[1525, 1400, 1570], [1525, 1505, 1570]], expiry_dates=expiry)
    self.assertAllClose(
        self.evaluate(vols),
        [[0.14046875, 0.11547945, 0.1],
         [0.14046875, 0.12300392, 0.1]], atol=1e-6)

  def test_volatility_2d_interpolation(self):
    """Test using externally specified interpolator."""
    vol_surface = build_surface(2, False)
    expiry = tff.datetime.dates_from_ordinals(
        [[737592, 737942, 739252],
         [737592, 737942, 739252]])
    vols = vol_surface.volatility(
        strike=[[1525, 1400, 1570], [1525, 1505, 1570]], expiry_dates=expiry)
    self.assertAllClose(
        self.evaluate(vols),
        [[0.14046875, 0.11547945, 0.1],
         [0.14046875, 0.12300392, 0.1]], atol=1e-6)

  def test_volatility_2d_floats(self):
    vol_surface = build_surface(2)
    expiry = tff.datetime.dates_from_ordinals(
        [[737592, 737942, 739252],
         [737592, 737942, 739252]])
    valuation_date = tff.datetime.convert_to_date_tensor([(2020, 6, 24)])
    expiries = tf.cast(valuation_date.days_until(expiry),
                       dtype=vol_surface._dtype) / 365.0
    vols = vol_surface.volatility(
        strike=[[1525, 1400, 1570], [1525, 1505, 1570]],
        expiry_times=expiries)
    self.assertAllClose(
        self.evaluate(vols),
        [[0.14046875, 0.11547945, 0.1],
         [0.14046875, 0.12300392, 0.1]], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
