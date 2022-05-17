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
"""Tests for cap_floor.py."""

from absl.testing import parameterized

import numpy as np
from numpy import testing as np_testing
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class CapFloorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(CapFloorTest, self).setUp()
    self.maturity_date = [(2022, 1, 15)]
    self.start_date = [(2021, 1, 15)]
    self.valuation_date = [(2021, 1, 1)]

  def get_market(self):
    val_date = dates.convert_to_date_tensor(self.valuation_date)
    curve_dates = val_date + dates.periods.months([0, 3, 12, 24])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.005, 0.01, 0.015, 0.02], dtype=np.float64),
        valuation_date=val_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    return market

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cap_correctness(self, dtype):
    notional = 100.0

    period3m = dates.periods.months(3)

    cap = instruments.CapAndFloor(
        self.start_date,
        self.maturity_date,
        period3m,
        0.005,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        notional=notional,
        dtype=dtype)

    price = self.evaluate(
        cap.price(
            self.valuation_date,
            self.get_market(),
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=0.5))
    np_testing.assert_allclose(price, 1.0474063612452953, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_floor_correctness(self, dtype):
    notional = 100.0
    period3m = dates.periods.months(3)
    cap = instruments.CapAndFloor(
        self.start_date,
        self.maturity_date,
        period3m,
        0.01,  # since this is a floor, we use different strike
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        notional=notional,
        is_cap=False,
        dtype=dtype)
    price = self.evaluate(
        cap.price(
            self.valuation_date,
            self.get_market(),
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=0.5))
    np_testing.assert_allclose(price, 0.01382758837128641, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cap_many(self, dtype):
    notional = 100.
    batch_maturity_date = dates.convert_to_date_tensor([(2022, 1, 15),
                                                        (2022, 1, 15)])
    batch_start_date = dates.convert_to_date_tensor([(2021, 1, 15),
                                                     (2021, 1, 15)])
    batch_valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])

    period3m = dates.periods.months(3)
    cap = instruments.CapAndFloor(
        batch_start_date,
        batch_maturity_date,
        period3m,
        [0.005, 0.01],
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        notional=notional,
        dtype=dtype)
    price = self.evaluate(
        cap.price(
            batch_valuation_date,
            self.get_market(),
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=0.5))
    np_testing.assert_allclose(price,
                               [1.0474063612452953, 0.5656630014452084],
                               atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cap_reset(self, dtype):
    notional = 100.0
    maturity_date = dates.convert_to_date_tensor([(2022, 1, 15),
                                                  (2022, 1, 15)])
    start_date = dates.convert_to_date_tensor([(2021, 1, 15),
                                               (2021, 1, 15)])
    valuation_date = dates.convert_to_date_tensor([(2021, 2, 1)])

    period3m = dates.periods.months(3)
    cap = instruments.CapAndFloor(
        start_date,
        maturity_date,
        period3m,
        [0.005, 0.01],
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        notional=notional,
        dtype=dtype)
    curve_valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    curve_dates = curve_valuation_date + dates.periods.months([0, 3, 12, 24])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.005, 0.01, 0.015, 0.02], dtype=np.float64),
        valuation_date=curve_valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve,
        discount_curve=reference_curve,
        libor_rate=[0.006556, 0.006556])

    price = self.evaluate(
        cap.price(
            valuation_date,
            market,
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=0.5))
    np_testing.assert_allclose(price,
                               [0.9389714183634128, 0.5354250398709062],
                               atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cap_fwd_rate(self, dtype):
    notional = 100.0
    period3m = dates.periods.months(3)
    cap = instruments.CapAndFloor(
        self.start_date,
        self.maturity_date,
        period3m,
        0.005,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        notional=notional,
        dtype=dtype)
    fwd_rates = self.evaluate(
        cap._get_forward_rate(
            dates.convert_to_date_tensor(self.valuation_date),
            self.get_market()))
    print(fwd_rates)
    np_testing.assert_allclose(fwd_rates,
                               [0.010966, 0.013824, 0.017164, 0.020266],
                               atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cap_price_lognormal_rate_model(self, dtype):
    notional = 100.0
    period3m = dates.periods.months(3)
    cap = instruments.CapAndFloor(
        self.start_date,
        self.maturity_date,
        period3m,
        0.005,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        notional=notional,
        dtype=dtype)
    price = self.evaluate(
        cap._price_lognormal_rate(
            dates.convert_to_date_tensor(self.valuation_date),
            self.get_market(),
            pricing_context=0.5))
    print(price)
    np_testing.assert_allclose(
        price, [0.146671, 0.218595, 0.303358, 0.378782], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
