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
"""Tests for swaption.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class SwaptionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_swaption_correctness(self, dtype):
    notional = 1.e6
    maturity_date = dates.convert_to_date_tensor([(2025, 2, 8)])
    start_date = dates.convert_to_date_tensor([(2022, 2, 8)])
    expiry_date = dates.convert_to_date_tensor([(2022, 2, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])

    period3m = dates.periods.months(3)
    period6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period6m, currency='usd', notional=notional,
        coupon_rate=0.03134,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period3m, reference_rate_term=period3m,
        reset_frequency=period3m, currency='usd', notional=notional,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0., coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    swap = instruments.InterestRateSwap(start_date, maturity_date,
                                        [fix_spec], [flt_spec],
                                        dtype=dtype)
    swaption = instruments.Swaption(swap, expiry_date, dtype=dtype)

    curve_dates = valuation_date + dates.periods.years(
        [1, 2, 3, 5, 7, 10, 30])

    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(
        swaption.price(
            valuation_date,
            market,
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=0.5))
    np.testing.assert_allclose(price, 24145.254011, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_swaption_many(self, dtype):
    notional = 1.e6
    maturity_date = dates.convert_to_date_tensor([(2025, 2, 8), (2025, 2, 8)])
    start_date = dates.convert_to_date_tensor([(2022, 2, 8), (2022, 2, 8)])
    expiry_date = dates.convert_to_date_tensor([(2022, 2, 8), (2022, 2, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])

    period3m = dates.periods.months(3)
    period6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period6m, currency='usd', notional=notional,
        coupon_rate=0.03134,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period3m, reference_rate_term=period3m,
        reset_frequency=period3m, currency='usd', notional=notional,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0., coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    swap = instruments.InterestRateSwap(start_date, maturity_date,
                                        [fix_spec, fix_spec],
                                        [flt_spec, flt_spec],
                                        dtype=dtype)
    swaption = instruments.Swaption(swap, expiry_date, dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])

    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(
        swaption.price(
            valuation_date,
            market,
            model=instruments.InterestRateModelType.LOGNORMAL_RATE,
            pricing_context=[0.5, 0.5]))
    np.testing.assert_allclose(price, [24145.254011, 24145.254011], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
