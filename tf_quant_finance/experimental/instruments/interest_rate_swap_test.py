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
"""Tests for interest_rate_swap.py."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class InterestRateSwapTest(tf.test.TestCase):

  def setUp(self):
    super(InterestRateSwapTest, self).setUp()
    self.maturity_date = [(2023, 2, 8)]
    self.start_date = [(2020, 2, 8)]
    self.valuation_date = [(2020, 2, 8)]

  def get_market(self):
    val_date = dates.convert_to_date_tensor(self.valuation_date)
    curve_dates = val_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])

    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=np.float64),
        valuation_date=val_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    return market

  def test_irs_correctness(self):
    dtype = np.float64
    notional = 1.

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

    swap = instruments.InterestRateSwap(self.start_date, self.maturity_date,
                                        [fix_spec], [flt_spec],
                                        dtype=dtype)

    price = self.evaluate(swap.price(self.valuation_date, self.get_market()))
    np.testing.assert_allclose(price, 1.e-7, atol=1e-6)

  def test_irs_correctness_scalar_spec(self):
    dtype = np.float64
    notional = 1.

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

    swap = instruments.InterestRateSwap(self.start_date, self.maturity_date,
                                        fix_spec, flt_spec,
                                        dtype=dtype)

    price = self.evaluate(swap.price(self.valuation_date, self.get_market()))
    np.testing.assert_allclose(price, 1.e-7, atol=1e-6)

  def test_irs_correctness_batch(self):
    dtype = np.float64
    notional = 1.0
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 8), (2027, 2, 8)])
    start_date = dates.convert_to_date_tensor([(2020, 2, 8), (2020, 2, 8)])

    period3m = dates.periods.months([3, 3])
    period6m = dates.periods.months([6, 6])
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period6m, currency='usd',
        notional=notional,
        coupon_rate=[0.03134, 0.03181],
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period3m, reference_rate_term=period3m,
        reset_frequency=period3m, currency='usd',
        notional=notional,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.0, coupon_multiplier=1.0,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    swap = instruments.InterestRateSwap(start_date, maturity_date,
                                        fix_spec, flt_spec,
                                        dtype=dtype)

    price = self.evaluate(swap.price(self.valuation_date, self.get_market()))
    np.testing.assert_allclose(price, [1.0e-7, 1.0e-7], atol=1e-6)

  def test_irs_parrate(self):
    dtype = np.float64
    notional = 1.

    period3m = dates.periods.months(3)
    period6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period6m, currency='usd', notional=notional,
        coupon_rate=0.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period3m, reference_rate_term=period3m,
        reset_frequency=period3m, currency='usd', notional=notional,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0., coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    swap = instruments.InterestRateSwap(self.start_date, self.maturity_date,
                                        [fix_spec], [flt_spec],
                                        dtype=dtype)

    price = self.evaluate(swap.par_rate(self.valuation_date,
                                        self.get_market()))
    np.testing.assert_allclose(price, 0.03134, atol=1e-6)

  def test_irs_parrate_many(self):
    dtype = np.float64
    notional = 1.
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 8), (2025, 2, 8)])
    start_date = dates.convert_to_date_tensor([(2020, 2, 8), (2020, 2, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])

    period3m = dates.periods.months(3)
    period6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period6m, currency='usd', notional=notional,
        coupon_rate=0.,
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

    price = self.evaluate(swap.par_rate(valuation_date, market))
    np.testing.assert_allclose(price, [0.03134, 0.03152], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
