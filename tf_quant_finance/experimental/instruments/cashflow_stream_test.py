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
"""Tests for cashflow_stream.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class CashflowStreamTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fixed_stream(self, dtype):
    start_date = dates.convert_to_date_tensor([(2020, 2, 2)])
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 2)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
    period_6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period_6m,
        currency='usd',
        notional=1.,
        coupon_rate=0.03134,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)

    cf_stream = instruments.FixedCashflowStream(start_date, maturity_date,
                                                [fix_spec],
                                                dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cf_stream.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.089259267853547, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fixed_stream_many(self, dtype):
    start_date = dates.convert_to_date_tensor([(2020, 2, 2), (2020, 2, 2)])
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 2), (2023, 2, 2)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
    period_6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period_6m,
        currency='usd',
        notional=1.,
        coupon_rate=0.03134,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)

    cf_stream = instruments.FixedCashflowStream(start_date, maturity_date,
                                                [fix_spec, fix_spec],
                                                dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cf_stream.price(valuation_date, market))
    np.testing.assert_allclose(price, [0.089259267853547, 0.089259267853547],
                               atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_floating_stream(self, dtype):
    start_date = dates.convert_to_date_tensor([(2020, 2, 2)])
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 2)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=1.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    cf_stream = instruments.FloatingCashflowStream(start_date,
                                                   maturity_date, [flt_spec],
                                                   dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cf_stream.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.089259685614769, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_floating_stream_many(self, dtype):
    start_date = dates.convert_to_date_tensor([(2020, 2, 2), (2020, 2, 2)])
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 2), (2023, 2, 2)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 2)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=1.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    cf_stream = instruments.FloatingCashflowStream(start_date,
                                                   maturity_date,
                                                   [flt_spec, flt_spec],
                                                   dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cf_stream.price(valuation_date, market))
    np.testing.assert_allclose(price, [0.089259685614769, 0.089259685614769],
                               atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_floating_stream_past_fixing(self, dtype):
    start_date = dates.convert_to_date_tensor([(2020, 2, 2), (2020, 2, 1)])
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 2), (2023, 2, 2)])
    valuation_date = dates.convert_to_date_tensor([(2020, 7, 3)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=1.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    cf_stream = instruments.FloatingCashflowStream(start_date,
                                                   maturity_date,
                                                   [flt_spec, flt_spec],
                                                   dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve,
        libor_rate=[0.01, 0.02])

    price = self.evaluate(cf_stream.price(valuation_date, market))
    np.testing.assert_allclose(price, [0.07720258, 0.08694714],
                               atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_fixed_stream_past_fixing(self, dtype):
    start_date = dates.convert_to_date_tensor([(2020, 2, 2), (2020, 2, 2)])
    maturity_date = dates.convert_to_date_tensor([(2023, 2, 2), (2024, 2, 2)])
    valuation_date = dates.convert_to_date_tensor([(2021, 3, 2)])
    period_6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=period_6m,
        currency='usd',
        notional=1.,
        coupon_rate=0.03134,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)

    cf_stream = instruments.FixedCashflowStream(start_date, maturity_date,
                                                [fix_spec, fix_spec],
                                                dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
            0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cf_stream.price(valuation_date, market))
    np.testing.assert_allclose(price, [0.06055127, 0.08939763],
                               atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
