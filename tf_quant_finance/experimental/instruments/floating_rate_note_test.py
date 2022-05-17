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
"""Tests for floating_rate_note.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class FloatingRateNoteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_frn_correctness(self, dtype):
    settlement_date = dates.convert_to_date_tensor([(2021, 1, 15)])
    maturity_date = dates.convert_to_date_tensor([(2022, 1, 15)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 15)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=100.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    frn = instruments.FloatingRateNote(settlement_date, maturity_date,
                                       [flt_spec],
                                       dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.0, 0.005, 0.007, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                            reference_curve=reference_curve)

    price = self.evaluate(frn.price(valuation_date, market))
    np.testing.assert_allclose(price, 100.0, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_frn_correctness_fwd_start(self, dtype):
    settlement_date = dates.convert_to_date_tensor([(2021, 1, 15)])
    start_date = dates.convert_to_date_tensor([(2021, 4, 15)])
    maturity_date = dates.convert_to_date_tensor([(2022, 1, 15)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 15)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=100.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    frn = instruments.FloatingRateNote(settlement_date, maturity_date,
                                       [flt_spec],
                                       start_date=start_date,
                                       dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.0, 0.005, 0.007, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                            reference_curve=reference_curve)

    price = self.evaluate(frn.price(valuation_date, market))
    np.testing.assert_allclose(price, 99.9387155246714656, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_frn_many(self, dtype):
    settlement_date = dates.convert_to_date_tensor([(2021, 1, 15),
                                                    (2021, 1, 15)])
    maturity_date = dates.convert_to_date_tensor([(2022, 1, 15),
                                                  (2022, 1, 15)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 15)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=100.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    frn = instruments.FloatingRateNote(settlement_date, maturity_date,
                                       [flt_spec, flt_spec],
                                       dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.0, 0.005, 0.007, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                            reference_curve=reference_curve)

    price = self.evaluate(frn.price(valuation_date, market))
    np.testing.assert_allclose(price, [100.0, 100.0], atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_frn_basis(self, dtype):
    settlement_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2022, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=100.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.01,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    frn = instruments.FloatingRateNote(settlement_date, maturity_date,
                                       [flt_spec],
                                       dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.001, 0.005, 0.007, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                            reference_curve=reference_curve)

    price = self.evaluate(frn.price(valuation_date, market))
    np.testing.assert_allclose(price, 100.996314114175, atol=1e-7)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_frn_stub_begin(self, dtype):
    settlement_date = dates.convert_to_date_tensor([(2021, 3, 1)])
    maturity_date = dates.convert_to_date_tensor([(2022, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 3, 1)])
    first_coupon_date = dates.convert_to_date_tensor([(2021, 4, 1)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=100.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.01,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    frn = instruments.FloatingRateNote(settlement_date, maturity_date,
                                       [flt_spec],
                                       first_coupon_date=first_coupon_date,
                                       dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.001, 0.005, 0.007, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                            reference_curve=reference_curve)

    price = self.evaluate(frn.price(valuation_date, market))
    np.testing.assert_allclose(price, 100.83591541528823, atol=1e-7)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_frn_stub_end(self, dtype):
    settlement_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2022, 2, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    penultimate_coupon_date = dates.convert_to_date_tensor([(2022, 1, 1)])
    period_3m = dates.periods.months(3)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=period_3m,
        reference_rate_term=period_3m,
        reset_frequency=period_3m,
        currency='usd',
        notional=100.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.01,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)

    frn = instruments.FloatingRateNote(
        settlement_date,
        maturity_date, [flt_spec],
        penultimate_coupon_date=penultimate_coupon_date,
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([0, 6, 12, 36])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([0.001, 0.005, 0.007, 0.015], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = instruments.InterestRateMarket(discount_curve=reference_curve,
                                            reference_curve=reference_curve)

    price = self.evaluate(frn.price(valuation_date, market))
    np.testing.assert_allclose(price, 101.08057198860133, atol=1e-7)

if __name__ == '__main__':
  tf.test.main()
