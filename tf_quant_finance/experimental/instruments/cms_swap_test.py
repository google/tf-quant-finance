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
"""Tests for cms_swap.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime
instruments = tff.experimental.instruments


@test_util.run_all_in_graph_and_eager_modes
class CMSSwapTest(tf.test.TestCase, parameterized.TestCase):

  def get_cms_coupon_spec(self, fix_rate, fix_leg_freq='6m'):
    p3m = dates.periods.months(3)
    p6m = dates.periods.months(6)
    p1y = dates.periods.year()
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=p6m if fix_leg_freq == '6m' else p3m,
        currency='usd',
        notional=1.,
        coupon_rate=fix_rate,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    flt_spec = instruments.FloatCouponSpecs(
        coupon_frequency=p3m,
        reference_rate_term=p3m,
        reset_frequency=p3m,
        currency='usd',
        notional=1.,
        businessday_rule=dates.BusinessDayConvention.NONE,
        coupon_basis=0.,
        coupon_multiplier=1.,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)
    cms_spec = instruments.CMSCouponSpecs(
        coupon_frequency=p3m,
        tenor=p1y,
        float_leg=flt_spec,
        fixed_leg=fix_spec,
        notional=1.,
        coupon_basis=0.,
        coupon_multiplier=1.,
        businessday_rule=None,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365)
    return cms_spec

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cms_stream_no_convexity(self, dtype):
    start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2023, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])

    cms = instruments.CMSCashflowStream(
        start_date, maturity_date, [self.get_cms_coupon_spec(0.0)],
        dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([0, 1, 2, 3, 5])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02, 0.02, 0.025, 0.03, 0.035
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cms.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.0555126295434207, atol=1e-7)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cms_stream_many(self, dtype):
    start_date = dates.convert_to_date_tensor([(2021, 1, 1), (2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2023, 1, 1), (2022, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])

    cms_spec = self.get_cms_coupon_spec(0.0)
    cms = instruments.CMSCashflowStream(
        start_date, maturity_date, [cms_spec, cms_spec], dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([0, 1, 2, 3, 5])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02, 0.02, 0.025, 0.03, 0.035
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cms.price(valuation_date, market))
    np.testing.assert_allclose(price,
                               [0.0555126295434207, 0.022785926686551876],
                               atol=1e-7)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cms_stream_past_fixing(self, dtype):
    start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2023, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 10)])

    cms = instruments.CMSCashflowStream(
        start_date, maturity_date, [self.get_cms_coupon_spec(0.0)], dtype=dtype)

    curve_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    curve_dates = curve_date + dates.periods.years([0, 1, 2, 3, 5])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02, 0.02, 0.025, 0.03, 0.035
        ], dtype=np.float64),
        valuation_date=curve_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve,
        discount_curve=reference_curve,
        swap_rate=0.01)

    price = self.evaluate(cms.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.053034387186703995, atol=1e-7)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_cms_to_fixed_swap(self, dtype):
    start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2023, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    p6m = dates.periods.months(6)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=p6m,
        currency='usd',
        notional=1.,
        coupon_rate=0.02,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    cms_spec = self.get_cms_coupon_spec(0.0)

    cms = instruments.CMSSwap(
        start_date, maturity_date, [fix_spec],
        [cms_spec], dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([0, 1, 2, 3, 5])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02, 0.02, 0.025, 0.03, 0.035
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cms.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.016629820479418966, atol=1e-7)

  @parameterized.named_parameters(
      ('LognormalRate', instruments.InterestRateModelType.LOGNORMAL_RATE, 0.15,
       0.0006076380479708987),
      ('NormalRate', instruments.InterestRateModelType.NORMAL_RATE, 0.003,
       0.0005958813803308),
      ('ReplicationLn',
       instruments.InterestRateModelType.LOGNORMAL_SMILE_CONSISTENT_REPLICATION,
       0.15, 0.0006076275782589),
      ('ReplicationNormal',
       instruments.InterestRateModelType.NORMAL_SMILE_CONSISTENT_REPLICATION,
       0.003, 0.0005956153840977),
  )
  def test_cms_convexity_model(self, model, parameter, expected):
    dtype = np.float64
    start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2031, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    p3m = dates.periods.months(3)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=p3m,
        currency='usd',
        notional=1.,
        coupon_rate=0.02,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    cms_spec = self.get_cms_coupon_spec(0.0, '3m')

    cms = instruments.CMSSwap(
        start_date, maturity_date, [fix_spec],
        [cms_spec], dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([0, 360])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02, 0.02
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cms.price(valuation_date, market, model=model,
                                    pricing_context=parameter))
    np.testing.assert_allclose(price, expected, atol=1e-7)

  @parameterized.named_parameters(
      ('None', None, 0.15, 0.0009080833232544),
      ('LognormalRate', instruments.InterestRateModelType.LOGNORMAL_RATE, 0.15,
       0.0011142108850073),
      ('NormalRate', instruments.InterestRateModelType.NORMAL_RATE, 0.003,
       0.0010975575420384),
      ('ReplicationLn',
       instruments.InterestRateModelType.LOGNORMAL_SMILE_CONSISTENT_REPLICATION,
       0.15, 0.0011141562723458),
      ('ReplicationNormal',
       instruments.InterestRateModelType.NORMAL_SMILE_CONSISTENT_REPLICATION,
       0.003, 0.0010973379723202),
  )
  def test_cms_convexity_model_6m(self, model, parameter, expected):
    dtype = np.float64
    start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    maturity_date = dates.convert_to_date_tensor([(2031, 1, 1)])
    valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
    p3m = dates.periods.months(3)
    fix_spec = instruments.FixedCouponSpecs(
        coupon_frequency=p3m,
        currency='usd',
        notional=1.,
        coupon_rate=0.02,
        daycount_convention=instruments.DayCountConvention.ACTUAL_365,
        businessday_rule=dates.BusinessDayConvention.NONE)
    cms_spec = self.get_cms_coupon_spec(0.0)

    cms = instruments.CMSSwap(
        start_date, maturity_date, [fix_spec],
        [cms_spec], dtype=dtype)

    curve_dates = valuation_date + dates.periods.years([0, 360])
    reference_curve = instruments.RateCurve(
        curve_dates,
        np.array([
            0.02, 0.02
        ], dtype=np.float64),
        valuation_date=valuation_date,
        dtype=np.float64)
    market = instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)

    price = self.evaluate(cms.price(valuation_date, market, model=model,
                                    pricing_context=parameter))
    np.testing.assert_allclose(price, expected, atol=1e-7)

if __name__ == '__main__':
  tf.test.main()
