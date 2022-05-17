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
"""Tests for swaption_impl.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
core = tff.experimental.pricing_platform.framework.core
dates = tff.datetime
market_data = tff.experimental.pricing_platform.framework.market_data
rates_instruments = tff.experimental.pricing_platform.framework.rate_instruments

instrument_protos = tff.experimental.pricing_platform.instrument_protos
date_pb2 = instrument_protos.date
decimal_pb2 = instrument_protos.decimal
swap_pb2 = instrument_protos.interest_rate_swap
period_pb2 = instrument_protos.period
rate_indices_pb2 = instrument_protos.rate_indices
swaption_pb2 = instrument_protos.swaption


# @test_util.run_all_in_graph_and_eager_modes
class SwaptionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    notional = 100.0
    self._maturity_date = dates.convert_to_date_tensor([(2025, 2, 10)])
    self._start_date = dates.convert_to_date_tensor([(2022, 2, 10)])
    self._expiry_date = dates.convert_to_date_tensor([(2022, 2, 10)])

    self._maturity_date_batch = dates.convert_to_date_tensor([(2025, 2, 10),
                                                              (2025, 2, 10)])
    self._start_date_batch = dates.convert_to_date_tensor([(2022, 2, 10),
                                                           (2022, 2, 10)])
    self._expiry_date_batch = dates.convert_to_date_tensor([(2022, 2, 10),
                                                            (2022, 2, 10)])

    self._valuation_date = dates.convert_to_date_tensor([(2020, 2, 10)])
    hull_white_config = core.models.HullWhite1FactorConfig(
        mean_reversion=[0.03], volatility=[0.01])
    self._swaption_config = rates_instruments.swaption.SwaptionConfig(
        model_params=hull_white_config)

    period3m = dates.periods.months(3)
    period6m = dates.periods.months(6)
    calendar = dates.create_holiday_calendar(
        weekend_mask=dates.WeekendMask.NONE)
    self._fix_spec = rates_instruments.coupon_specs.FixedCouponSpecs(
        coupon_frequency=period6m, currency='USD', notional_amount=notional,
        fixed_rate=0.011,
        daycount_convention=core.daycount_conventions.DayCountConventions
        .ACTUAL_365,
        businessday_rule=core.business_days.BusinessDayConvention.NO_ADJUSTMENT,
        settlement_days=0,
        calendar=calendar)
    self._flt_spec = rates_instruments.coupon_specs.FloatCouponSpecs(
        coupon_frequency=period3m,
        reset_frequency=period3m, currency='USD', notional_amount=notional,
        businessday_rule=core.business_days.BusinessDayConvention.NO_ADJUSTMENT,
        daycount_convention=core.daycount_conventions.DayCountConventions
        .ACTUAL_365,
        settlement_days=0,
        spread=0.,
        calendar=calendar,
        floating_rate_type=core.rate_indices.RateIndex(
            type=core.rate_indices.RateIndexType.LIBOR_3M))

    # Create swaption protos
    swapproto = swap_pb2.InterestRateSwap(
        effective_date=date_pb2.Date(year=2022, month=2, day=10),
        maturity_date=date_pb2.Date(year=2025, month=2, day=10),
        currency=core.currencies.Currency.USD(),
        pay_leg=swap_pb2.SwapLeg(
            fixed_leg=swap_pb2.FixedLeg(
                currency=core.currencies.Currency.USD(),
                coupon_frequency=period_pb2.Period(type='MONTH', amount=6),
                notional_amount=decimal_pb2.Decimal(units=100),
                fixed_rate=decimal_pb2.Decimal(nanos=11000000),
                daycount_convention=core.daycount_conventions
                .DayCountConventions.ACTUAL_365(),
                business_day_convention=core.business_days.BusinessDayConvention
                .NO_ADJUSTMENT(),
                settlement_days=0)),
        receive_leg=swap_pb2.SwapLeg(
            floating_leg=swap_pb2.FloatingLeg(
                currency=core.currencies.Currency.USD(),
                coupon_frequency=period_pb2.Period(type='MONTH', amount=3),
                reset_frequency=period_pb2.Period(type='MONTH', amount=3),
                notional_amount=decimal_pb2.Decimal(units=100),
                floating_rate_type=rate_indices_pb2.RateIndex(type='LIBOR_3M'),
                daycount_convention=core.daycount_conventions
                .DayCountConventions.ACTUAL_365(),
                business_day_convention=core.business_days.BusinessDayConvention
                .NO_ADJUSTMENT(),
                settlement_days=0)))
    self._swaption_proto = swaption_pb2.Swaption(
        swap=swapproto, expiry_date=date_pb2.Date(year=2022, month=2, day=10))

    curve_dates = self._valuation_date + dates.periods.years(
        [1, 2, 3, 5, 7, 10, 30])

    curve_discounts = np.exp(-0.01 * np.array([1, 2, 3, 5, 7, 10, 30]))

    libor_3m_config = {
        'interpolation_method': core.interpolation_method.InterpolationMethod
                                .LINEAR
    }

    market_data_dict = {
        'rates': {
            'USD': {
                'risk_free_curve': {
                    'dates': curve_dates,
                    'discounts': curve_discounts,
                },
                'LIBOR_3M': {
                    'dates': curve_dates,
                    'discounts': curve_discounts,
                    'config': libor_3m_config,
                }
            }
        },
        'reference_date': self._valuation_date,
    }

    self._market = market_data.MarketDataDict(
        market_data_dict)
    super(SwaptionTest, self).setUp()

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_swaption_correctness(self, dtype):
    swap = rates_instruments.interest_rate_swap.InterestRateSwap(
        start_date=self._start_date,
        maturity_date=self._maturity_date,
        pay_leg=self._fix_spec,
        receive_leg=self._flt_spec,
        dtype=dtype)
    swaption = rates_instruments.swaption.Swaption(
        swap, self._expiry_date, config=self._swaption_config,
        dtype=dtype)
    price = self.evaluate(swaption.price(self._market))
    np.testing.assert_allclose(price, 1.38594754, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_swaption_correctness_batch(self, dtype):
    swap = rates_instruments.interest_rate_swap.InterestRateSwap(
        start_date=self._start_date_batch,
        maturity_date=self._maturity_date_batch,
        pay_leg=self._fix_spec,
        receive_leg=self._flt_spec,
        dtype=dtype)
    swaption = rates_instruments.swaption.Swaption(
        swap, self._expiry_date_batch, config=self._swaption_config,
        dtype=dtype)
    price = self.evaluate(swaption.price(self._market))
    np.testing.assert_allclose(price, [1.38594754, 1.38594754], atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_swaption_correctness_pb(self, dtype):
    swaption = rates_instruments.swaption.Swaption.from_protos(
        [self._swaption_proto], config=self._swaption_config)
    price = self.evaluate(swaption[0].price(self._market))
    np.testing.assert_allclose(price, 1.38594754, atol=1e-6)

  @parameterized.named_parameters(
      ('DoublePrecision', np.float64),
  )
  def test_swaption_correctness_batch_pb(self, dtype):
    swaption = rates_instruments.swaption.Swaption.from_protos(
        [self._swaption_proto, self._swaption_proto],
        config=self._swaption_config)
    price = self.evaluate(swaption[0].price(self._market))
    np.testing.assert_allclose(price, [1.38594754, 1.38594754], atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
