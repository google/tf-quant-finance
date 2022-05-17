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
"""Tests for forward_rate_agreement.py."""

import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
dates = tff.datetime


@test_util.run_all_in_graph_and_eager_modes
class ForwardRateAgreementTest(tf.test.TestCase):

  def test_fra_correctness(self):
    dtype = np.float64
    notional = 1.
    settlement_date = dates.convert_to_date_tensor(
        [(2021, 2, 8)])
    fixing_date = dates.convert_to_date_tensor([(2021, 2, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    fixed_rate = 0.02
    rate_term = dates.periods.months(3)
    fra = tff.experimental.instruments.ForwardRateAgreement(
        settlement_date, fixing_date, fixed_rate, notional=notional,
        rate_term=rate_term, dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
    reference_curve = tff.experimental.instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    price = self.evaluate(fra.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.00377957, atol=1e-6)

  def test_fra_explicit_maturity(self):
    dtype = np.float64
    notional = 1.
    settlement_date = dates.convert_to_date_tensor(
        [(2021, 2, 8)])
    fixing_date = dates.convert_to_date_tensor([(2021, 2, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    fixed_rate = 0.02
    maturity_date = dates.convert_to_date_tensor(
        [(2021, 5, 8)])
    fra = tff.experimental.instruments.ForwardRateAgreement(
        settlement_date, fixing_date, fixed_rate, notional=notional,
        maturity_date=maturity_date, dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
    reference_curve = tff.experimental.instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    price = self.evaluate(fra.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.00377957, atol=1e-6)

  def test_fra_many(self):
    dtype = np.float64
    notional = 1.
    settlement_date = dates.convert_to_date_tensor(
        [(2021, 2, 8), (2021, 5, 8), (2021, 8, 8)])
    fixing_date = dates.convert_to_date_tensor(
        [(2021, 2, 8), (2021, 5, 8), (2021, 8, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    fixed_rate = tf.convert_to_tensor([0.02, 0.021, 0.022], dtype=dtype)
    rate_term = dates.periods.months([3, 3, 3])
    fra = tff.experimental.instruments.ForwardRateAgreement(
        settlement_date, fixing_date, fixed_rate, notional=notional,
        rate_term=rate_term, dtype=dtype)

    curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
    reference_curve = tff.experimental.instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    price = self.evaluate(fra.price(valuation_date, market))
    np.testing.assert_allclose(price, [0.00377957, 0.0042278427, 0.004548173],
                               atol=1e-6)

  def test_fra_many_act365(self):
    dtype = np.float64
    notional = 1.
    settlement_date = dates.convert_to_date_tensor(
        [(2021, 2, 8), (2021, 5, 8), (2021, 8, 8)])
    fixing_date = dates.convert_to_date_tensor(
        [(2021, 2, 8), (2021, 5, 8), (2021, 8, 8)])
    valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
    fixed_rate = tf.convert_to_tensor([0.02, 0.021, 0.022], dtype=dtype)
    rate_term = dates.periods.months([3, 3, 3])
    fra = tff.experimental.instruments.ForwardRateAgreement(
        settlement_date,
        fixing_date,
        fixed_rate,
        notional=notional,
        rate_term=rate_term,
        dtype=dtype,
        daycount_convention=tff.experimental.instruments.DayCountConvention
        .ACTUAL_365)

    curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
    reference_curve = tff.experimental.instruments.RateCurve(
        curve_dates,
        np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
        valuation_date=valuation_date,
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    price = self.evaluate(fra.price(valuation_date, market))
    np.testing.assert_allclose(price, [0.003844721, 0.004297866, 0.00462077292],
                               atol=1e-6)

if __name__ == '__main__':
  tf.test.main()
