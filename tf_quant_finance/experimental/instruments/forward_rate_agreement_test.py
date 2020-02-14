# Lint as: python3
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


@test_util.run_all_in_graph_and_eager_modes
class ForwardRateAgreementTest(tf.test.TestCase):

  def test_fra_correctness(self):
    dtype = np.float64
    notional = 1.
    settlement_date = tff.experimental.dates.convert_to_date_tensor(
        [(2021, 2, 8)])
    fixing_date = tff.experimental.dates.convert_to_date_tensor([(2021, 2, 8)])
    valuation_date = tff.experimental.dates.convert_to_date_tensor([(2020, 2, 8)
                                                                   ])
    fixed_rate = 0.02
    rate_term = tff.experimental.dates.periods.PeriodTensor(
        3, tff.experimental.dates.PeriodType.MONTH)
    fra = tff.experimental.instruments.ForwardRateAgreement(
        settlement_date, fixing_date, fixed_rate, notional=notional,
        rate_term=rate_term, dtype=dtype)

    reference_curve = tff.experimental.instruments.RateCurve(
        np.array([1./12, 2./12, 0.25, 1., 2., 5.], dtype=dtype),
        np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
        dtype=dtype)
    market = tff.experimental.instruments.InterestRateMarket(
        reference_curve=reference_curve, discount_curve=reference_curve)
    price = self.evaluate(fra.price(valuation_date, market))
    np.testing.assert_allclose(price, 0.00378275, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
