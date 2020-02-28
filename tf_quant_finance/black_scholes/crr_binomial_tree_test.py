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
"""Tests for Cox Ross Rubinstein Binomial tree method."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


# TODO(b/150448863): Enhance test coverage.
@test_util.run_all_in_graph_and_eager_modes
class BinomialModelPrice(parameterized.TestCase, tf.test.TestCase):
  """Tests for Binomial tree method prices."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_option_prices(self, dtype):
    """Tests that the BS prices are correct."""
    spots = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)
    volatilities = np.array([0.1, 0.22, 0.32, 0.01, 0.4], dtype=dtype)
    is_call_options = np.array([True, True, False, False, False])
    is_american = np.array([False, True, True, False, True])
    discount_rates = np.array(0.035, dtype=dtype)
    dividend_rates = np.array([0.02, 0.0, 0.07, 0.01, 0.0], dtype=dtype)
    expiries = np.array(1.0, dtype=dtype)

    prices = self.evaluate(
        tff.black_scholes.option_price_binomial(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            spots=spots,
            discount_rates=discount_rates,
            dividend_rates=dividend_rates,
            is_call_options=is_call_options,
            is_american=is_american,
            dtype=dtype))
    expected_prices = np.array(
        [0., 0.0098847, 0.41299509, 0., 0.06046989])
    self.assertArrayNear(expected_prices, prices, 1e-5)


if __name__ == '__main__':
  tf.test.main()
