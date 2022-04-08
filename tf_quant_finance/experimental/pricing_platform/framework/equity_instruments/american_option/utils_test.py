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
"""American option pricer test."""

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

from tf_quant_finance.experimental.pricing_platform.framework.equity_instruments.american_option import utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(tf.test.TestCase):

  def test_put_payoff_function(self):
    """Tests Black-Scholes batched Monte Carlo American option pricing."""
    dtype = tf.float64
    spot = tf.constant([100, 100], dtype=dtype)
    strikes = tf.constant([104, 90], dtype=dtype)
    is_call_option = [True, False]
    # These are expiries at which the strike prices are computed
    expiries = tf.constant([0.1, 1.0], dtype=dtype)
    volatility = 0.2 * tf.ones_like(spot)
    discount_factors = 0.95 * tf.ones_like(spot)
    seed = tf.constant([12, 42])
    # Compute option prices
    am_option_prices = utils.bs_lsm_price(
        spots=spot,
        expiry_times=expiries,
        strikes=strikes,
        volatility=volatility,
        discount_factors=discount_factors,
        num_samples=20000,
        num_exercise_times=50,
        num_calibration_samples=2000,
        is_call_option=is_call_option,
        seed=seed,
        dtype=dtype)

    # Expected prices obtained using binomial pricer
    # tff.black_scholes.option_price_binomial
    expected_prices = [3.15257036, 2.44969723]
    self.assertAllClose(expected_prices, am_option_prices, rtol=1e-2, atol=1e-2)

  def test_basis_fn(self):
    """Tests option pricing with a basis functions of polynomials of order 4."""
    dtype = tf.float64
    spot = tf.constant([696, 286], dtype=dtype)
    strikes = tf.constant([2131, 408], dtype=dtype)
    risk_free_rates = tf.constant([0.2, 0.14], dtype=dtype)
    is_call_option = [True, False]
    # These are expiries at which the strike prices are computed
    expiries = tf.constant([7.2, 4.2], dtype=dtype)
    volatility = tf.constant([0.23, 0.55], dtype=dtype)
    discount_factors = tf.math.exp(-expiries * risk_free_rates)
    seed = tf.constant([4, 2])
    # Compute option prices
    am_option_prices = utils.bs_lsm_price(
        spots=spot,
        expiry_times=expiries,
        strikes=strikes,
        volatility=volatility,
        discount_factors=discount_factors,
        num_samples=6000,
        num_exercise_times=50,
        num_calibration_samples=1000,
        is_call_option=is_call_option,
        basis_fn=tff.models.longstaff_schwartz.make_polynomial_basis(4),
        seed=seed,
        dtype=dtype)

    # Expected prices obtained using binomial pricer
    # tff.black_scholes.option_price_binomial
    expected_prices = [262.3, 143.72]
    self.assertAllClose(expected_prices, am_option_prices,
                        rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
  tf.test.main()
