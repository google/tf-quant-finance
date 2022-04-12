# Copyright 2019 Google LLC
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
"""TensorFlow Quantitative Finance volatility surfaces and vanilla options."""

from tf_quant_finance.black_scholes import approximations
from tf_quant_finance.black_scholes import brownian_bridge
from tf_quant_finance.black_scholes import crr_binomial_tree
from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.black_scholes import variance_swaps
from tf_quant_finance.black_scholes.implied_vol_approximation import implied_vol as implied_vol_approx
from tf_quant_finance.black_scholes.implied_vol_lib import implied_vol
from tf_quant_finance.black_scholes.implied_vol_lib import ImpliedVolMethod
from tf_quant_finance.black_scholes.implied_vol_newton_root import implied_vol as implied_vol_newton
from tf_quant_finance.black_scholes.implied_vol_utils import UnderlyingDistribution as ImpliedVolUnderlyingDistribution

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

asset_or_nothing_price = vanilla_prices.asset_or_nothing_price
binary_price = vanilla_prices.binary_price
barrier_price = vanilla_prices.barrier_price
brownian_bridge_single = brownian_bridge.brownian_bridge_single
brownian_bridge_double = brownian_bridge.brownian_bridge_double
option_price = vanilla_prices.option_price
option_price_binomial = crr_binomial_tree.option_price_binomial
swaption_price = vanilla_prices.swaption_price
variance_swap_fair_strike = variance_swaps.fair_strike

_allowed_symbols = [
    'approximations',
    'asset_or_nothing_price',
    'binary_price',
    'brownian_bridge_single',
    'brownian_bridge_double',
    'implied_vol',
    'implied_vol_approx',
    'implied_vol_newton',
    'option_price',
    'option_price_binomial',
    'ImpliedVolMethod',
    'ImpliedVolUnderlyingDistribution',
    'barrier_price',
    'swaption_price',
    'variance_swap_fair_strike'
]

remove_undocumented(__name__, _allowed_symbols)
