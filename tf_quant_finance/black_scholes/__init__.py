# Lint as: python3
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

from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.black_scholes.implied_vol_approximation import implied_vol as implied_vol_approx
from tf_quant_finance.black_scholes.implied_vol_lib import implied_vol
from tf_quant_finance.black_scholes.implied_vol_lib import ImpliedVolMethod
from tf_quant_finance.black_scholes.implied_vol_newton_root import implied_vol as implied_vol_newton

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

binary_price = vanilla_prices.binary_price
option_price = vanilla_prices.option_price

_allowed_symbols = [
    'binary_price',
    'implied_vol',
    'implied_vol_approx',
    'implied_vol_newton',
    'option_price',
    'ImpliedVolMethod',
]

remove_undocumented(__name__, _allowed_symbols)
