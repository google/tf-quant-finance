# Copyright 2021 Google LLC
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
"""Approximate analytic solutions to the Sabr model."""

from tf_quant_finance.models.sabr.approximations.calibration import calibration
from tf_quant_finance.models.sabr.approximations.european_options import option_price as european_option_price
from tf_quant_finance.models.sabr.approximations.implied_volatility import implied_volatility
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrApproximationType
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrImpliedVolatilityType
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'calibration',
    'european_option_price',
    'implied_volatility',
    'SabrApproximationType',
    'SabrImpliedVolatilityType',
]

remove_undocumented(__name__, _allowed_symbols)
