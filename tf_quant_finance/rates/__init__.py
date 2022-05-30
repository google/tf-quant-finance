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
"""Functions to handle rates."""
import copy

from tf_quant_finance.rates import analytics
from tf_quant_finance.rates import constant_fwd
from tf_quant_finance.rates import forwards
from tf_quant_finance.rates import hagan_west
from tf_quant_finance.rates import nelson_seigel_svensson
from tf_quant_finance.rates import swap_curve_bootstrap as swap_curve_boot_lib
from tf_quant_finance.rates import swap_curve_fit as swap_curve_fit_lib
from tf_quant_finance.rates.swap_curve_common import SwapCurveBuilderResult
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

swap_curve_fit = swap_curve_fit_lib.swap_curve_fit
swap_curve_bootstrap = swap_curve_boot_lib.swap_curve_bootstrap

_allowed_symbols = [
    'forwards',
    'analytics',
    'hagan_west',
    'constant_fwd',
    'swap_curve_fit',
    'swap_curve_bootstrap',
    'nelson_seigel_svensson',
    'SwapCurveBuilderResult',
]

remove_undocumented(__name__, _allowed_symbols)
