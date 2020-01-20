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

# Lint as: python2, python3
"""Functions to handle rates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_quant_finance.rates import cashflows
from tf_quant_finance.rates import forwards
from tf_quant_finance.rates import hagan_west
from tf_quant_finance.rates import constant_fwd
from tf_quant_finance.rates import swap_curve_fit as swap_curve_fit_lib

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

swap_curve_fit = swap_curve_fit_lib.swap_curve_fit
SwapCurveBuilderResult = swap_curve_fit_lib.SwapCurveBuilderResult

_allowed_symbols = [
    'cashflows',
    'forwards',
    'hagan_west',
    'constant_fwd',
    'swap_curve_fit',
    'SwapCurveBuilderResult',
]

remove_undocumented(__name__, _allowed_symbols)
