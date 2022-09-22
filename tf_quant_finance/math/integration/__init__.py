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

"""Numerical integration methods."""

from tf_quant_finance.math.integration.gauss_legendre import gauss_legendre
from tf_quant_finance.math.integration.integrate import integrate
from tf_quant_finance.math.integration.integrate import IntegrationMethod
from tf_quant_finance.math.integration.simpson import simpson
from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'integrate',
    'simpson',
    'gauss_legendre',
    'IntegrationMethod',
]

remove_undocumented(__name__, _allowed_symbols)
