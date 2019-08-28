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
"""PDE time marching schemes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_scheme import TimeMarchingScheme
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_schemes import crank_nicolson_scheme
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_schemes import crank_nicolson_with_oscillation_damping
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_schemes import explicit_scheme
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_schemes import ExtrapolationMarchingScheme
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_schemes import implicit_scheme
from tf_quant_finance.math.pde.time_marching_schemes.pde_time_marching_schemes import WeightedImplicitExplicitScheme

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'TimeMarchingScheme', 'crank_nicolson_scheme',
    'crank_nicolson_with_oscillation_damping', 'explicit_scheme',
    'ExtrapolationMarchingScheme', 'implicit_scheme',
    'WeightedImplicitExplicitScheme'
]

remove_undocumented(__name__, _allowed_symbols)
