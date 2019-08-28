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
"""PDE solver methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_quant_finance.math.pde import grids
from tf_quant_finance.math.pde import time_marching_schemes
from tf_quant_finance.math.pde.grid_stepper import BackwardGridStepper
from tf_quant_finance.math.pde.grid_stepper import GridStepperState
from tf_quant_finance.math.pde.pde_kernels import ParabolicDifferentialEquationSolver

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'BackwardGridStepper',
    'GridStepperState',
    'grids',
    'ParabolicDifferentialEquationSolver',
    'time_marching_schemes',
]

remove_undocumented(__name__, _allowed_symbols)
