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
"""Steppers for PDE solvers."""

from tf_quant_finance.math.pde.steppers import composite_stepper
from tf_quant_finance.math.pde.steppers import crank_nicolson
from tf_quant_finance.math.pde.steppers import douglas_adi
from tf_quant_finance.math.pde.steppers import explicit
from tf_quant_finance.math.pde.steppers import extrapolation
from tf_quant_finance.math.pde.steppers import implicit
from tf_quant_finance.math.pde.steppers import multidim_parabolic_equation_stepper
from tf_quant_finance.math.pde.steppers import oscillation_damped_crank_nicolson
from tf_quant_finance.math.pde.steppers import parabolic_equation_stepper
from tf_quant_finance.math.pde.steppers import weighted_implicit_explicit

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'composite_stepper',
    'crank_nicolson',
    'douglas_adi',
    'explicit',
    'extrapolation',
    'implicit',
    'multidim_parabolic_equation_stepper',
    'oscillation_damped_crank_nicolson',
    'parabolic_equation_stepper',
    'weighted_implicit_explicit',
]

remove_undocumented(__name__, _allowed_symbols)
