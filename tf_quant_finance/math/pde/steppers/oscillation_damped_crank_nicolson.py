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
"""Crank-Nicolson with oscillation damping time marching scheme."""

from tf_quant_finance.math.pde.steppers.composite_stepper import composite_scheme_step
from tf_quant_finance.math.pde.steppers.crank_nicolson import crank_nicolson_scheme
from tf_quant_finance.math.pde.steppers.extrapolation import extrapolation_scheme


def oscillation_damped_crank_nicolson_step(extrapolation_steps=1):
  """Scheme similar to Crank-Nicolson, but ensuring damping of oscillations.

  Performs first (or first few) steps with Extrapolation scheme, then proceeds
  with Crank-Nicolson scheme. This combines absence of oscillations by virtue
  of Extrapolation scheme with lower computational cost of Crank-Nicolson
  scheme.

  See [1], [2] ([2] mostly discusses using fully implicit scheme on the first
  step, but mentions using extrapolation scheme for better accuracy in the end).

  #### References:
  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods for
    Parabolic Partial Differential Equations. I. 1978 SIAM Journal on Numerical
    Analysis. 15. 1212-1224.
    https://epubs.siam.org/doi/abs/10.1137/0715082
  [2]: B. Giles, Michael & Carter, Rebecca. Convergence analysis of
    Crank-Nicolson and Rannacher time-marching. J. Comput. Finance. 9. 2005.
    https://core.ac.uk/download/pdf/1633712.pdf

  Args:
    extrapolation_steps: number of first steps to which to apply the
      Extrapolation scheme. Defaults to `1`.

  Returns:
    Callable to use as `one_step_fn` in fd_solvers.
  """
  return composite_scheme_step(extrapolation_steps, extrapolation_scheme,
                               crank_nicolson_scheme)

__all__ = ["oscillation_damped_crank_nicolson_step"]
