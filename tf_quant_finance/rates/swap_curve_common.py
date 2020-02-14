# Lint as: python3
# Copyright 2020 Google LLC
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
"""Common utilities and data structures for swap curve construction."""

import collections

SwapCurveBuilderResult = collections.namedtuple(
    'SwapCurveBuilderResult',
    [
        # Rank 1 real `Tensor`. Times for the computed rates.
        'times',
        # Rank 1 `Tensor` of the same dtype as `times`.
        # The inferred zero rates.
        'rates',
        # Rank 1 `Tensor` of the same dtype as `times`.
        # The inferred discount factors.
        'discount_factors',
        # Rank 1 `Tensor` of the same dtype as `times`. The
        # initial guess for the rates.
        'initial_rates',
        # Scalar boolean `Tensor`. Whether the procedure converged.
        'converged',
        # Scalar boolean `Tensor`. Whether the procedure failed.
        'failed',
        # Scalar int32 `Tensor`. Number of iterations performed.
        'iterations',
        # Scalar real `Tensor`. The objective function at the optimal soultion.
        'objective_value'
    ])
