# Lint as: python3
# Copyright 2021 Google LLC
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
"""Parameterization utilities for the SVI volatility model."""

import tensorflow.compat.v2 as tf

__all__ = ['total_variance_from_raw']


def total_variance_from_raw(svi_parameters,
                            log_moneyness,
                            dtype=None,
                            name=None):
  r"""Computes modeled total variance from raw SVI parameters.

  The SVI volatility model parameterizes an option's total implied variance,
  defined as w(k,t) := sigmaBS(k,t)^2 * t, where k := log(K/F) is the options's
  log-moneyness, t is the time to expiry, and sigmaBS(k,t) is the Black-Scholes
  market implied volatility. For a fixed timeslice (i.e. given expiry t), the
  raw SVI parameterization consists of 5 parameters (a,b,rho,m,sigma), and
  the model approximation formula for w(k,t) as a function of k is (cf.[1]):
  ```None
  w(k) = a + b * (rho * (k - m) + sqrt{(k - m)^2 + sigma^2)}
  ```
  The raw parameters have the following interpretations (cf.[2]):
  a      vertically shifts the variance graph
  b      controls the angle between the left and right asymptotes
  rho    controls the rotation of the variance graph
  m      horizontally shifts the variance graph
  sigma  controls the graph smoothness at the vertex (ATM)

  #### References:
  [1] Gatheral J., Jaquier A., Arbitrage-free SVI volatility surfaces.
  https://arxiv.org/pdf/1204.0646.pdf
  [2] Gatheral J, A parsimonious arbitrage-free implied volatility
  parameterization with application to the valuation of volatility derivatives.
  http://faculty.baruch.cuny.edu/jgatheral/madrid2004.pdf

  Args:
    svi_parameters: A rank 2 real `Tensor` of shape [batch_size, 5]. The raw SVI
      parameters for each volatility skew.
    log_moneyness: A rank 2 real `Tensor` of shape [batch_size, num_strikes].
    dtype: Optional `tf.Dtype`. If supplied, the dtype for the input and output
      `Tensor`s will be converted to this.
      Default value: `None` which maps to the dtype inferred from
        `log_moneyness`.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to `svi_parameterization`.

  Returns:
    A rank 2 real `Tensor` of shape [batch_size, num_strikes].
  """
  name = name or 'svi_parameterization'
  with tf.name_scope(name):
    log_moneyness = tf.convert_to_tensor(
        log_moneyness, dtype=dtype, name='log_moneyness')
    dtype = dtype or log_moneyness.dtype

    svi_parameters = tf.convert_to_tensor(
        svi_parameters, dtype=dtype, name='svi_parameters')
    # Introduce standard aliases, same as in [1]. Keep the length 1 rightmost
    # dimension of SVI parameters for broadcasting compatibility in the formula.
    a = svi_parameters[..., 0:1]
    b = svi_parameters[..., 1:2]
    rho = svi_parameters[..., 2:3]
    m = svi_parameters[..., 3:4]
    sigma = svi_parameters[..., 4:5]
    k = log_moneyness

    return a + b * (rho * (k - m) + tf.sqrt((k - m)**2 + sigma**2))
