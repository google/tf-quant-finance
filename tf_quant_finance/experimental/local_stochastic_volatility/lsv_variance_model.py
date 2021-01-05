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
"""Variance process for LSV models."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.models import generic_ito_process


class LSVVarianceModel(generic_ito_process.GenericItoProcess):
  r"""Implements Heston like variance process for the LSV models.

  Local stochastic volatility (LSV) models assume that the spot price of an
  asset follows the following stochastic differential equation under the risk
  neutral measure [1]:

  ```None
    dS / S =  (r - d) dt + sqrt(v) * L(t, S(t)) * dW_s
    dv = a(t, v) dt + b(t, v) dW_v
    E[dW_s * dW_v] = rho dt
  ```
  where `r` and `d` denote the risk free interest rate and dividend yield
  respectively. `S` is the spot price, `v` denotes the stochastic variance
  and the function `L(t, S)` is the leverage function which is calibrated
  using the volatility smile data. The functions `a(t, v)` and `b(t, v)` denote
  the drift and volitility of the stochastic process for the variance and `rho`
  denotes the instantaneous correlation between the spot and the variance
  process. LSV models thus combine the local volatility dynamics with
  stochastic volatility.

  Using the relationship between the local volatility and the expectation of
  future instantaneous variance, leverage function can be computed as follows
  [2]:

  ```
  sigma(T,K)^2 = L(T,K)^2 * E[v(T)|S(T)=K]
  ```
  where the local volatility function `sigma(T,K)` can be computed using the
  Dupire's formula.

  The `LSVVarianceModel` class implements Heston like mean-reverting model
  for the dynamics of the variance process. The variance model dynamics is
  as follows:

  ```None
    dv = k * (m - v) dt + alpha * sqrt(v) * dW
  ```

  #### References:
    [1]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
    guide. Chapter 5. 2011.
    [2]: I. Gyongy. Mimicking the one-dimensional marginal distributions of
    processes having an ito differential. Probability Theory and Related
    Fields, 71, 1986.
  """

  def __init__(self, k, m, alpha, dtype=None):
    """Initializes the variance model."""
    self._k = tf.convert_to_tensor(k, dtype=dtype)
    self._dtype = dtype or self._k.dtype
    self._m = tf.convert_to_tensor(m, dtype=dtype)
    self._alpha = tf.convert_to_tensor(alpha, dtype=dtype)

    def _vol_fn(t, state):
      """Volatility function of LSV model."""
      del t
      return self._alpha * tf.math.sqrt(state)

    # Drift function
    def _drift_fn(t, state):
      """Drift function of LSV model."""
      del t
      return self._k * (self._m - state)

    # TODO(b/175878101): Rework to variance model to make it more specific.
    super(LSVVarianceModel, self).__init__(1, _drift_fn, _vol_fn, self._dtype)
