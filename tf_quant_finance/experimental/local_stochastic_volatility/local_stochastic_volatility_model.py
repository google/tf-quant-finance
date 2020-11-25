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

"""Local Stochastic Volatility process."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.models import generic_ito_process


class LocalStochasticVolatilityModel(generic_ito_process.GenericItoProcess):
  r"""Local stochastic volatility model.

  Local stochastic volatility (LSV) models assume that the spot price of an
  asset follows the following stochastic differential equation under the risk
  neutral measure [1]:

  ```None
    dS(t) / S(t) =  (r - d) dt + sqrt(v(t)) * L(t, S(t)) * dW_s(t)
    dv(t) = a(v(t)) dt + b(v(t)) dW_v(t)
    E[dW_s(t)dW_v(t)] = rho dt
  ```
  where `r` and `d` denote the risk free interest rate and dividend yield
  respectively. `S(t)` is the spot price, `v(t)` denotes the stochastic variance
  and the function `L(t, S(t))`  is the leverage function which is calibrated
  using the volatility smile data. The functions `a(v(t))` and `b(v(t))` denote
  the drift and volitility of the stochastic process for the variance and `rho`
  denotes the instantabeous correlation between the spot and the variance
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

  The `LocalStochasticVolatilityModel` class contains a generic implementation
  of the LSV model with the flexibility to specify an arbitrary variance
  process. The default variance process is a Heston type process with
  mean-reverting variance (as in Ref. [1]):

  ```
  dv(t) = k(m - v(t)) dt + alpha*sqrt(v(t)) dW_v(t)
  ```

  #### References:
    [1]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
    guide. Chapter 5. 2011.
    [2]: I. Gyongy. Mimicking the one-dimensional marginal distributions of
    processes having an ito differential. Probability Theory and Related
    Fields, 71, 1986.
  """

  def __init__(self,
               leverage_fn,
               variance_process,
               corr_matrix=None,
               dtype=None,
               name=None):
    """Initializes the Local stochastic volatility model.

    Args:
      leverage_fn: A Python callable which returns the Leverage function
        `L(t, S(t))` as a function of state and time. The function must accept
        a scalar `Tensor` corresponding to time 't' and a real `Tensor` of shape
        `[num_samples, 1]` corresponding to the underlying price (S) as
        inputs  and return a real `Tensor` containing the leverage function
        computed at (S,t).
      variance_process: An instance of `LSVVarianceModel` specifying the
        dynamics of the variance process of the LSV model.
      corr_matrix: A real `Tensor` of shape `[variance_dim+1,variance_dim+1]`
        specifying the correlation between the underlying and the variance
        process. `variance_dim` is the dimensionality of the variance process.
        Default value: `None` in which case the correlation is assumed to be
        zero.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        `local_stochastic_volatility_model`.
    """
    self._name = name or "local_stochastic_volatility_model"
    with tf.name_scope(self._name):
      self._dtype = dtype or tf.convert_to_tensor([0.0]).dtype

      def _vol_fn(t, state):
        """Volatility function of LSV model."""
        del t, state

      # Drift function
      def _drift_fn(t, state):
        """Drift function of LSV model."""
        del t, state

      super(LocalStochasticVolatilityModel, self).__init__(
          1 + variance_process.dim, _drift_fn, _vol_fn, self._dtype,
          self._name)

  @classmethod
  def from_market_data(
      cls, variance_process, valuation_date, expiry_dates, strikes,
      implied_volatilities, spot, risk_free_rate=None, dividend_yield=None,
      dtype=None):
    """Creates a `LocalStochsticVolatilityModel` from market data."""

  @classmethod
  def from_volatility_surface(
      cls, variance_process, spot, implied_volatility_surface,
      risk_free_rate=None, dividend_yield=None, dtype=None):
    """Creates a `LocalStochasticVolatilityModel` from volatility surface."""
