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

"""Local Volatility Model."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.models import generic_ito_process


class LocalVocalityModel(generic_ito_process.GenericItoProcess):
  r"""Local volatility model for smile modelling.

  Local volatility (LV) model specifies that the dynamics of an asset is
  governed by the following stochastic differential equation:

  ```None
    dS(t) =  mu(t, S(t)) dt + sigma(t, S(t)) * dW(t)
  ```
  where `mu(t, S(t))` is the drift and `sigma(t, S(t))` is the instantaneous
  volatility. The local volatility function `sigma(t, S(t))` is state dependent
  and is computed by caibrating against a given implied volatility surface
  `sigma_iv(T, K)` using the Dupire's formula [1]:

  ```
  sigma(T,K)^2 = 2 * (dC(T,K)/dT + (r - q)K dC(T,K)/dK + qC(T,K)) /
                     (K^2 d2C(T,K)/dK2)
  ```
  where the derivatives above are the partial derivatives. The LV model provides
  a flexible framework to model any (arbitrage free) volatility surface.

  #### References:
    [1]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
    guide. Chapter 5. 2011.
  """

  def __init__(self,
               dim,
               local_volatility_fn=None,
               corr_matrix=None,
               dtype=None,
               name=None):
    """Initializes the Local volatility model.

    Args:
      dim: A Python scalar which corresponds to the number of underlying assets
        comprising the model.
      local_volatility_fn: A Python callable which returns instantaneous
        volatility as a function of state and time. The function must accept two
        real `Tensor` inputs to shape `[dim,]` corresponding to the time (t)
        and state (S) and returns a real `Tensor` of same shape containing the
        local volatility computed at (t,S).
      corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as
        `expiries`. Corresponds to the instantaneous correlation between the
        underlying assets `Rho`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
        `local_volatility_model`.
    """
    self._name = name or 'local_volatility_model'
    self._local_volatility_fn = local_volatility_fn

    with tf.name_scope(self._name):

      def _vol_fn(t, state):
        """Volatility function of LV model."""
        return self._local_volatility_fn(t, state)

      # Drift function
      def _drift_fn(t, state):
        """Drift function of LV model."""
        del t, state
        return None

      super(LocalVocalityModel, self).__init__(
          dim, _drift_fn, _vol_fn, dtype, name)

  @classmethod
  def from_market_data(cls, volatility_surface):
    """Create `LocalVolatilityModel` from implied volatility data."""
    return None
