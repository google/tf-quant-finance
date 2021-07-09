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

from tf_quant_finance import types

__all__ = [
    'total_variance_from_raw_svi_parameters',
    'implied_volatility_from_raw_svi_parameters'
]


def total_variance_from_raw_svi_parameters(
    *,
    svi_parameters: types.RealTensor,
    log_moneyness: types.RealTensor = None,
    forwards: types.RealTensor = None,
    strikes: types.RealTensor = None,
    dtype: tf.DType = None,
    name: str = None) -> types.RealTensor:
  r"""Computes modeled total variance using raw SVI parameters.

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

  #### Example

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  svi_parameters = np.array([-0.1825, 0.3306, -0.0988, 0.0368, 0.6011])

  forwards = np.array([2402.])
  strikes = np.array([[1800., 2000., 2200., 2400., 2600., 2800., 3000.]])

  total_var = tff.experimental.svi.total_variance_from_raw_svi_parameters(
      svi_parameters=svi_parameters, forwards=forwards, strikes=strikes)

  # Expected: total_var tensor (rounded to 4 decimal places) should contain
  # [[0.0541, 0.0363, 0.02452, 0.0178, 0.0153, 0.0161, 0.0194]]
  ```

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
      The log-moneyness of the option.
    forwards: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      forward price of the option at expiry.
    strikes: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      option's strike price.
    dtype: Optional `tf.Dtype`. If supplied, the dtype for the input and output
      `Tensor`s will be converted to this.
      Default value: `None` which maps to the dtype inferred from
        `log_moneyness`.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to `svi_total_variance`.

  Returns:
    A rank 2 real `Tensor` of shape [batch_size, num_strikes].

  Raises:
    ValueError: If exactly one of `forwards` and `strikes` is supplied.
    ValueError: If both `log_moneyness' and `forwards` are supplied or if
    neither is supplied.
  """

  if (strikes is None) != (forwards is None):
    raise ValueError(
        'Either both `forwards` and `strikes` must be supplied, or neither.')
  if (log_moneyness is None) == (forwards is None):
    raise ValueError(
        'Exactly one of `log_moneyness` or `forwards` must be provided.')

  name = name or 'svi_total_variance'
  with tf.name_scope(name):
    if log_moneyness is None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
      strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
      log_moneyness = tf.math.log(strikes / forwards)
    else:
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


def implied_volatility_from_raw_svi_parameters(
    *,
    svi_parameters: types.RealTensor,
    log_moneyness: types.RealTensor = None,
    forwards: types.RealTensor = None,
    strikes: types.RealTensor = None,
    expiries: types.RealTensor = None,
    dtype: tf.DType = None,
    name: str = None) -> types.RealTensor:
  r"""Computes modeled implied volatility using raw SVI parameters.

  The SVI volatility model parameterizes an option's total implied variance. For
  a fixed timeslice (i.e. given expiry t), raw SVI parameters (a,b,rho,m,sigma)
  and option's log-moneyness k:=log(K/F), the modeled total variance is
  ```None
  w(k) = a + b * (rho * (k - m) + sqrt{(k - m)^2 + sigma^2)}
  ```

  The modeled Black-Scholes implied volatility sigmaBS is computed from w(k)
  and the option's expiry t from the equation
  ```None
  w(k,t) = sigmaBS(k,t)^2 * t
  ```

  See [1] and documentation for `total_variance_from_raw_svi_parameters` for
  additional details.

  #### Example

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  svi_parameters = np.array([-0.1825, 0.3306, -0.0988, 0.0368, 0.6011])

  forwards = np.array([2402.])
  expiries = np.array([0.23])
  strikes = np.array([[1800., 2000., 2200., 2400., 2600., 2800., 3000.]])

  implied_vol = tff.experimental.svi.implied_volatility_from_raw_svi_parameters(
      svi_parameters=svi_parameters,
      forwards=forwards,
      strikes=strikes,
      expiries=expiries)

  # Expected: implied_vol tensor (rounded to 4 decimal places) should contain
  # [[0.4849, 0.3972, 0.3265, 0.2785, 0.2582, 0.2647, 0.2905]]
  ```

  #### References:
  [1] Gatheral J., Jaquier A., Arbitrage-free SVI volatility surfaces.
  https://arxiv.org/pdf/1204.0646.pdf

  Args:
    svi_parameters: A rank 2 real `Tensor` of shape [batch_size, 5]. The raw SVI
      parameters for each volatility skew.
    log_moneyness: A rank 2 real `Tensor` of shape [batch_size, num_strikes].
      The log-moneyness of the options.
    forwards: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      forward prices of the options at expiries.
    strikes: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
      options strike prices.
    expiries: A rank 1 real `Tensor` of shape [batch_size]. The options
      expiries.
    dtype: Optional `tf.Dtype`. If supplied, the dtype for the input and output
      `Tensor`s will be converted to this.
      Default value: `None` which maps to the dtype inferred from
        `log_moneyness`.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to `svi_implied_volatility`.

  Returns:
    A rank 2 real `Tensor` of shape [batch_size, num_strikes].

  Raises:
    ValueError: If exactly one of `forwards` and `strikes` is supplied.
    ValueError: If both `log_moneyness' and `forwards` are supplied or if
    neither is supplied.
  """
  name = name or 'svi_implied_volatility'
  with tf.name_scope(name):
    total_variance = total_variance_from_raw_svi_parameters(
        svi_parameters=svi_parameters,
        log_moneyness=log_moneyness,
        forwards=forwards,
        strikes=strikes,
        name=name)
    dtype = dtype or total_variance.dtype
    expiries = tf.convert_to_tensor(expiries, dtype, name='expiries')
    implied_volatilities = tf.math.sqrt(total_variance / expiries[:, None])
    return implied_volatilities
