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
"""Method for semi-analytical Heston option price."""

import numpy as np
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.math import integration

__all__ = [
    'european_option_price'
]

_PI_ = np.pi
_COMPOSITE_SIMPSONS_RULE = integration.IntegrationMethod.COMPOSITE_SIMPSONS_RULE


def european_option_price(
    *,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    spots: types.RealTensor = None,
    forwards: types.RealTensor = None,
    is_call_options: types.BoolTensor = None,
    discount_rates: types.RealTensor = None,
    dividend_rates: types.RealTensor = None,
    discount_factors: types.RealTensor = None,
    variances: types.RealTensor,
    mean_reversion: types.RealTensor,
    theta: types.RealTensor,
    volvol: types.RealTensor,
    rho: types.RealTensor = None,
    integration_method: integration.IntegrationMethod = None,
    dtype: tf.DType = None,
    name: str = None,
    **kwargs) -> types.RealTensor:
  """Calculates European option prices under the Heston model.

  Heston originally published in 1993 his eponymous model [3]. He provided
  a semi- analytical formula for pricing European option via Fourier transform
  under his model. However, as noted by Albrecher [1], the characteristic
  function used in Heston paper can suffer numerical issues because of the
  discontinuous nature of the square root function in the complex plane, and a
  second version of the characteric function which doesn't suffer this
  shortcoming should be used instead. Attari [2] further refined the numerical
  method by reducing the number of numerical integrations (only one Fourier
  transform instead of two) and with an integrand function decaying
  quadratically instead of linearly. Attari's numerical method is implemented
  here.

  Heston model:
  ```
    dF/F = sqrt(V) * dW_1
    dV = mean_reversion * (theta - V) * dt * sigma * sqrt(V) * dW_2
    <dW_1,dW_2> = rho *dt
  ```
  The variance V follows a square root process.

  #### Example
  ```python
  import tf_quant_finance as tff
  import numpy as np
  prices = tff.models.heston.approximations.european_option_price(
      variances=0.11,
      strikes=102.0,
      expiries=1.2,
      forwards=100.0,
      is_call_options=True,
      mean_reversion=2.0,
      theta=0.5,
      volvol=0.15,
      rho=0.3,
      discount_factors=1.0,
      dtype=np.float64)
  # Expected print output of prices:
  # 24.82219619
  ```
  #### References
  [1] Hansjorg Albrecher, The Little Heston Trap
  https://perswww.kuleuven.be/~u0009713/HestonTrap.pdf
  [2] Mukarram Attari, Option Pricing Using Fourier Transforms: A Numerically
  Efficient Simplification
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=520042
  [3] Steven L. Heston, A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options
  http://faculty.baruch.cuny.edu/lwu/890/Heston93.pdf
  Args:
    strikes: A real `Tensor` of any shape and dtype. The strikes of the options
      to be priced.
    expiries: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The expiry of each option.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `strikes`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `strikes`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `strikes`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    discount_rates: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `strikes`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      discount_rates and discount_factors can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      discount_factors also not given.
    dividend_rates: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `strikes`.
      Default value: `None`, equivalent to q = 0.
    discount_factors: An optional real `Tensor` of same dtype as the
      `strikes`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is
      given, no discounting is applied (i.e. the undiscounted option price is
      returned). If `spots` is supplied and `discount_factors` is not `None`
      then this is also used to compute the forwards to expiry. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, which maps to e^(-rT) calculated from
      discount_rates.
    variances: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The initial value of the variance.
    mean_reversion: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The mean reversion strength of the variance square root
      process.
    theta: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The mean reversion level of the variance square root process.
    volvol: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The volatility of the variance square root process (volatility
      of volatility)
    rho: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The correlation between spot and variance.
    integration_method: An instance of `math.integration.IntegrationMethod`.
      Default value: `None` which maps to the Simpsons integration rule.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name
      `heston_price`.
    **kwargs: Additional parameters for the underlying integration method.
      If not supplied and `integration_method` is Simpson, then uses
      `IntegrationMethod.COMPOSITE_SIMPSONS_RULE` with `num_points=1001`, and
      bounds `lower=1e-9`, `upper=100`.
  Returns:
    A `Tensor` of the same shape as the input data which is the price of
    European options under the Heston model.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')

  with tf.compat.v1.name_scope(name, default_name='eu_option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    mean_reversion = tf.convert_to_tensor(mean_reversion, dtype=dtype,
                                          name='mean_reversion')
    theta = tf.convert_to_tensor(theta, dtype=dtype, name='theta')
    volvol = tf.convert_to_tensor(volvol, dtype=dtype, name='volvol')
    rho = tf.convert_to_tensor(rho, dtype=dtype, name='rho')
    variances = tf.convert_to_tensor(variances, dtype=dtype, name='variances')

    if discount_factors is not None:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')

    if discount_rates is not None:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
    elif discount_factors is not None:
      discount_rates = -tf.math.log(discount_factors) / expiries
    else:
      discount_rates = tf.convert_to_tensor(
          0.0, dtype=dtype, name='discount_rates')

    if dividend_rates is None:
      dividend_rates = 0.0
    dividend_rates = tf.convert_to_tensor(
        dividend_rates, dtype=dtype, name='dividend_rates')

    if discount_factors is None:
      discount_factors = tf.exp(-discount_rates * expiries)  # pylint: disable=invalid-unary-operand-type

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      cost_of_carries = discount_rates - dividend_rates
      forwards = spots * tf.exp(cost_of_carries * expiries)

    # Cast as complex for the characteristic function calculation
    expiries_real = tf.complex(expiries, tf.zeros_like(expiries))
    mean_reversion_real = tf.complex(mean_reversion,
                                     tf.zeros_like(mean_reversion))
    theta_real = tf.complex(theta, tf.zeros_like(theta))
    volvol_real = tf.complex(volvol, tf.zeros_like(volvol))
    rho_real = tf.complex(rho, tf.zeros_like(rho))
    variances_real = tf.complex(variances, tf.zeros_like(variances))

    # Prepare inputs to build an integrand_function
    expiries_real = tf.expand_dims(expiries_real, -1)
    mean_reversion_real = tf.expand_dims(mean_reversion_real, -1)
    theta_real = tf.expand_dims(theta_real, -1)
    volvol_real = tf.expand_dims(volvol_real, -1)
    rho_real = tf.expand_dims(rho_real, -1)
    variances_real = tf.expand_dims(variances_real, -1)
    if integration_method is None:
      integration_method = _COMPOSITE_SIMPSONS_RULE
    if integration_method == _COMPOSITE_SIMPSONS_RULE:
      if 'num_points' not in kwargs:
        kwargs['num_points'] = 1001
      if 'lower' not in kwargs:
        kwargs['lower'] = 1e-9
      if 'upper' not in kwargs:
        kwargs['upper'] = 100
    def char_fun(u):
      # Using 'second formula' for the (first) characteristic function of
      # log( spot_T / forwards )
      # (noted 'phi_2' in 'The Little Heston Trap', (Albrecher))
      u_real = tf.complex(u, tf.zeros_like(u))
      u_imag = tf.complex(tf.zeros_like(u), u)
      s = rho_real * volvol_real * u_imag
      # TODO(b/156221007): investigate why
      # s_mean_reversion = (s - mean_reversion_real)**2 leads to a wrong result
      # in graph mode.
      s_mean_reversion = ((s - mean_reversion_real) * s
                          - (s - mean_reversion_real) * mean_reversion_real)
      d = s_mean_reversion - volvol_real ** 2 * (-u_imag - u_real ** 2)
      d = tf.math.sqrt(d)
      g = (mean_reversion_real - s - d) / (mean_reversion_real - s + d)
      a = mean_reversion_real * theta_real
      h = g * tf.math.exp(-d * expiries_real)
      m = 2 * tf.math.log((1 - h) / (1 - g))
      c = (a / volvol_real ** 2) * ((mean_reversion_real - s - d)
                                    * expiries_real - m)
      e = (1 - tf.math.exp(-d * expiries_real))
      d_new = (mean_reversion_real - s - d) / volvol_real ** 2 * (e / (1 - h))
      return tf.math.exp(c + d_new * variances_real)

    def integrand_function(u, k):
      # Note that with [2], integrand is in 1 / u**2,
      # which converges faster than Heston 1993 (which is in 1 /u)
      char_fun_complex = char_fun(u)
      char_fun_real_part = tf.math.real(char_fun_complex)
      char_fun_imag_part = tf.math.imag(char_fun_complex)

      a = (char_fun_real_part + char_fun_imag_part / u) * tf.math.cos(u * k)
      b = (char_fun_imag_part - char_fun_real_part / u) * tf.math.sin(u * k)

      return (a + b) / (1.0 + u * u)

    k = tf.expand_dims(tf.math.log(strikes / forwards), axis=-1)

    integral = integration.integrate(
        lambda u: integrand_function(u, k),
        method=integration_method,
        dtype=dtype,
        **kwargs)
    undiscounted_call_prices = forwards - strikes * (0.5 + integral / _PI_)

    if is_call_options is None:
      return undiscounted_call_prices * discount_factors
    else:
      is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool,
                                             name='is_call_options')
      # Use call-put parity for Put
      undiscounted_put_prices = undiscounted_call_prices - forwards + strikes

      undiscount_prices = tf.where(
          is_call_options,
          undiscounted_call_prices,
          undiscounted_put_prices)
      return undiscount_prices * discount_factors
