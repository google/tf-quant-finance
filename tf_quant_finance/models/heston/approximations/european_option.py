# Lint as: python3
# Copyright 2020 Google LLC
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

from tf_quant_finance.math import integration

_PI_ = np.pi
_COMPOSITE_SIMPSONS_RULE = integration.IntegrationMethod.COMPOSITE_SIMPSONS_RULE


def european_option_price(*,
                          strikes=None,
                          expiries=None,
                          is_call_options=None,
                          variances=None,
                          kappas=None,
                          thetas=None,
                          sigmas=None,
                          rhos=None,
                          spots=None,
                          forwards=None,
                          discount_rates=None,
                          continuous_dividends=None,
                          cost_of_carries=None,
                          discount_factors=None,
                          integration_method=None,
                          dtype=None,
                          name=None,
                          **kwargs):
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
    dV = kappa * (theta - V) * dt * sigma * sqrt(V) * dW_2
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
      kappas=2.0,
      thetas=0.5,
      sigmas=0.15,
      rhos=0.3,
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
    is_call_options: A boolean `Tensor` of a shape compatible with
      `strikes`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    variances: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The initial value of the variance.
    kappas: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The mean reversion strength of the variance square root
      process.
    thetas: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The mean reversion level of the variance square root process.
    sigmas: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The volatility of the variance square root process (volatility
      of volatility)
    rhos: A real `Tensor` of the same dtype and compatible shape as
      `strikes`. The correlation between spot and variance.
        spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `strikes`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    discount_rates: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `strikes`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      discount_rates and discount_factors can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      discount_factors also not given.
    continuous_dividends: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `strikes`.
      If not `None`, `cost_of_carries` is calculated as r - q,
      where r are the `discount_rates` and q is `continuous_dividends`. Either
      this or `cost_of_carries` can be given.
      Default value: `None`, equivalent to q = 0.
    cost_of_carries: An optional real `Tensor` of same dtype as the
      `strikes` and of the shape that broadcasts with `strikes`.
      Cost of storing a physical commodity, the cost of interest paid when
      long, or the opportunity cost, or the cost of paying dividends when short.
      If not `None`, and `spots` is supplied, used to calculate forwards from
      `spots`: F = e^(bT) * S, where F is the forwards price, b is the cost of
      carries, T is expiries and S is the spot price. If `None`, value assumed
      to be equal to the `discount_rate` - `continuous_dividends`
      Default value: `None`, equivalent to b = r.
    discount_factors: An optional real `Tensor` of same dtype as the
      `strikes`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with discount_rate and cost_of_carry.
      If neither is given, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not `None` then this is also used to compute the forwards to expiry.
      At most one of discount_rates and discount_factors can be supplied.
      Default value: `None`, which maps to -log(discount_factors) / expiries
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
  if (continuous_dividends is not None) and (cost_of_carries is not None):
    raise ValueError('At most one of continuous_dividends and cost_of_carries '
                     'may be supplied')

  with tf.compat.v1.name_scope(name, default_name='eu_option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    kappas = tf.convert_to_tensor(kappas, dtype=dtype, name='kappas')
    thetas = tf.convert_to_tensor(thetas, dtype=dtype, name='thetas')
    sigmas = tf.convert_to_tensor(sigmas, dtype=dtype, name='sigmas')
    rhos = tf.convert_to_tensor(rhos, dtype=dtype, name='rhos')
    variances = tf.convert_to_tensor(variances, dtype=dtype, name='variances')
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')

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

    if continuous_dividends is None:
      continuous_dividends = tf.convert_to_tensor(
          0.0, dtype=dtype, name='continuous_dividends')

    if cost_of_carries is not None:
      cost_of_carries = tf.convert_to_tensor(
          cost_of_carries, dtype=dtype, name='cost_of_carries')
    else:
      cost_of_carries = discount_rates - continuous_dividends

    if discount_factors is None:
      discount_factors = tf.exp(-discount_rates * expiries)  # pylint: disable=invalid-unary-operand-type

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots * tf.exp(cost_of_carries * expiries)

    # Cast as complex for the characteristic function calculation
    expiries_real = tf.complex(expiries, tf.zeros_like(expiries))
    kappas_real = tf.complex(kappas, tf.zeros_like(kappas))
    thetas_real = tf.complex(thetas, tf.zeros_like(thetas))
    sigmas_real = tf.complex(sigmas, tf.zeros_like(sigmas))
    rhos_real = tf.complex(rhos, tf.zeros_like(rhos))
    variances_real = tf.complex(variances, tf.zeros_like(variances))

    # Prepare inputs to build an integrand_function
    expiries_real = tf.expand_dims(expiries_real, -1)
    kappas_real = tf.expand_dims(kappas_real, -1)
    thetas_real = tf.expand_dims(thetas_real, -1)
    sigmas_real = tf.expand_dims(sigmas_real, -1)
    rhos_real = tf.expand_dims(rhos_real, -1)
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
      s = rhos_real * sigmas_real * u_imag
      # TODO(b/156221007): investigate why s_kappa = (s - kappas_real)**2 leads
      # to a wrong result in graph mode.
      s_kappa = (s - kappas_real) * s - (s - kappas_real) * kappas_real
      d = s_kappa - sigmas_real ** 2 * (-u_imag - u_real ** 2)
      d = tf.math.sqrt(d)
      g = (kappas_real - s - d) / (kappas_real - s + d)
      a = kappas_real * thetas_real
      h = g * tf.math.exp(-d * expiries_real)
      m = 2 * tf.math.log((1 - h) / (1 - g))
      c = (a / sigmas_real ** 2) * ((kappas_real - s - d) * expiries_real - m)
      e = (1 - tf.math.exp(-d * expiries_real))
      d_new = (kappas_real - s - d) / sigmas_real ** 2 * (e / (1 - h))
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
