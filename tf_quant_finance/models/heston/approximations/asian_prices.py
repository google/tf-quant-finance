# Copyright 2022 Google LLC
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
"""Heston prices of a batch of Asian options."""

import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.black_scholes import AveragingFrequency
from tf_quant_finance.black_scholes import AveragingType
from tf_quant_finance.math import integration


__all__ = [
    'asian_option_price'
]


def asian_option_price(
    *,
    variances: types.RealTensor,
    mean_reversion: types.RealTensor,
    theta: types.RealTensor,
    volvol: types.RealTensor,
    rho: types.RealTensor,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    spots: types.RealTensor = None,
    forwards: types.RealTensor = None,
    sampling_times: types.RealTensor = None,
    past_fixings: types.RealTensor = None,
    discount_rates: types.RealTensor = None,
    dividend_rates: types.RealTensor = None,
    discount_factors: types.RealTensor = None,
    is_call_options: types.BoolTensor = None,
    averaging_type: AveragingType = AveragingType.GEOMETRIC,
    averaging_frequency: AveragingFrequency = AveragingFrequency.DISCRETE,
    integration_method: integration.IntegrationMethod = None,
    dtype: tf.DType = None,
    name: str = None,
    **kwargs) -> types.RealTensor:
  """Computes the Heston price for a batch of asian options.

  Discrete and continuous geometric asian options can be priced using a
  semi-analytical expression in the Heston model.

  #### Example
  ```python
    # Price a batch of seasoned discrete geometric asians in Heston model
    # This example reproduces some prices from the reference paper
    variances = 0.09
    mean_reversion = 1.15
    volvol = 0.39
    theta = 0.0348
    rho = -0.64
    spots = 100.0
    strikes = [90.0, 100.0, 110.0]
    discount_rates = 0.05

    T = 0.5
    expiries = T
    sampling_times = np.linspace(1/365, T, 182)[:, np.newaxis]

    computed_prices_asians = asian_option_price(
      variances=variances,
      mean_reversion=mean_reversion,
      theta=theta,
      volvol=volvol,
      rho=rho,
      strikes=strikes,
      expiries=expiries,
      spots=spots,
      sampling_times=sampling_times,
      discount_rates=discount_rates
    )
    # Expected print output of computed prices:
    # [[11.884178 ], [ 5.080889 ], [ 1.3527349]]
  ```

  #### References:
  [1] B. Kim, J. Kim, J. Kim & I. S. Wee, "A Recursive Method for Discretely
    Monitored Geometric Asian Option Prices", Bull. Korean Math. Soc. 53,
    733-749 (2016)

  Args:
    variances: Real `Tensor` of any shape compatible with a `batch_shape` and
      and any real dtype. The initial value of the variance.
    mean_reversion: A real `Tensor` of the same dtype and compatible shape as
      `variances`. Corresponds to the mean reversion rate.
    theta: A real `Tensor` of the same dtype and compatible shape as
      `variances`. Corresponds to the long run price variance.
    volvol: A real `Tensor` of the same dtype and compatible shape as
      `variances`. Corresponds to the volatility of the volatility.
    rho: A real `Tensor` of the same dtype and compatible shape as
      `variances`. Corresponds to the correlation between dW_{X}` and `dW_{V}`.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `variances`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `variances`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `variances`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `variances`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    sampling_times: A real `Tensor` of same dtype as expiries and shape
      `[n] + batch_shape` where `n` is the number of sampling times for
      the asian options
      Default value: `None`, which will raise an error for discrete sampling
      asian options
    past_fixings: A real `Tensor` of same dtype as spots or forwards and shape
      `[n] + batch_shape` where n is the number of past fixings that have
      already been observed
      Default value: `None`, equivalent to no past fixings (ie. unseasoned)
    discount_rates: An optional real `Tensor` of same dtype as the
      `variances` and of the shape that broadcasts with `variances`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      `discount_factors` also not given.
    dividend_rates: An optional real `Tensor` of same dtype as the
      `variances` and of the shape that broadcasts with `variances`.
      Default value: `None`, equivalent to q = 0.
    discount_factors: An optional real `Tensor` of same dtype as the
      `variances`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is
      given, no discounting is applied (i.e. the undiscounted option price is
      returned). If `spots` is supplied and `discount_factors` is not `None`
      then this is also used to compute the forwards to expiry. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, which maps to e^(-rT) calculated from
      discount_rates.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `variances`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    averaging_type: Enum value of AveragingType to select the averaging method
      for the payoff calculation
      Default value: AveragingType.GEOMETRIC
    averaging_frequency: Enum value of AveragingFrequency to select the
      averaging type for the payoff calculation (discrete vs continuous)
      Default value: AveragingFrequency.DISCRETE
    integration_method: An instance of `math.integration.IntegrationMethod`.
      Default value: `None` which maps to Gaussian quadrature.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name
      `asian_option_price`.
    **kwargs: Additional parameters for the underlying integration method.
      If not supplied, uses bounds `lower=1e-9`, `upper=100`.

  Returns:
    option_prices: A `Tensor` of the same shape as `strikes`. The Heston price
    of the options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
    ValueError: If both `discount_rates` and `discount_factors` is supplied.
    ValueError: If option is arithmetic.
    NotImplementedError: if option is continuous averaging.
    NotImplementedError: if any of the Heston model parameters are of type
      PiecewiseConstantFunc.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')
  if averaging_type == AveragingType.ARITHMETIC:
    raise ValueError('Cannot price arithmetic averaging asians analytically '
                     'under Heston model')
  if averaging_frequency == AveragingFrequency.DISCRETE:
    if sampling_times is None:
      raise ValueError('Sampling times required for discrete sampling asians')
  if averaging_frequency == AveragingFrequency.CONTINUOUS:
    raise NotImplementedError('Pricing continuous averaging asians not yet '
                              'supported')

  with tf.name_scope(name or 'asian_option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    results_shape = strikes.shape

    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    sampling_times = tf.convert_to_tensor(
        sampling_times, dtype=dtype, name='sampling_times')

    # Fail assertion if sampling comes after expiry
    assertions = []
    if averaging_frequency == AveragingFrequency.DISCRETE:
      assertions = [tf.debugging.assert_equal(
          tf.math.maximum(sampling_times[-1], expiries), expiries,
          'Sampling times cannot occur after expiry times'
      )]

    variances = tf.convert_to_tensor(variances, dtype=dtype, name='variances')
    mean_reversion = tf.convert_to_tensor(
        mean_reversion, dtype=dtype, name='mean_reversion')
    mean_reversion = tf.broadcast_to(mean_reversion, strikes.shape)
    theta = tf.convert_to_tensor(theta, dtype=dtype, name='theta')
    volvol = tf.convert_to_tensor(volvol, dtype=dtype, name='volvol')
    rho = tf.convert_to_tensor(rho, dtype=dtype, name='rho')

    # Add control dependencies for TF1 compatibility
    with tf.control_dependencies(assertions):
      if sampling_times.shape.rank:
        # In this case sampling_times has some `batch_shape`
        batch_shape = utils.common_shape(
            variances, expiries, theta, volvol, strikes, rho, sampling_times[0])
      else:
        batch_shape = utils.common_shape(
            variances, expiries, theta, volvol, strikes, rho)
      variances = tf.broadcast_to(variances, batch_shape)
      expiries = tf.broadcast_to(expiries, batch_shape)
      theta = tf.broadcast_to(theta, batch_shape)
      volvol = tf.broadcast_to(volvol, batch_shape)
      strikes = tf.broadcast_to(strikes, batch_shape)
      rho = tf.broadcast_to(rho, batch_shape)
      num_sampling_times = utils.get_shape(sampling_times)[0]
      # Broadcast sampling_times to `[num_sampling_times] + batch_shape`
      sampling_times_broadcast_shape = tf.concat(
          [[num_sampling_times], batch_shape], axis=0)
      sampling_times = tf.broadcast_to(
          sampling_times, sampling_times_broadcast_shape)

    if discount_rates is not None:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
      discount_factors = tf.exp(-discount_rates * expiries)
    elif discount_factors is not None:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')
      discount_rates = -tf.math.log(discount_factors) / expiries
    else:
      discount_rates = tf.convert_to_tensor(
          0.0, dtype=dtype, name='discount_rates')
      discount_factors = tf.convert_to_tensor(
          1.0, dtype=dtype, name='discount_factors')

    if dividend_rates is None:
      dividend_rates = tf.convert_to_tensor(
          0.0, dtype=dtype, name='dividend_rates')

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
      spots = forwards * tf.exp(-(discount_rates - dividend_rates) * expiries)
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)

    forwards = tf.broadcast_to(forwards, batch_shape)
    spots = tf.broadcast_to(spots, batch_shape)
    batch_rank = spots.shape.rank

    #  To account for seasoning, we keep pricing time fixed at t=0 and adjust
    #  the pricing parameters as follows:
    #    - calculate prefactor running_prod ^ (1/#_total_fixings)
    #    - strikes \to strikes / prefactor
    #    - k_star \to #_past_fixings
    #    - prepend #_past_fixings negative values to sampling times
    #    - prices \to prefactor * prices
    #
    #  This is slightly different to the approach in the paper, where instead
    #  t is allowed to increase above 0 as pricing time moves forwards

    if past_fixings is None:
      running_accumulator = tf.constant(1.0, dtype=dtype)
      fixing_count = 0
    else:
      past_fixings = tf.convert_to_tensor(
          past_fixings, dtype=dtype, name='past_fixings')
      running_accumulator = tf.reduce_prod(past_fixings, axis=0)
      fixing_count = utils.get_shape(past_fixings)[0]

    start_times = tf.zeros(batch_shape, dtype=dtype)
    # Shape [num_samplint_times + 2] + batch_shape
    tau_k = tf.concat([[start_times], sampling_times, [expiries]], axis=0)

    # n = tf.cast(num_sampling_times + fixing_count, dtype=dtype)
    n = num_sampling_times + fixing_count

    prefactor = running_accumulator ** (1 / tf.cast(n, dtype=dtype))
    adj_strikes = strikes / prefactor

    # Shape [(num_sampling_times + fixing_count), batch_shape] - in this
    # operation we are adding additional negative fixings that account for the
    # already observed fixings of this asian option
    broadcast_shape = tf.concat(
        [[fixing_count], batch_shape], axis=0)
    sampling_times = tf.concat(
        [
            -1 * tf.ones(broadcast_shape, dtype=dtype),
            sampling_times
        ], axis=0)
    k_star = fixing_count

    # Prepare inputs to build an integrand_function
    spots = tf.expand_dims(spots, axis=-1)
    rho = tf.expand_dims(rho, axis=-1)
    variances = tf.expand_dims(variances, axis=-1)
    mean_reversion = tf.expand_dims(mean_reversion, axis=-1)
    volvol = tf.expand_dims(volvol, axis=-1)
    theta = tf.expand_dims(theta, axis=-1)
    expiries = tf.expand_dims(expiries, axis=-1)
    strikes = tf.expand_dims(adj_strikes, axis=-1)
    start_times = tf.expand_dims(start_times, axis=-1)
    discount_rates = tf.expand_dims(discount_rates, axis=-1)
    dividend_rates = tf.expand_dims(dividend_rates, axis=-1)
    prefactor = tf.expand_dims(prefactor, axis=-1)

    d_tau_k = tau_k[1:] - tau_k[:-1]

    k_tensor = tf.range(k_star+1, n+2, dtype=dtype)
    shape_k_tensor = utils.get_shape(k_tensor)

    for _ in range(batch_rank):
      k_tensor = k_tensor[..., tf.newaxis]

    # Shape [k_tensor_size] + batch_shape
    k_tensor = tf.broadcast_to(
        k_tensor, tf.concat([shape_k_tensor, batch_shape], axis=0))

    asian_prices_handler = _AsianPricesHandler(
        spots=tf.cast(spots, tf.complex128),
        discount_rates=tf.cast(discount_rates, tf.complex128),
        dividend_rates=tf.cast(dividend_rates, tf.complex128),
        variances=tf.cast(variances, tf.complex128),
        mean_reversion=tf.cast(mean_reversion, tf.complex128),
        volvol=tf.cast(volvol, tf.complex128),
        rho=tf.cast(rho, tf.complex128),
        theta=tf.cast(theta, tf.complex128),
        strikes=tf.cast(strikes, tf.complex128),
        start_times=tf.cast(start_times, tf.complex128),
        expiries=tf.cast(expiries, tf.complex128),
        sampling_times=tf.cast(sampling_times, tf.complex128),
        k_tensor=tf.cast(k_tensor, tf.complex128),
        d_tau_k=tf.cast(d_tau_k, tf.complex128),
        k_star=k_star,
        n=n,
        batch_shape=batch_shape,
        dtype=tf.complex128
    )

    if integration_method is None:
      integration_method = integration.IntegrationMethod.GAUSS_LEGENDRE

    f = lambda x: tf.cast(asian_prices_handler.integrand(x), dtype)

    if 'lower' not in kwargs:
      kwargs['lower'] = 1e-9
    if 'upper' not in kwargs:
      kwargs['upper'] = 100

    kwargs['lower'] = tf.constant(kwargs['lower'], dtype=dtype)
    kwargs['upper'] = tf.constant(kwargs['upper'], dtype=dtype)

    integral = integration.integrate(
        f, method=integration_method, dtype=dtype, **kwargs)

    asian_forward = tf.cast(
        tf.math.real(asian_prices_handler.phi(1, 0)), dtype=dtype)
    dcfs = tf.math.exp(-discount_rates * expiries)

    # calls currently [`batch_size`, 1], then reshaped to match results_shape
    calls = prefactor * dcfs * ((asian_forward - strikes) / 2
                                + tf.expand_dims(integral, -1) / np.pi)
    calls = tf.reshape(calls, results_shape)

    if is_call_options is None:
      return calls

    is_call_options = tf.convert_to_tensor(
        is_call_options, dtype=bool, name='is_call_options')
    is_call_options = tf.reshape(is_call_options, results_shape)

    # puts currently [`batch_size`, 1], then reshaped to match results_shape
    puts = prefactor * dcfs * ((strikes - asian_forward) / 2
                               + tf.expand_dims(integral, -1) / np.pi)
    puts = tf.reshape(puts, results_shape)

    return tf.where(is_call_options, calls, puts)


@utils.dataclass
class _AsianPricesHandler:
  """Handles the various pricing functions for asian options."""
  spots: types.ComplexTensor
  discount_rates: types.ComplexTensor
  dividend_rates: types.ComplexTensor
  variances: types.ComplexTensor
  mean_reversion: types.ComplexTensor
  theta: types.ComplexTensor
  volvol: types.ComplexTensor
  rho: types.ComplexTensor
  start_times: types.ComplexTensor
  expiries: types.ComplexTensor
  strikes: types.ComplexTensor
  sampling_times: types.ComplexTensor
  k_tensor: types.ComplexTensor
  d_tau_k: types.ComplexTensor
  k_star: int
  n: int
  batch_shape: types.IntTensor
  dtype: tf.DType

  def f(self, z1, z2, tau):
    """Calculates F(z1, z2) (eq. (11) from the referenced paper)."""
    x = tf.math.sqrt(tf.math.pow(self.mean_reversion, 2)
                     - 2 * z1 * tf.math.pow(self.volvol, 2))

    term_1 = tf.math.cosh(0.5 * tau * x)
    term_2 = ((self.mean_reversion - z2 * tf.math.pow(self.volvol, 2))
              * tf.math.sinh(0.5 * tau * x)) / x

    return term_1 + term_2

  def f_tilde(self, z1, z2, tau):
    """Calculates F_tilde(z1, z2) (eq. (11) from the referenced paper)."""
    x = tf.math.sqrt(tf.math.pow(self.mean_reversion, 2)
                     - 2 * z1 * tf.math.pow(self.volvol, 2))

    term_1 = 0.5 * x * tf.math.sinh(0.5 * tau * x)
    term_2 = 0.5 * ((self.mean_reversion - z2 * tf.math.pow(self.volvol, 2))
                    * tf.math.cosh(0.5 * tau * x))

    return term_1 + term_2

  def a(self, s, w):
    """Calculates a(s, w) (eq. (17) from the referenced paper)."""
    raw_drift = self.discount_rates - self.dividend_rates
    corr_drift = self.rho * self.mean_reversion * self.theta / self.volvol
    mod_drift = raw_drift - corr_drift

    # summed_samplings shape [`batch_size`, 1] after the reduction
    summed_samplings = tf.reduce_sum(self.sampling_times[self.k_star:],
                                     axis=0)[..., tf.newaxis]

    n_c = tf.cast(self.n, dtype=self.dtype)
    k_star_c = tf.cast(self.k_star, dtype=self.dtype)

    term_1 = ((s * (n_c - k_star_c) / n_c + w)
              * (tf.math.log(self.spots)
                 - self.rho * self.variances / self.volvol
                 - self.start_times * mod_drift))
    term_2 = (s * (summed_samplings) / n_c + w * self.expiries) * mod_drift

    return term_1 + term_2

  def z(self, s, w):
    """Calculates z_k(s, w) (eq. (14) from the referenced paper)."""
    n_real = tf.cast(self.n, dtype=self.dtype)
    k_s_w_tensor = (tf.expand_dims(n_real - self.k_tensor + 1, axis=-1)
                    * s + n_real * w)
    term_1 = ((2 * self.rho * self.mean_reversion - self.volvol)
              * (k_s_w_tensor) / (self.volvol * n_real))
    term_2 = ((1 - tf.math.pow(self.rho, 2))
              * tf.cast(tf.math.pow(k_s_w_tensor, 2), dtype=self.dtype)
              / tf.cast(tf.math.pow(n_real, 2), dtype=self.dtype))

    return 0.5 * (term_1 + term_2)

  def omega_k(self, s, w):
    """Calculates omega_k(s, w) (eq. (15) from the referenced paper)."""
    n_real = tf.cast(self.n, dtype=self.dtype)
    # Shape batch_shape + [num_sampling_times - fixing_count + 2]
    # + [num_integration_points]
    return tf.concat(
        [
            # Shape batch_shape + [1] + [num_integration_points]
            tf.expand_dims(self.start_times, axis=-2) * s,
            # Shape batch_shape + [num_sampling_times - fixing_count + 2]
            # + [num_integration_points]
            tf.expand_dims(
                tf.broadcast_to(
                    self.rho / (self.volvol * n_real),
                    tf.concat([self.batch_shape, [self.n - self.k_star]],
                              axis=0)), axis=-1) * s,
            # Shape batch_shape + [1] + [num_integration_points]
            tf.expand_dims(self.rho / self.volvol, axis=-1) * w
        ], axis=-2)

  def omega_tilde_k(self, s, w, z_k_tensor):
    """Calculates tilde{omega}_k(s, w) (eq. (19) from the referenced paper)."""
    omega_tilde = self.omega_k(s, w)
    mr_v2 = self.mean_reversion / tf.math.pow(self.volvol, 2)
    x0 = utils.get_shape(omega_tilde)[-2] - 1
    # If shape of omega_tilde is known, use TensorArray of known shape
    if isinstance(x0, int):
      ta = tf.TensorArray(omega_tilde.dtype, size=x0 + 1,
                          clear_after_read=False)
    else:
      # Otherwise, use dynamically shaped TensorArray
      ta = tf.TensorArray(omega_tilde.dtype, size=0, dynamic_size=True)
    last_row = omega_tilde[..., 0, :]

    condition = lambda q, ta, last_row: q >= 0

    def modify_row(q, row, last_row):
      d_tau = tf.expand_dims(self.d_tau_k[q], axis=-1)
      z_k = z_k_tensor[q]

      return row + mr_v2 - (2 * self.f_tilde(z_k, last_row, d_tau)
                            / (self.f(z_k, last_row, d_tau)
                               * tf.math.pow(self.volvol, 2)))

    def body(q, ta, last_row):
      # Use tf.where here for better vecotrization
      row = tf.where(q < x0,
                     modify_row(tf.minimum(q, x0 - 1),
                                omega_tilde[..., tf.minimum(q, x0 - 1), :],
                                last_row),
                     omega_tilde[..., q, :])
      ta = ta.write(q, row)
      last_row = row
      return (q-1, ta, last_row)

    _, r, _ = tf.while_loop(condition, body, [x0, ta, last_row])
    # Shape batch_shape + [num_sampling_times - fixing_count + 2]
    # + [num_integration_points]
    return r.stack()

  def phi(self, s, w):
    """Calculates Phi(s, w) (eq. (21) from the referenced paper)."""
    z_k_tensor = self.z(s, w)
    omega_tilde = self.omega_tilde_k(s, w, z_k_tensor)
    tte = (self.expiries - self.start_times)

    term_1 = self.a(s, w) + omega_tilde[0] * self.variances
    term_2 = (tf.math.pow(self.mean_reversion, 2) * self.theta
              * tte / tf.math.pow(self.volvol, 2))
    term_3 = 2 * self.mean_reversion * self.theta / tf.math.pow(self.volvol, 2)

    # Summation shape `batch_size + [s]` after the reduction, where `s` is the
    # tensor of integration variable values
    summation = tf.reduce_sum(
        tf.math.log(self.f(z_k_tensor, omega_tilde[1:],
                           tf.expand_dims(self.d_tau_k, axis=-1))), axis=0)

    return tf.math.exp(term_1 + term_2 - term_3 * summation)

  def integrand(self, x):
    """Calculates pricing integrand (eq. (23), (24) from the referenced paper).
    """
    x = tf.cast(x, self.dtype)
    w = tf.zeros_like(x)

    term_1 = self.phi(1 + x * 1j, w) - self.strikes * self.phi(x * 1j, w)
    term_2 = tf.math.exp(-1j * x * tf.math.log(self.strikes))/(x * 1j)

    return tf.math.real(term_1 * term_2)
