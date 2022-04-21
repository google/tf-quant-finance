# Copyright 2021 Google LLC
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
"""Analytic approximation for European option prices using SABR model."""

import enum

import tensorflow.compat.v2 as tf


@enum.unique
class SabrApproximationType(enum.Enum):
  """Approximation to the SABR model.

  * `HAGAN`: Using the Hagan approximation [1].

  #### References
  [1] Hagan et al, Managing Smile Risk, Wilmott (2002), 1:84-108
  """
  HAGAN = 1


@enum.unique
class SabrImpliedVolatilityType(enum.Enum):
  """The implied volality arising from the SABR approximate solution.

  * `NORMAL`: The volatility for the normal model, i.e. the `sigma_n` for a
    stochastic model of the underlying `F` behaving like:

    ```
    dF = sigma_n dW
    ```

  * `LOGNORMAL`: The volatility for the lognomal (aka Black) model, i.e. the
    `sigma_B` for a stochastic model of the underlying `F` behaving like:

    ```
    dF = sigma_b F dW
    ```

  """
  NORMAL = 1
  LOGNORMAL = 2


def implied_volatility(*,
                       strikes,
                       expiries,
                       forwards,
                       alpha,
                       beta,
                       volvol,
                       rho,
                       shift=0.0,
                       volatility_type=SabrImpliedVolatilityType.LOGNORMAL,
                       approximation_type=SabrApproximationType.HAGAN,
                       dtype=None,
                       name=None):
  """Computes the implied volatility under the SABR model.

  The SABR model specifies the risk neutral dynamics of the underlying as the
  following set of stochastic differential equations:

  ```
    dF = sigma F^beta dW_1
    dsigma = volvol sigma dW_2
    dW1 dW2 = rho dt

    F(0) = f
    sigma(0) = alpha
  ```
  where F(t) represents the value of the forward price as a function of time,
  and sigma(t) is the volatility.

  Here, we implement an approximate solution as proposed by Hagan [1], and back
  out the equivalent implied volatility that would've been obtained under either
  the normal model or the Black model.

  #### Example
  ```python
  import tf_quant_finance as tff
  import tensorflow.compat.v2 as tf

  equiv_vol = tff.models.sabr.approximations.implied_volatility(
      strikes=np.array([106.0, 11.0]),
      expiries=np.array([17.0 / 365.0, 400.0 / 365.0]),
      forwards=np.array([120.0, 20.0]),
      alpha=1.63,
      beta=0.6,
      rho=0.00002,
      volvol=3.3,
      dtype=tf.float64)
  # Expected: [0.33284656705268817, 1.9828728139982792]

  # Running this inside a unit test passes:
  # equiv_vol = self.evaluate(equiv_vol)
  # self.assertAllClose(equiv_vol, 0.33284656705268817)
  ```
  #### References
  [1] Hagan et al, Managing Smile Risk, Wilmott (2002), 1:84-108

  Args:
    strikes: Real `Tensor` of arbitrary shape, specifying the strike prices.
      Values must be strictly positive.
    expiries: Real `Tensor` of shape compatible with that of `strikes`,
      specifying the corresponding time-to-expiries of the options. Values must
      be strictly positive.
    forwards: Real `Tensor` of shape compatible with that of `strikes`,
      specifying the observed forward prices of the underlying. Values must be
      strictly positive.
    alpha: Real `Tensor` of shape compatible with that of `strikes`, specifying
      the initial values of the stochastic volatility. Values must be strictly
      positive.
    beta: Real `Tensor` of shape compatible with that of `strikes`, specifying
      the model exponent `beta`. Values must satisfy 0 <= `beta` <= 1.
    volvol: Real `Tensor` of shape compatible with that of `strikes`,
      specifying the model vol-vol multipliers. Values of `volvol` must be
      non-negative.
    rho: Real `Tensor` of shape compatible with that of `strikes`, specifying
      the correlation factors between the Wiener processes modeling the forward
      and the volatility. Values must satisfy -1 < `rho` < 1.
    shift: Optional `Tensor` of shape compatible with that of `strkies`,
      specifying the shift parameter(s). In the shifted model, the process
      modeling the forward is modified as: dF = sigma * (F + shift) ^ beta * dW.
      With this modification, negative forward rates are valid as long as
      F > -shift.
      Default value: 0.0
    volatility_type: Either SabrImpliedVolatility.NORMAL or LOGNORMAL.
      Default value: `LOGNORMAL`.
    approximation_type: Instance of `SabrApproxmationScheme`.
      Default value: `HAGAN`.
    dtype: Optional: `tf.DType`. If supplied, the dtype to be used for
      converting values to `Tensor`s.
      Default value: `None`, which means that the default dtypes inferred from
        `strikes` is used.
    name: str. The name for the ops created by this function.
      Default value: 'sabr_approx_implied_volatility'.

  Returns:
    A real `Tensor` of the same shape as `strikes`, containing the
    corresponding equivalent implied volatilities.
  """
  name = name or 'sabr_approx_implied_volatility'
  del approximation_type  # Currently, only HAGAN approximation is supported.

  with tf.name_scope(name):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = dtype or strikes.dtype

    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    alpha = tf.convert_to_tensor(alpha, dtype=dtype, name='alpha')
    beta = tf.convert_to_tensor(beta, dtype=dtype, name='beta')
    rho = tf.convert_to_tensor(rho, dtype=dtype, name='rho')
    volvol = tf.convert_to_tensor(volvol, dtype=dtype, name='volvol')

    # Apply the shift.
    strikes += shift
    forwards += shift

    moneyness = forwards / strikes
    log_moneyness = tf.math.log(moneyness)
    adj_moneyness = tf.math.pow(moneyness, 1.0 - beta)
    sqrt_adj_moneyness = tf.math.sqrt(adj_moneyness)

    # adjusted alpha = alpha * K^(beta - 1)
    adj_alpha = alpha * tf.math.pow(strikes, beta - 1.0)

    # Zeta, as defined in (eq. A.69b in [1])
    zeta = (volvol / adj_alpha) * sqrt_adj_moneyness * log_moneyness
    # Zeta / xhat(zeta), as defined in (eq. A.69b in [1])
    zeta_by_xhat = _zeta_by_xhat(zeta, rho, dtype)

    # This is the denominator term occurring in the ((1 + ...) / (1 + ...)) of
    # (eq. A.69a) in [1].
    denom = _denom(beta, log_moneyness)

    # The correction terms occurring in (1 + {...}) of (eq. A.69a) of [1], where
    # we have multiplied in the "t_ex" to make the quantities dimensionless.
    correction_2 = ((rho * beta / 4.0) * (1.0 / sqrt_adj_moneyness) *
                    (adj_alpha * volvol * expiries))

    correction_3 = ((2.0 - 3.0 * rho * rho) / 24.0
                    * (volvol * volvol * expiries))

    if volatility_type == SabrImpliedVolatilityType.NORMAL:
      correction_1 = ((-beta * (2.0 - beta) / 24.0) * (1.0 / adj_moneyness) *
                      (adj_alpha * adj_alpha * expiries))

      # This is the denominator term occurring in the ((1 + ...) / (1 + ...)) of
      # (eq. A.69a) in [1], and is effectively the same as setting beta = 0.0
      number = _denom(0.0, log_moneyness)

      return (adj_alpha * strikes * tf.math.pow(moneyness, beta / 2.0) *
              (number / denom) * zeta_by_xhat *
              (1 + correction_1 + correction_2 + correction_3))

    elif volatility_type == SabrImpliedVolatilityType.LOGNORMAL:
      correction_1 = (((1.0 - beta) *
                       (1.0 - beta) / 24.0) * (1.0 / adj_moneyness) *
                      (adj_alpha * adj_alpha * expiries))

      return (adj_alpha * (1.0 / sqrt_adj_moneyness) * (1.0 / denom) *
              zeta_by_xhat * (1.0 + correction_1 + correction_2 + correction_3))
    else:
      raise ValueError('Invalid value of `volatility_type`')


def _epsilon(dtype):
  dtype = tf.as_dtype(dtype).as_numpy_dtype
  eps = 1e-6 if dtype == tf.float32.as_numpy_dtype else 1e-10
  return eps


def _zeta_by_xhat(zeta, rho, dtype):
  zbxh = tf.math.divide_no_nan(
      zeta,
      tf.math.log(
          (tf.math.sqrt(1 - 2 * rho * zeta + zeta * zeta) - rho + zeta) /
          (1.0 - rho)))
  eps = _epsilon(dtype)

  # When zeta -> 0, the limit of zeta / x_hat(zeta) reduces to 1.0
  return tf.where(tf.abs(zeta) > eps, zbxh, 1.0)


def _denom(beta, log_f_by_k):
  s = (1.0 - beta) * log_f_by_k
  s_squared = s * s
  return 1.0 + s_squared / 24.0 + (s_squared * s_squared) / 1920.0
