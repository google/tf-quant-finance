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
"""Black Scholes prices of a batch of Asian options."""

import enum
from typing import Optional
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.black_scholes import vanilla_prices

__all__ = ['asian_option_price']


@enum.unique
class AveragingType(enum.Enum):
  r"""Averaging types for asian options.

  * `GEOMETRIC`: C = ( \prod S(t_i) ) ^ {\frac{1}{n}}
  * `ARITHMETIC`: C = \frac{1}{n} \sum S(t_i)
  """
  GEOMETRIC = 1
  ARITHMETIC = 2


@enum.unique
class AveragingFrequency(enum.Enum):
  """Averaging types for asian options.

  * `DISCRETE`: Option samples on discrete times
  * `CONTINUOUS`: Option samples continuously throughout lifetime
  """
  DISCRETE = 1
  CONTINUOUS = 2


def asian_option_price(
    *,
    volatilities: types.RealTensor,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    spots: Optional[types.RealTensor] = None,
    forwards: Optional[types.RealTensor] = None,
    sampling_times: Optional[types.RealTensor] = None,
    past_fixings: Optional[types.RealTensor] = None,
    discount_rates: Optional[types.RealTensor] = None,
    dividend_rates: Optional[types.RealTensor] = None,
    discount_factors: Optional[types.RealTensor] = None,
    is_call_options: Optional[types.BoolTensor] = None,
    is_normal_volatility: bool = False,
    averaging_type: AveragingType = AveragingType.GEOMETRIC,
    averaging_frequency: AveragingFrequency = AveragingFrequency.DISCRETE,
    dtype: tf.DType = None,
    name: str = None) -> types.RealTensor:
  """Computes the Black Scholes price for a batch of asian options.

  In Black-Scholes, the marginal distribution of the underlying at each sampling
  date is lognormal. The product of a sequence of lognormal variables is also
  lognormal so we can re-express these options as vanilla options with modified
  parameters and use the vanilla pricer to price them.

  TODO(b/261568763): support volatility term structures


  #### Example

  ```python
    # Price a batch of 5 seasoned discrete geometric Asian options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    sampling_times = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                               [1.0, 1.0, 1.0, 1.0, 1.0]])
    past_fixings = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    computed_prices = tff.black_scholes.asian_option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        sampling_times=sampling_times,
        past_fixings=past_fixings)
  # Expected print output of computed prices:
  # [ 0.0, 0.0, 0.52833763, 0.99555802, 1.91452834]
  ```

  #### References:
  [1] Haug, E. G., The Complete Guide to Option Pricing Formulas. McGraw-Hill.

  Args:
    volatilities: Real `Tensor` of any shape compatible with a `batch_shape` and
      and anyy real dtype. The volatilities to expiry of the options to price.
      Here `batch_shape` corresponds to a batch of priced options.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying. Either this
      argument or the `forwards` (but not both) must be supplied.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `volatilities`. The forwards to maturity. Either this argument or the
      `spots` must be supplied but both must not be supplied.
    sampling_times: A real `Tensor` of same dtype as expiries and shape `[n] +
      batch_shape` where n is the number of sampling times for the Asian options
      Default value: `None`, which will raise an error for discrete sampling
      Asian options
    past_fixings: A real `Tensor` of same dtype as spots or forwards and shape
      `[n] + batch_shape` where n is the number of past fixings that have
      already been observed.
      Default value: `None`, equivalent to no past fixings (ie. unseasoned)
    discount_rates: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`. If
      not `None`, discount factors are calculated as e^(-rT), where r are the
      discount rates, or risk free rates. At most one of `discount_rates` and
      `discount_factors` can be supplied.
      Default value: `None`, equivalent to `r = 0` and `discount factors = 1`
      when `discount_factors` also not given.
    dividend_rates: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      Default value: `None`, equivalent to q = 0.
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is
      given, no discounting is applied (i.e. the undiscounted option price is
      returned). If `spots` is supplied and `discount_factors` is not `None`
      then this is also used to compute the forwards to expiry. At most one of
      `discount_rates` and `discount_factors` can be supplied.
      Default value: `None`, which maps to e^(-rT) calculated from
      `discount_rates`.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    is_normal_volatility: An optional Python boolean specifying whether the
      `volatilities` correspond to lognormal Black volatility (if False) or
      normal Black volatility (if True).
      Default value: False, which corresponds to lognormal volatility.
    averaging_type: Enum value of AveragingType to select the averaging method
      for the payoff calculation.
      Default value: AveragingType.GEOMETRIC
    averaging_frequency: Enum value of AveragingFrequency to select the
      averaging type for the payoff calculation (discrete vs continuous)
      Default value: AveragingFrequency.DISCRETE
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name
      `asian_option_price`.

  Returns:
    option_prices: A `Tensor` of shape `batch_shape` and the same dtype as
    `volatilities`. The Black Scholes price of the Asian options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
    ValueError: If both `discount_rates` and `discount_factors` is supplied.
    ValueError: If `is_normal_volatility` is true and option is geometric, or
      `is_normal_volatility` is false (ie. lognormal) and option is arithmetic.
    ValueError: If option is discrete averaging and `sampling_dates` is None of
      if last sampling date is later than option expiry date.
    NotImplementedError: if option is continuous averaging.
    NotImplementedError: if option is arithmetic.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')
  if is_normal_volatility and averaging_type == AveragingType.GEOMETRIC:
    raise ValueError('Cannot price geometric averaging asians analytically '
                     'under normal volatility')
  if not is_normal_volatility and averaging_type == AveragingType.ARITHMETIC:
    raise ValueError('Cannot price arithmetic averaging asians analytically '
                     'under lognormal volatility')
  if averaging_frequency == AveragingFrequency.DISCRETE:
    if sampling_times is None:
      raise ValueError('Sampling times required for discrete sampling asians')
    if not np.all(np.maximum(sampling_times[-1], expiries) == expiries):
      raise ValueError('Sampling times cannot occur after expiry times')
  if averaging_frequency == AveragingFrequency.CONTINUOUS:
    raise NotImplementedError('Pricing continuous averaging asians not yet '
                              'supported')
  if averaging_type == AveragingType.ARITHMETIC:
    raise NotImplementedError('Pricing arithmetic Asians not yet supported')

  with tf.name_scope(name or 'asian_option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

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

    if past_fixings is None:
      running_accumulator = tf.convert_to_tensor(1.0, dtype=dtype)
      fixing_count = 0
    else:
      running_accumulator = tf.reduce_prod(past_fixings, 0)
      fixing_count = past_fixings.shape[0]

    sample_count = sampling_times.shape[0] + fixing_count
    sampling_time_forwards = (
        spots * tf.exp((discount_rates - dividend_rates) * sampling_times))

    # We can price a discrete geometric asian option under BS using a vanilla
    # pricer, if we re-express the following parameters:
    #
    # t1 = \frac{1}{n} \sum_{i=1}^n t_i
    # t2 = \frac{1}{n^2} \sum_{i,j=1}^n \min(t_i, t_j)
    #
    # \sigma \to \sigma \sqrt{t2 / t}
    # F \to ( \prod_{i=1}^n F_i )^{\frac{1}{n}} e^{0.5 * \sigma^2 (t1 - t2)}
    #
    # where t_i are the sampling times, t is the expiry time, and F_i are the
    # forwards at the sampling times. Additionally dividend rates must be
    # adjusted to ensure the new forward is consistent with the discount factors
    # provided

    t1 = tf.reduce_sum(sampling_times, 0) / sample_count
    t2 = tf.reduce_sum(
        tf.vectorized_map(
            lambda x: tf.minimum(*tf.meshgrid(x, tf.transpose(x))),
            tf.transpose(sampling_times),
            fallback_to_while_loop=False), [1, 2]) / sample_count**2

    asian_forwards = (
        tf.math.pow(
            running_accumulator *
            tf.reduce_prod(sampling_time_forwards, axis=0), 1 / sample_count) *
        tf.math.exp(-0.5 * volatilities * volatilities * (t1 - t2)))

    effective_volatilities = volatilities * tf.math.sqrt(t2 / expiries)
    effective_dividend_rates = (
        discount_rates - tf.math.log(asian_forwards / spots) / expiries)

    return vanilla_prices.option_price(
        volatilities=effective_volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=asian_forwards,
        dividend_rates=effective_dividend_rates,
        discount_factors=discount_factors,
        is_call_options=is_call_options,
        dtype=dtype)
