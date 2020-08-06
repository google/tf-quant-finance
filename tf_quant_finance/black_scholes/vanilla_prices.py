# Lint as: python3
# Copyright 2019 Google LLC
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
"""Black Scholes prices of a batch of European options."""

import numpy as np
import tensorflow.compat.v2 as tf


def option_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 discount_rates=None,
                 continuous_dividends=None,
                 cost_of_carries=None,
                 discount_factors=None,
                 is_call_options=None,
                 is_normal_volatility=False,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of call or put options.

  #### Example

  ```python
    # Price a batch of 5 vanilla call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
  # Expected print output of computed prices:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```

  #### References:
  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Black-Scholes model. Available at:
    https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
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
    discount_rates: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      If not `None`, discount factors are calculated as e^(-rT),
      where r are the discount rates, or risk free rates. At most one of
      discount_rates and discount_factors can be supplied.
      Default value: `None`, equivalent to r = 0 and discount factors = 1 when
      discount_factors also not given.
    continuous_dividends: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      If not `None`, `cost_of_carries` is calculated as r - q,
      where r are the `discount_rates` and q is `continuous_dividends`. Either
      this or `cost_of_carries` can be given.
      Default value: `None`, equivalent to q = 0.
    cost_of_carries: An optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      Cost of storing a physical commodity, the cost of interest paid when
      long, or the opportunity cost, or the cost of paying dividends when short.
      If not `None`, and `spots` is supplied, used to calculate forwards from
      `spots`: F = e^(bT) * S, where F is the forwards price, b is the cost of
      carries, T is expiries and S is the spot price. If `None`, value assumed
      to be equal to the `discount_rate` - `continuous_dividends`
      Default value: `None`, equivalent to b = r.
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not `None`, these are the discount factors to expiry
      (i.e. e^(-rT)). Mutually exclusive with discount_rate and cost_of_carry.
      If neither is given, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not `None` then this is also used to compute the forwards to expiry.
      At most one of discount_rates and discount_factors can be supplied.
      Default value: `None`, which maps to -log(discount_factors) / expiries
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    is_normal_volatility: An optional Python boolean specifying whether the
      `volatilities` correspond to lognormal Black volatility (if False) or
      normal Black volatility (if True).
      Default value: False, which corresponds to lognormal volatility.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
        TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name `option_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
    ValueError: If both `discount_rates` and `discount_factors` is supplied.
    ValueError: If both `continuous_dividends` and `cost_of_carries` is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if (discount_rates is not None) and (discount_factors is not None):
    raise ValueError('At most one of discount_rates and discount_factors may '
                     'be supplied')
  if (continuous_dividends is not None) and (cost_of_carries is not None):
    raise ValueError('At most one of continuous_dividends and cost_of_carries '
                     'may be supplied')

  with tf.name_scope(name or 'option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

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

    if discount_factors is not None:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')
    else:
      discount_factors = tf.exp(-discount_rates * expiries)

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots * tf.exp(cost_of_carries * expiries)

    sqrt_var = volatilities * tf.math.sqrt(expiries)
    if not is_normal_volatility:  # lognormal model
      d1 = (tf.math.log(forwards / strikes) +
            sqrt_var * sqrt_var / 2) / sqrt_var
      d2 = d1 - sqrt_var
      undiscounted_calls = forwards * _ncdf(d1) - strikes * _ncdf(d2)
    else:  # normal model
      d1 = (forwards - strikes) / sqrt_var
      undiscounted_calls = (forwards - strikes) * _ncdf(
          d1) + sqrt_var * tf.math.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)

    if is_call_options is None:
      return discount_factors * undiscounted_calls
    undiscounted_forward = forwards - strikes
    undiscounted_puts = undiscounted_calls - undiscounted_forward
    predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
    return discount_factors * tf.where(predicate, undiscounted_calls,
                                       undiscounted_puts)


def barrier_price(*,
                  volatilities,
                  strikes,
                  expiries,
                  spots,
                  barriers,
                  rebates=None,
                  discount_rates=None,
                  continuous_dividends=None,
                  cost_of_carries=None,
                  is_barrier_down=None,
                  is_knock_out=None,
                  is_call_options=None,
                  dtype=None,
                  name=None):
  """Prices barrier options in a Black-Scholes Model.

  Computes the prices of options with a single barrier in Black-Scholes world as
  described in Ref. [1]. Note that the barrier is applied continuously.

  #### Example

  This example is taken from Ref. [2], Page 154.

  ```python
  import tf_quant_finance as tff

  dtype = np.float32
  discount_rates = np.array([.08, .08])
  continuous_dividends = np.array([.04, .04])
  spots = np.array([100., 100.])
  strikes = np.array([90., 90.])
  barriers = np.array([95. 95.])
  rebates = np.array([3. 3.])
  volatilities = np.array([.25, .25])
  expiries = np.array([.5, .5])
  barriers_type = np.array([5, 1])
  is_barrier_down = np.array([True, False])
  is_knock_out = np.array([False, False])
  is_call_option = np.array([True, True])

  price = tff.black_scholes.barrier_price(
    discount_rates, continuous_dividends, spots, strikes,
    barriers, rebates, volatilities,
    expiries, is_barrier_down, is_knock_out, is_call_options)

  # Expected output
  #  `Tensor` with values [9.024, 7.7627]
  ```

  #### References

  [1]: Lee Clewlow, Javier Llanos, Chris Strickland, Caracas Venezuela
    Pricing Exotic Options in a Black-Scholes World, 1994
    https://warwick.ac.uk/fac/soc/wbs/subjects/finance/research/wpaperseries/1994/94-54.pdf
  [2]: Espen Gaarder Haug, The Complete Guide to Option Pricing Formulas,
    2nd Edition, 1997

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and compatible shape as
      `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
      `volatilities`. The expiry of each option. The units should be such that
      `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `volatilities`. The current spot price of the underlying.
    barriers: A real `Tensor` of same dtype as the `volatilities` and of the
      shape that broadcasts with `volatilities`. The barriers of each option.
    rebates: A real `Tensor` of same dtype as the `volatilities` and of the
      shape that broadcasts with `volatilities`. For knockouts, this is a
      fixed cash payout in case the barrier is breached. For knockins, this is a
      fixed cash payout in case the barrier level is not breached. In the former
      case, the rebate is paid immediately on breach whereas in the latter, the
      rebate is paid at the expiry of the option.
      Default value: `None` which maps to no rebates.
    discount_rates: A real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      Discount rates, or risk free rates.
      Default value: `None`, equivalent to discount_rate = 0.
    continuous_dividends: A real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`. A
      continuous dividend rate paid by the underlier. If `None`, then
      defaults to zero dividends.
      Default value: `None`, equivalent to zero dividends.
    cost_of_carries: A optional real `Tensor` of same dtype as the
      `volatilities` and of the shape that broadcasts with `volatilities`.
      Cost of storing a physical commodity, the cost of interest paid when
      long, or the opportunity cost, or the cost of paying dividends when short.
      If not `None`, `continuous_dividends` is calculated as r - c,
      where r are the `discount_rates` and c is `cost_of_carries`.
    is_barrier_down: A real `Tensor` of `boolean` values and of the shape
      that broadcasts with `volatilities`. True if barrier is below asset
      price at expiration.
      Default value: `True`.
    is_knock_out: A real `Tensor` of `boolean` values and of the shape
      that broadcasts with `volatilities`. True if option is knock out
      else false.
      Default value: `True`.
    is_call_options: A real `Tensor` of `boolean` values and of the shape
      that broadcasts with `volatilities`. True if option is call else
      false.
      Default value: `True`.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: `None` which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: `None` which is mapped to the default name `barrier_price`.
  Returns:
    option_prices: A `Tensor` of same shape as `spots`. The approximate price of
    the barriers option under black scholes.
  """
  # The computation is done as in Ref [2] where each integral is split into
  # two matrices. The first matrix contains the algebraic terms and the second
  # matrix contains the probability distribution terms. Masks are used to filter
  # appropriate terms for calculating the integral. Then a dot product of each
  # row in the matricies coupled with the masks work to calculate the prices of
  # the barriers option.
  if (continuous_dividends is not None) and (cost_of_carries is not None):
    raise ValueError('At most one of continuous_dividends and cost of carries '
                     'may be supplied')
  with tf.name_scope(name or 'barrier_price'):
    spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
    dtype = spots.dtype
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    barriers = tf.convert_to_tensor(barriers, dtype=dtype, name='barriers')
    if rebates is not None:
      rebates = tf.convert_to_tensor(rebates, dtype=dtype, name='rebates')
    else:
      rebates = tf.zeros_like(spots, dtype=dtype, name='rebates')

    # Convert all to tensor and enforce float dtype where required
    if discount_rates is not None:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
    else:
      discount_rates = tf.convert_to_tensor(
          1, dtype=dtype, name='discount_rates')

    if continuous_dividends is None:
      continuous_dividends = tf.convert_to_tensor(
          0.0, dtype=dtype, name='continuous_dividends')

    if cost_of_carries is not None:
      cost_of_carries = tf.convert_to_tensor(
          cost_of_carries, dtype=dtype, name='cost_of_carries')
    else:
      cost_of_carries = discount_rates - continuous_dividends

    if is_barrier_down is None:
      is_barrier_down = tf.constant(1, name='is_barrier_down')
    else:
      is_barrier_down = tf.convert_to_tensor(is_barrier_down, dtype=tf.bool,
                                             name='is_barrier_down')
      is_barrier_down = tf.where(is_barrier_down, 1, 0)
    if is_knock_out is None:
      is_knock_out = tf.constant(1, name='is_knock_out')
    else:
      is_knock_out = tf.convert_to_tensor(is_knock_out, dtype=tf.bool,
                                          name='is_knock_out')
      is_knock_out = tf.where(is_knock_out, 1, 0)
    if is_call_options is None:
      is_call_options = tf.constant(1, name='is_call_options')
    else:
      is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool,
                                             name='is_call_options')
      is_call_options = tf.where(is_call_options, 1, 0)

    # Indices which range from 0-7 are used to select the appropriate
    # mask for each barrier
    indices = tf.bitwise.left_shift(
        is_barrier_down, 2) + tf.bitwise.left_shift(
            is_knock_out, 1) + is_call_options

    # Masks select the appropriate terms for integral approximations
    # Integrals are seperated by algebraic terms and probability
    # distribution terms. This give 12 different terms per matrix
    # (6 integrals, 2 terms each)
    # shape = [8, 12]
    mask_matrix_greater_strike = tf.constant([
        [1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0],  # up and in put
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # up and in call
        [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1],  # up and out put
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # up and out call
        [0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1],  # down and in put
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # down and in call
        [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1],  # down and out put
        [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1]])  # down and out call

    mask_matrix_lower_strike = tf.constant([
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # up and in put
        [0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0],  # up and in call
        [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1],  # up and out put
        [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1],  # up and out call
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # down and in put
        [1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0],  # down and in call
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # down and out put
        [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1]])  # down and out call

    # Create masks
    # Masks are shape [strikes.shape, 12]
    masks_lower = tf.gather(mask_matrix_lower_strike, indices, axis=0)
    masks_greater = tf.gather(mask_matrix_greater_strike, indices, axis=0)
    strikes_greater = tf.expand_dims(strikes > barriers, axis=-1)
    masks = tf.where(strikes_greater, masks_greater, masks_lower)
    masks = tf.cast(masks, dtype=dtype)
    one = tf.constant(1, dtype=dtype)
    call_or_put = tf.cast(tf.where(tf.equal(is_call_options, 0), -one, one),
                          dtype=dtype)
    below_or_above = tf.cast(tf.where(tf.equal(is_barrier_down, 0), -one, one),
                             dtype=dtype)

    # Calculate params for integrals
    sqrt_var = volatilities * tf.math.sqrt(expiries)
    mu = (cost_of_carries) - ((volatilities**2) / 2)
    lamda = 1 + (mu / (volatilities**2))
    x = (tf.math.log(spots / strikes) / (sqrt_var)) + (lamda * sqrt_var)
    x1 = (tf.math.log(spots / barriers) / (sqrt_var)) + (lamda * sqrt_var)
    y = (tf.math.log((barriers**2) / (spots * strikes)) / (
        sqrt_var)) + (lamda * sqrt_var)
    y1 = (tf.math.log(barriers / spots) / (sqrt_var)) + (lamda * sqrt_var)
    b = ((mu**2) + (2 * (volatilities**2) * discount_rates)) / (volatilities**2)
    z = (tf.math.log(barriers / spots) / (sqrt_var)) + (b * sqrt_var)
    a = mu / (volatilities**2)

    # Other params used for integrals
    discount_rates_exponent = tf.math.exp(-discount_rates * expiries,
                                          name='discount_rates_exponent')
    continuous_dividends_exponent = tf.math.exp(
        (cost_of_carries-discount_rates) * expiries,
        name='continuous_dividends_exponent')
    barriers_ratio = tf.math.divide(barriers, spots, name='barriers_ratio')
    spots_term = call_or_put * spots * continuous_dividends_exponent
    strikes_term = call_or_put * strikes * discount_rates_exponent

    # rank is used to stack elements and reduce_sum
    strike_rank = strikes.shape.rank

    # Constructing Matrix with first and second algebraic terms for each
    # integral [strike.shape, 12]
    terms_mat = tf.stack(
        (spots_term, -strikes_term,
         spots_term, -strikes_term,
         spots_term * (barriers_ratio**(2 * lamda)),
         -strikes_term * (barriers_ratio**((2 * lamda) - 2)),
         spots_term * (barriers_ratio**(2 * lamda)),
         -strikes_term * (barriers_ratio**((2 * lamda) - 2)),
         rebates * discount_rates_exponent,
         -rebates * discount_rates_exponent * (
             barriers_ratio**((2 * lamda) - 2)),
         rebates * (barriers_ratio**(a + b)),
         rebates * (barriers_ratio**(a - b))),
        name='term_matrix', axis=strike_rank)

    # Constructing Matrix with first and second norm for each integral
    # [strikes.shape, 12]
    cdf_mat = tf.stack(
        (call_or_put * x,
         call_or_put * (x - sqrt_var),
         call_or_put * x1,
         call_or_put * (x1 - sqrt_var),
         below_or_above * y,
         below_or_above * (y - sqrt_var),
         below_or_above * y1,
         below_or_above * (y1 - sqrt_var),
         below_or_above * (x1 - sqrt_var),
         below_or_above * (y1 - sqrt_var),
         below_or_above * z,
         below_or_above * (z - (2 * b * sqrt_var))),
        name='cdf_matrix', axis=strike_rank)
    cdf_mat = _ncdf(cdf_mat)
    # Calculating and returning price for each option
    return tf.reduce_sum(masks * terms_mat * cdf_mat, axis=strike_rank)


# TODO(b/154806390): Binary price signature should be the same as that of the
# vanilla price.
def binary_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the Black Scholes price for a batch of binary call or put options.

  The binary call (resp. put) option priced here is that which pays off a unit
  of cash if the underlying asset has a value greater (resp. smaller) than the
  strike price at expiry. Hence the binary option price is the discounted
  probability that the asset will end up higher (resp. lower) than the
  strike price at expiry.

  #### Example

  ```python
    # Price a batch of 5 binary call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.binary_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
  # Expected print output of prices:
  # [0.         0.         0.15865525 0.99764937 0.85927418]
  ```

  #### References:

  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Binary option. Available at:
  https://en.wikipedia.org/w/index.php?title=Binary_option

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
      expiry of the options to price.
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
    discount_factors: An optional real `Tensor` of same dtype as the
      `volatilities`. If not None, these are the discount factors to expiry
      (i.e. e^(-rT)). If None, no discounting is applied (i.e. the undiscounted
      option price is returned). If `spots` is supplied and `discount_factors`
      is not None then this is also used to compute the forwards to expiry.
      Default value: None, equivalent to discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `binary_price`.

  Returns:
    binary_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the binary options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')

  with tf.name_scope(name or 'binary_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

    if discount_factors is None:
      discount_factors = tf.convert_to_tensor(
          1.0, dtype=dtype, name='discount_factors')
    else:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots / discount_factors

    sqrt_var = volatilities * tf.math.sqrt(expiries)
    d1 = (tf.math.log(forwards / strikes) + sqrt_var * sqrt_var / 2) / sqrt_var
    d2 = d1 - sqrt_var
    undiscounted_calls = _ncdf(d2)
    if is_call_options is None:
      return discount_factors * undiscounted_calls
    is_call_options = tf.convert_to_tensor(is_call_options,
                                           dtype=tf.bool,
                                           name='is_call_options')
    undiscounted_puts = 1 - undiscounted_calls
    predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
    return discount_factors * tf.where(predicate, undiscounted_calls,
                                       undiscounted_puts)


def swaption_price(*,
                   volatilities,
                   expiries,
                   floating_leg_start_times,
                   floating_leg_end_times,
                   fixed_leg_payment_times,
                   floating_leg_daycount_fractions,
                   fixed_leg_daycount_fractions,
                   fixed_leg_coupon,
                   floating_leg_start_times_discount_factors,
                   floating_leg_end_times_discount_factors,
                   fixed_leg_payment_times_discount_factors,
                   notional=None,
                   is_payer_swaption=None,
                   is_normal_volatility=True,
                   dtype=None,
                   name=None):
  """Calculates the price of European Swaptions using the Black model.

  A European Swaption is a contract that gives the holder an option to enter a
  swap contract at a future date at a prespecified fixed rate. A swaption that
  grants the holder to pay fixed rate and receive floating rate is called a
  payer swaption while the swaption that grants the holder to receive fixed and
  pay floating payments is called the receiver swaption. Typically the start
  date (or the inception date) of the swap coincides with the expiry of the
  swaption.

  #### Example
  The example shows how value a batch of 1y x 1y and 1y x 2y swaptions using the
  Black (normal) model for the swap rate.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  volatilities = [0.01, 0.005]
  expiries = [1.0, 1.0]
  float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],
                            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]
  float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
  fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                              [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
  float_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                                   [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                                   0.25]]
  fixed_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                                   [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                                   0.25]]
  fixed_leg_coupon = [0.011, 0.011]
  discount_fn = lambda x: np.exp(-0.01 * np.array(x))
  price = self.evaluate(
  tff.black_scholes.swaption_price(
      volatilities=volatilities,
      expiries=expiries,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      floating_leg_start_times_discount_factors=discount_fn(
          float_leg_start_times),
      floating_leg_end_times_discount_factors=discount_fn(
          float_leg_end_times),
      fixed_leg_payment_times_discount_factors=discount_fn(
          fixed_leg_payment_times),
      is_normal_volatility=is_normal_model,
      notional=100.,
      dtype=dtype))
  # Expected value: [0.3458467885511461, 0.3014786656395892] # shape = (2,)
  ````

  Args:
    volatilities: Real `Tensor` of any shape and dtype. The Black volatilities
      of the swaptions to price. The shape of this input determines the number
      (and shape) of swaptions to be priced and the shape of the output.
    expiries: A real `Tensor` of same shape and dtype as `volatilities`. The
      time to expiration of the swaptions.
    floating_leg_start_times: A real `Tensor` of the same dtype as
      `volatilities`. The times when accrual begins for each payment in the
      floating leg. The shape of this input should be `expiries.shape + [m]` or
      `batch_shape + [m]` where `m` denotes the number of floating payments in
      each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `volatilities`.
      The times when accrual ends for each payment in the floating leg. The
      shape of this input should be `batch_shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as
      `volatilities`.  The payment times for each payment in the fixed leg.
      The shape of this input should be `batch_shape + [n]` where `n` denotes
      the number of fixed payments in each leg.
    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `floating_leg_start_times`. The daycount fractions
      for each payment in the floating leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions
      for each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and shape compatible
      to `batch_shape`. The fixed coupon rate for each payment in the fixed leg.
    floating_leg_start_times_discount_factors: A real `Tensor` of the same
      shape and dtype as `floating_leg_start_times`. The discount factors
      corresponding to `floating_leg_start_times`.
    floating_leg_end_times_discount_factors: A real `Tensor` of the same
      shape and dtype as `floating_leg_end_times`. The discount factors
      corresponding to `floating_leg_end_times`.
    fixed_leg_payment_times_discount_factors: A real `Tensor` of the same
      shape and dtype as `fixed_leg_payment_times`. The discount factors
      corresponding to `fixed_leg_payment_times`.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `volatilities` specifying the notional amount for the underlying swap.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the swaption is a payer (if True) or a receiver
      (if False) swaption. If not supplied, payer swaptions are assumed.
    is_normal_volatility: An optional Python boolean specifying whether the
      `volatilities` correspond to normal Black volatility (if True) or
      lognormal Black volatility (if False).
      Default value: True, which corresponds to normal volatility.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
      TensorFlow are used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `hw_swaption_price`.

  Returns:
    A `Tensor` of real dtype and shape `batch_shape` containing the
    computed swaption prices.
  """
  name = name or 'black_swaption_price'
  del floating_leg_daycount_fractions
  with tf.name_scope(name):
    volatilities = tf.convert_to_tensor(volatilities, dtype=dtype,
                                        name='volatilities')
    dtype = dtype or volatilities.dtype
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    floating_leg_start_times = tf.convert_to_tensor(
        floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
    floating_leg_end_times = tf.convert_to_tensor(
        floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions, dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
    float_leg_start_times_discount_factors = tf.convert_to_tensor(
        floating_leg_start_times_discount_factors, dtype=dtype,
        name='float_leg_start_times_discount_factors')
    float_leg_end_times_discount_factors = tf.convert_to_tensor(
        floating_leg_end_times_discount_factors, dtype=dtype,
        name='float_leg_end_times_discount_factors')
    fixed_leg_payment_times_discount_factors = tf.convert_to_tensor(
        fixed_leg_payment_times_discount_factors, dtype=dtype,
        name='fixed_leg_payment_times_discount_factors')

    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    if is_payer_swaption is None:
      is_payer_swaption = True
    is_payer_swaption = tf.convert_to_tensor(
        is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')

    swap_annuity = tf.math.reduce_sum(
        fixed_leg_daycount_fractions * fixed_leg_payment_times_discount_factors,
        axis=-1)
    forward_swap_rate = tf.math.reduce_sum(
        float_leg_start_times_discount_factors -
        float_leg_end_times_discount_factors, axis=-1) / swap_annuity
    swaption_value = option_price(volatilities=volatilities,
                                  strikes=fixed_leg_coupon,
                                  expiries=expiries,
                                  forwards=forward_swap_rate,
                                  is_call_options=is_payer_swaption,
                                  is_normal_volatility=is_normal_volatility,
                                  dtype=dtype,
                                  name=name + '_option_price')
    return notional * swap_annuity * swaption_value


def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)
