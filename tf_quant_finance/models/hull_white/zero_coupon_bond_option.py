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
"""Pricing of zero coupon bond options using Hull-White model."""

import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.models import utils
from tf_quant_finance.models.hull_white import vector_hull_white


def _ncdf(x):
  """Implements the cumulative normal distribution function."""
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


def _prepare_indices(idx0, idx1, idx2, idx3):
  """Prepare indices to get relevant slice from discount curve simulations."""
  len0 = idx0.shape.as_list()[0]
  len1 = idx1.shape.as_list()[0]
  len3 = idx3.shape.as_list()[0]
  idx0 = tf.repeat(idx0, len1*len3)
  idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
  idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
  idx3 = tf.tile(idx3, [len0*len1])

  return tf.stack([idx0, idx1, idx2, idx3], axis=-1)


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)


def bond_option_price(*,
                      strikes,
                      expiries,
                      maturities,
                      discount_rate_fn,
                      dim,
                      mean_reversion,
                      volatility,
                      # TODO(b/159040541) Add correlation as an input.
                      is_call_options=True,
                      use_analytic_pricing=True,
                      num_samples=1,
                      random_type=None,
                      seed=None,
                      skip=0,
                      time_step=None,
                      dtype=None,
                      name=None):
  """Calculates European bond option prices using the Hull-White model.

  Bond options are fixed income securities which give the holder a right to
  exchange at a future date (the option expiry) a zero coupon bond for a fixed
  price (the strike of the option). The maturity date of the bond is after the
  the expiry of the option. If `P(t,T)` denotes the price at time `t` of a zero
  coupon bond with maturity `T`, then the payoff from the option at option
  expiry, `T0`, is given by:

  ```None
  payoff = max(P(T0, T) - X, 0)
  ```
  where `X` is the strike price of the option.

  #### Example

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  discount_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  expiries = np.array([1.0])
  maturities = np.array([5.0])
  strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
  price = tff.models.hull_white.bond_option_price(
      strikes=strikes,
      expiries=expiries,
      maturities=maturities,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.02],
      discount_rate_fn=discount_rate_fn,
      use_analytic_pricing=True,
      dtype=dtype)
  # Expected value: [[0.02817777]]
  ````

  Args:
    strikes: A real `Tensor` of any shape and dtype. The strike price of the
      options. The shape of this input determines the number (and shape) of the
      options to be priced and the output.
    expiries: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The time to expiry of each bond option.
    maturities: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The time to maturity of the underlying zero coupon bonds.
    discount_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape + dim`. Computes
      the zero coupon bond yield at the present time for the input expiry time.
    dim: A Python scalar which corresponds to the number of Hull-White Models
      to be used for pricing.
    mean_reversion: A real positive `Tensor` of shape `[dim]` or a Python
      callable. The callable can be one of the following:
      (a) A left-continuous piecewise constant object (e.g.,
      `tff.math.piecewise.PiecewiseConstantFunc`) that has a property
      `is_piecewise_constant` set to `True`. In this case the object should
      have a method `jump_locations(self)` that returns a `Tensor` of shape
      `[dim, num_jumps]` or `[num_jumps]`. In the first case,
      `mean_reversion(t)` should return a `Tensor` of shape `[dim] + t.shape`,
      and in the second, `t.shape + [dim]`, where `t` is a rank 1 `Tensor` of
      the same `dtype` as the output. See example in the class docstring.
      (b) A callable that accepts scalars (stands for time `t`) and returns a
      `Tensor` of shape `[dim]`.
      Corresponds to the mean reversion rate.
    volatility: A real positive `Tensor` of the same `dtype` as
      `mean_reversion` or a callable with the same specs as above.
      Corresponds to the lond run price variance.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `strikes`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
    use_analytic_pricing: A Python boolean specifying if analytic valuation
      should be performed. Analytic valuation is only supported for constant
      `mean_reversion` and piecewise constant `volatility`. If the input is
      `False`, then valuation using Monte-Carlo simulations is performed.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation. This input is ignored during analytic
      valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random
      number generator to use to generate the simulation paths. This input is
      relevant only for Monte-Carlo valuation and ignored during analytic
      valuation.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of
      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
      `HALTON_RANDOMIZED` the seed should be an Python integer. For
      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
      `Tensor` of shape `[2]`. This input is relevant only for Monte-Carlo
      valuation and ignored during analytic valuation.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation. This
      input is ignored during analytic valuation.
      Default value: `None`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
      TensorFlow are used.
    name: Python string. The name to give to the ops created by this class.
      Default value: `None` which maps to the default name
      `hw_bond_option_price`.

  Returns:
    A `Tensor` of real dtype and shape  `strikes.shape + [dim]` containing the
    computed option prices.
  """
  name = name or 'hw_bond_option_price'
  if dtype is None:
    dtype = tf.convert_to_tensor([0.0]).dtype
  with tf.name_scope(name):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    maturities = tf.convert_to_tensor(maturities, dtype=dtype,
                                      name='maturities')
    is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool,
                                           name='is_call_options')
    model = vector_hull_white.VectorHullWhiteModel(
        dim,
        mean_reversion=mean_reversion,
        volatility=volatility,
        initial_discount_rate_fn=discount_rate_fn,
        dtype=dtype)

    if use_analytic_pricing:
      return _analytic_valuation(
          discount_rate_fn, model, strikes, expiries, maturities, dim,
          is_call_options)

    if time_step is None:
      raise ValueError('`time_step` must be provided for simulation '
                       'based bond option valuation.')

    sim_times, _ = tf.unique(tf.reshape(expiries, shape=[-1]))
    longest_expiry = tf.reduce_max(sim_times)
    sim_times, _ = tf.unique(tf.concat([sim_times, tf.range(
        time_step, longest_expiry, time_step)], axis=0))
    sim_times = tf.sort(sim_times, name='sort_sim_times')
    tau = maturities - expiries
    curve_times_builder, _ = tf.unique(tf.reshape(tau, shape=[-1]))
    curve_times = tf.sort(curve_times_builder, name='sort_curve_times')

    p_t_tau, r_t = model.sample_discount_curve_paths(
        times=sim_times,
        curve_times=curve_times,
        num_samples=num_samples,
        random_type=random_type,
        seed=seed,
        skip=skip)

    dt_builder = tf.concat(
        [tf.convert_to_tensor([0.0], dtype=dtype),
         sim_times[1:] - sim_times[:-1]], axis=0)
    dt = tf.expand_dims(tf.expand_dims(dt_builder, axis=-1), axis=0)
    discount_factors_builder = tf.math.exp(-r_t * dt)
    # Transpose before (and after) because we want the cumprod along axis=1
    # and `matvec` operates on the last axis. The shape before and after would
    # be `(num_samples, len(times), dim)`
    discount_factors_builder = tf.transpose(
        utils.cumprod_using_matvec(
            tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])

    # make discount factors the same shape as `p_t_tau`. This involves adding
    # an extra dimenstion (corresponding to `curve_times`).
    discount_factors_builder = tf.expand_dims(
        discount_factors_builder,
        axis=1)
    discount_factors_simulated = tf.repeat(
        discount_factors_builder, p_t_tau.shape.as_list()[1], axis=1)

    # `sim_times` and `curve_times` are sorted for simulation. We need to
    # select the indices corresponding to our input.
    sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
    curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

    gather_index = _prepare_indices(
        tf.range(0, num_samples), curve_time_index, sim_time_index,
        tf.range(0, dim))

    # The shape after `gather_nd` would be (num_samples*num_strikes*dim,)
    payoff_discount_factors_builder = tf.gather_nd(
        discount_factors_simulated, gather_index)
    # Reshape to `[num_samples] + strikes.shape + [dim]`
    payoff_discount_factors = tf.reshape(
        payoff_discount_factors_builder,
        [num_samples] + strikes.shape + [dim])
    payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
    payoff_bond_price = tf.reshape(
        payoff_bond_price_builder, [num_samples] + strikes.shape + [dim])

    is_call_options = tf.reshape(
        tf.broadcast_to(is_call_options, strikes.shape),
        [1] + strikes.shape + [1])

    strikes = tf.reshape(strikes, [1] + strikes.shape + [1])
    payoff = tf.where(
        is_call_options,
        tf.math.maximum(payoff_bond_price - strikes, 0.0),
        tf.math.maximum(strikes - payoff_bond_price, 0.0))
    option_value = tf.math.reduce_mean(
        payoff_discount_factors * payoff, axis=0)

    return option_value


def _analytic_valuation(discount_rate_fn, model, strikes, expiries, maturities,
                        dim, is_call_options):
  """Performs analytic valuation."""

  discount_factor_expiry = tf.math.exp(
      -discount_rate_fn(expiries) * expiries)
  input_shape = expiries.shape
  variance = _bond_option_variance(
      model, tf.reshape(expiries, shape=[-1]), tf.reshape(maturities, [-1]),
      dim)
  variance = tf.reshape(variance, [dim] + input_shape)
  discount_factor_maturity = tf.math.exp(-discount_rate_fn(maturities) *
                                         maturities)
  forward_bond_price = discount_factor_maturity / discount_factor_expiry
  sqrt_variance = tf.math.sqrt(variance)
  log_moneyness = tf.expand_dims(tf.math.log(forward_bond_price / strikes),
                                 axis=0)
  d1 = tf.math.divide_no_nan(log_moneyness + 0.5 * variance, sqrt_variance)
  d2 = d1 - tf.math.sqrt(variance)
  option_value_call = (
      tf.expand_dims(discount_factor_maturity, axis=0) * _ncdf(d1) -
      tf.expand_dims(strikes * discount_factor_expiry, axis=0) * _ncdf(d2))
  option_value_put = (
      tf.expand_dims(strikes * discount_factor_expiry, axis=0) * _ncdf(-d2)
      - tf.expand_dims(discount_factor_maturity, axis=0) * _ncdf(-d1))

  is_call_options = tf.broadcast_to(is_call_options, [dim] + strikes.shape)
  intrinsic_value = tf.where(is_call_options,
                             tf.math.maximum(forward_bond_price - strikes, 0),
                             tf.math.maximum(strikes - forward_bond_price, 0))
  option_value = tf.where(
      maturities < expiries, tf.zeros_like(maturities),
      tf.where(sqrt_variance > 0.0,
               tf.where(is_call_options, option_value_call, option_value_put),
               intrinsic_value))

  # Make `dim` as the last dimension and return.
  return tf.transpose(
      option_value,
      perm=[i for i in range(1, len(option_value.shape.as_list()))] + [0])


# TODO(b/158501671): Clean-up this implementation.
def _bond_option_variance(model, option_expiry, bond_maturity, dim):
  """Computes black equivalent variance for bond options.

  Black equivalent variance is definied as the variance to use in the Black
  formula to obtain the model implied price of European bond options.

  Args:
    model: An instance of `VectorHullWhiteModel`.
    option_expiry: A rank 1 `Tensor` of real dtype specifying the time to
      expiry of each option.
    bond_maturity: A rank 1 `Tensor` of real dtype specifying the time to
      maturity of underlying zero coupon bonds.
    dim: Dimensionality of the Hull-White process.

  Returns:
    A rank 1 `Tensor` of same dtype and shape as the inputs with computed
    Black-equivalent variance for the underlying options.
  """
  # pylint: disable=protected-access
  if model._sample_with_generic:
    raise ValueError('The paramerization of `mean_reversion` and/or '
                     '`volatility` does not support analytic computation '
                     'of bond option variance.')
  mean_reversion = model.mean_reversion(option_expiry)
  volatility = model.volatility(option_expiry)

  option_expiry = tf.repeat(tf.expand_dims(option_expiry, axis=0), dim, axis=0)
  bond_maturity = tf.repeat(tf.expand_dims(bond_maturity, axis=0), dim, axis=0)

  var_between_vol_knots = model._variance_int(model._padded_knots,
                                              model._jump_locations,
                                              model._jump_values_vol,
                                              model._jump_values_mr)
  varx_at_vol_knots = tf.concat(
      [model._zero_padding,
       utils.cumsum_using_matvec(var_between_vol_knots)],
      axis=1)

  time_index = tf.searchsorted(model._jump_locations, option_expiry)
  vn = tf.concat(
      [model._zero_padding,
       model._jump_locations], axis=1)

  var_expiry = model._variance_int(
      tf.gather(vn, time_index, batch_dims=1), option_expiry,
      volatility, mean_reversion)
  var_expiry = var_expiry + tf.gather(
      varx_at_vol_knots, time_index, batch_dims=1)
  var_expiry = var_expiry * (
      tf.math.exp(-mean_reversion*option_expiry) - tf.math.exp(
          -mean_reversion*bond_maturity))**2 / mean_reversion**2
  # gpylint: enable=protected-access
  return var_expiry
