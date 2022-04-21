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
"""Pricing of zero coupon bond options using Heath-Jarrow-Morton model."""

from typing import Callable, Union

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.math import random
from tf_quant_finance.models.hjm import quasi_gaussian_hjm
from tf_quant_finance.models.hjm import zero_coupon_bond_option_util

__all__ = [
    'bond_option_price'
]


def bond_option_price(
    *,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    maturities: types.RealTensor,
    discount_rate_fn: Callable[..., types.RealTensor],
    dim: int,
    mean_reversion: types.RealTensor,
    volatility: Union[types.RealTensor, Callable[..., types.RealTensor]],
    corr_matrix: types.RealTensor = None,
    is_call_options: types.BoolTensor = True,
    num_samples: types.IntTensor = 1,
    random_type: random.RandomType = None,
    seed: types.IntTensor = None,
    skip: types.IntTensor = 0,
    time_step: types.RealTensor = None,
    dtype: tf.DType = None,
    name: str = None) -> types.RealTensor:
  """Calculates European bond option prices using the HJM model.

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
  price = tff.models.hjm.bond_option_price(
      strikes=strikes,
      expiries=expiries,
      maturities=maturities,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.02],
      discount_rate_fn=discount_rate_fn,
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
      `Tensor` and returns a `Tensor` of shape `input_shape`. Computes the
      zero coupon bond yield at the present time for the input expiry time.
    dim: A Python scalar which corresponds to the number of factors within a
      single HJM model.
    mean_reversion: A real positive `Tensor` of shape `[dim]`. Corresponds to
      the mean reversion rate of each factor.
    volatility: A real positive `Tensor` of the same `dtype` and shape as
      `mean_reversion` or a callable with the following properties: (a)  The
        callable should accept a scalar `Tensor` `t` and a 1-D `Tensor` `r(t)`
        of shape `[num_samples]` and returns a 2-D `Tensor` of shape
        `[num_samples, dim]`. The variable `t`  stands for time and `r(t)` is
        the short rate at time `t`.  The function returns instantaneous
        volatility `sigma(t) = sigma(t, r(t))`. When `volatility` is specified
        is a real `Tensor`, each factor is assumed to have a constant
        instantaneous volatility  and the  model is effectively a Gaussian HJM
        model. Corresponds to the instantaneous volatility of each factor.
    corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as
      `mean_reversion`. Corresponds to the correlation matrix `Rho`.
      Default value: None, meaning the factors are uncorrelated.
    is_call_options: A boolean `Tensor` of a shape compatible with `strikes`.
      Indicates whether the option is a call (if True) or a put (if False). If
      not supplied, call options are assumed.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random number
      generator to use to generate the simulation paths.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,
      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,
      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be an Python
      integer. For `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as
      an integer `Tensor` of shape `[2]`.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation.
      Default value: `None`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this class.
      Default value: `None` which maps to the default name
        `hw_bond_option_price`.

  Returns:
    A `Tensor` of real dtype and shape  `strikes.shape` containing the
    computed option prices.
  """
  if time_step is None:
    raise ValueError('`time_step` must be provided for simulation based '
                     'bond option valuation.')

  name = name or 'hjm_bond_option_price'
  if dtype is None:
    dtype = tf.convert_to_tensor([0.0]).dtype
  with tf.name_scope(name):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    maturities = tf.convert_to_tensor(
        maturities, dtype=dtype, name='maturities')
    is_call_options = tf.convert_to_tensor(
        is_call_options, dtype=tf.bool, name='is_call_options')
    model = quasi_gaussian_hjm.QuasiGaussianHJM(
        dim,
        mean_reversion=mean_reversion,
        volatility=volatility,
        initial_discount_rate_fn=discount_rate_fn,
        corr_matrix=corr_matrix,
        dtype=dtype)

    def sample_discount_curve_paths_fn(times, curve_times, num_samples):
      p_t_tau, r_t, _ = model.sample_discount_curve_paths(
          times=times,
          curve_times=curve_times,
          num_samples=num_samples,
          random_type=random_type,
          time_step=time_step,
          seed=seed,
          skip=skip)
      p_t_tau = tf.expand_dims(p_t_tau, axis=-1)
      r_t = tf.expand_dims(r_t, axis=-1)
      return p_t_tau, r_t

    # Shape batch_shape + [1]
    prices = zero_coupon_bond_option_util.options_price_from_samples(
        strikes,
        expiries,
        maturities,
        is_call_options,
        sample_discount_curve_paths_fn,
        num_samples,
        time_step,
        dtype=dtype)
    # Shape batch_shape
    return tf.squeeze(prices, axis=-1)
