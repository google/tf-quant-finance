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
"""Pricing of Interest rate Caps/Floors using Hull-White model."""

import tensorflow.compat.v2 as tf
from tf_quant_finance.models.hull_white import zero_coupon_bond_option as zcb


def cap_floor_price(*,
                    strikes,
                    expiries,
                    maturities,
                    daycount_fractions,
                    reference_rate_fn,
                    dim,
                    mean_reversion,
                    volatility,
                    notional=1.0,
                    is_cap=True,
                    use_analytic_pricing=True,
                    num_samples=1,
                    random_type=None,
                    seed=None,
                    skip=0,
                    time_step=None,
                    dtype=None,
                    name=None):
  """Calculates the prices of interest rate Caps/Floors using Hull-White model.

  An interest Cap (or Floor) is a portfolio of call (or put) options where the
  underlying for the individual options are successive forward rates. The
  individual options comprising a Cap are called Caplets and the corresponding
  options comprising a Floor are called Floorlets. For example, a
  caplet on forward rate `F(T_i, T_{i+1})` has the following payoff at time
  `T_{i_1}`:

  ```None

   caplet payoff = tau_i * max[F(T_i, T_{i+1}) - X, 0]

  ```
  where where `X` is the strake rate and `tau_i` is the daycount fraction. The
  caplet payoff (at `T_{i+1}`) can be expressed as the following at `T_i`:

  ```None

  caplet_payoff = (1.0 + tau_i * X) *
                  max[1.0 / (1 + tau_i * X) - P(T_i, T_{t+1}), 0]

  ```

  where `P(T_i, T_{i+1})` is the price at `T_i` of a zero coupon bond with
  maturity `T_{i+1}. Thus, a caplet can be priced as a put option on zero
  coupon bond [1].

  #### References
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007.

  #### Example
  The example shows how value a batch containing spot starting 1-year and
  2-year Caps and with quarterly frequency.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  reference_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  expiries = np.array([[0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
                       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]])
  maturities = np.array([[0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]])
  strikes = 0.01 * np.ones_like(expiries)
  daycount_fractions = np.array([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
  price = tff.models.hull_white.cap_floor_price(
      strikes=strikes,
      expiries=expiries,
      maturities=maturities,
      daycount_fractions=daycount_fractions,
      notional=1.0e6,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.02],
      reference_rate_fn=reference_rate_fn,
      use_analytic_pricing=True,
      dtype=dtype)
  # Expected value: [[0.4072088281493774], [1.3031872853339002]]
  ````

  Args:
    strikes: A real `Tensor` of any shape and dtype. The strike rate of the
      caplets or floorlets. The shape of this input determines the number
      (and shape) of the options to be priced and the shape of the output. For
      an N-dimensional input `Tensor`, the first N-1 dimensions correspond to
      the batch dimension, i.e., the distinct caps and floors and the last
      dimension correspond to the caplets or floorlets contained with an
      intrument.
    expiries: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The reset time of each caplet (or floorlet).
    maturities: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The maturity time of each caplet (or floorlet) and also the
      time at which payment is made.
    daycount_fractions: A real `Tensor` of the same dtype and compatible shape
      as `strikes`. The daycount fractions associated with the underlying
      forward rates.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape either `input_shape` or
      `input_shape + [dim]`. Returns the continuously compounded zero rate at
      the present time for the input expiry time.
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
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the cap (or floor).
       Default value: None in which case the notional is set to 1.
    is_cap: A boolean `Tensor` of a shape compatible with `strikes`. Indicates
      whether the option is a Cap (if True) or a Floor (if False). If not
      supplied, Caps are assumed.
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
      `hw_cap_floor_price`.

  Returns:
    A `Tensor` of real dtype and shape  strikes.shape[:-1] + [dim] containing
    the computed option prices. For caplets that have reset in the past
    (expiries<0), the function sets the corresponding caplet prices to 0.0.
  """
  name = name or 'hw_cap_floor_price'
  with tf.name_scope(name):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = dtype or strikes.dtype
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    maturities = tf.convert_to_tensor(maturities, dtype=dtype,
                                      name='maturities')
    daycount_fractions = tf.convert_to_tensor(daycount_fractions, dtype=dtype,
                                              name='daycount_fractions')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    is_cap = tf.convert_to_tensor(is_cap, dtype=tf.bool, name='is_cap')
    is_call_options = ~is_cap
    bond_option_strikes = 1.0 / (1.0 + daycount_fractions * strikes)

    # The dimension of `caplet_prices` is going to be strikes.shape + [dim]
    caplet_prices = zcb.bond_option_price(
        strikes=bond_option_strikes,
        expiries=expiries,
        maturities=maturities,
        discount_rate_fn=reference_rate_fn,
        dim=dim,
        mean_reversion=mean_reversion,
        volatility=volatility,
        is_call_options=is_call_options,
        use_analytic_pricing=use_analytic_pricing,
        num_samples=num_samples,
        random_type=random_type,
        seed=seed,
        skip=skip,
        time_step=time_step,
        dtype=dtype,
        name=name + '_bond_option')

    # Make sure we have the proper output shape when single cap is valued.
    keep_dims = expiries.shape.rank <= 1

    # Expand dims because `caplet_prices` has an additional dimension.
    expiries = tf.expand_dims(expiries, axis=-1)
    strikes = tf.expand_dims(strikes, axis=-1)
    daycount_fractions = tf.expand_dims(daycount_fractions, axis=-1)
    caplet_prices = tf.where(
        expiries < 0.0, tf.zeros_like(expiries), caplet_prices)

    axis_to_aggregate = caplet_prices.shape.rank - 2
    cap_prices = tf.math.reduce_sum(
        notional * (1.0 + daycount_fractions * strikes) * caplet_prices,
        axis=axis_to_aggregate, keepdims=keep_dims)
    return cap_prices
