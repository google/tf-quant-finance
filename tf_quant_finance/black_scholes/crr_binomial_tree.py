# Copyright 2019 Google LLC
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
"""Black Scholes prices of options using CRR binomial trees."""

import tensorflow.compat.v2 as tf


# TODO(b/150447187): Generalize to time dependent parameters.
def option_price_binomial(*,
                          volatilities,
                          strikes,
                          expiries,
                          spots,
                          discount_rates=None,
                          dividend_rates=None,
                          is_call_options=None,
                          is_american=None,
                          num_steps=100,
                          dtype=None,
                          name=None):
  """Computes the BS price for a batch of European or American options.

  Uses the Cox-Ross-Rubinstein version of the binomial tree method to compute
  the price of American or European options. Supports batching of the options
  and allows mixing of European and American style exercises in a batch.
  For more information about the binomial tree method and the
  Cox-Ross-Rubinstein method in particular see the references below.

  #### Example

  ```python
  # Prices 5 options with a mix of Call/Put, American/European features
  # in a single batch.
  dtype = np.float64
  spots = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)
  volatilities = np.array([0.1, 0.22, 0.32, 0.01, 0.4], dtype=dtype)
  is_call_options = np.array([True, True, False, False, False])
  is_american = np.array([False, True, True, False, True])
  discount_rates = np.array(0.035, dtype=dtype)
  dividend_rates = np.array([0.02, 0.0, 0.07, 0.01, 0.0], dtype=dtype)
  expiries = np.array(1.0, dtype=dtype)

  prices = option_price_binomial(
      volatilities=volatilities,
      strikes=strikes,
      expiries=expiries,
      spots=spots,
      discount_rates=discount_rates,
      dividend_rates=dividend_rates,
      is_call_options=is_call_options,
      is_american=is_american,
      dtype=dtype)
  # Prints [0., 0.0098847, 0.41299509, 0., 0.06046989]
  ```

  #### References

  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
  [2] Wikipedia contributors. Binomial Options Pricing Model. Available at:
    https://en.wikipedia.org/wiki/Binomial_options_pricing_model

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
    discount_rates: An optional real `Tensor` of same dtype as the
      `volatilities`. The risk free discount rate. If None the rate is assumed
      to be 0.
      Default value: None, equivalent to discount rates = 0..
    dividend_rates: An optional real `Tensor` of same dtype as the
      `volatilities`. If None the rate is assumed to be 0.
      Default value: None, equivalent to discount rates = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with
      `volatilities`. Indicates whether the option is a call (if True) or a put
      (if False). If not supplied, call options are assumed.
      Default value: None, equivalent to is_call_options = True.
    is_american: A boolean `Tensor` of a shape compatible with `volatilities`.
      Indicates whether the option exercise style is American (if True) or
      European (if False). If not supplied, European style exercise is assumed.
      Default value: None, equivalent to is_american = False.
    num_steps: A positive scalar int32 `Tensor`. The size of the time
      discretization to use.
      Default value: 100.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `option_price`.

  Returns:
    A `Tensor` of the same shape as the inferred batch shape of the input data.
    The Black Scholes price of the options computed on a binomial tree.
  """
  with tf.name_scope(name or 'crr_option_price'):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = strikes.dtype
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')

    if discount_rates is None:
      discount_rates = tf.zeros_like(volatilities)
    else:
      discount_rates = tf.convert_to_tensor(
          discount_rates, dtype=dtype, name='discount_rates')
    if dividend_rates is None:
      dividend_rates = tf.zeros_like(volatilities)
    else:
      dividend_rates = tf.convert_to_tensor(
          dividend_rates, dtype=dtype, name='dividend_rates')
    if is_call_options is None:
      is_call_options = tf.ones_like(
          volatilities, dtype=tf.bool, name='is_call_options')
    else:
      is_call_options = tf.convert_to_tensor(
          is_call_options, dtype=tf.bool, name='is_call_options')
    if is_american is None:
      is_american = tf.zeros_like(
          volatilities, dtype=tf.bool, name='is_american')
    else:
      is_american = tf.convert_to_tensor(
          is_american, dtype=tf.bool, name='is_american')

    num_steps = tf.cast(num_steps, dtype=dtype)
    dt = expiries / num_steps

    # CRR choices for the up and down move multipliers
    ln_up = volatilities * tf.math.sqrt(dt)
    ln_dn = -ln_up

    # Prepares the spot grid.
    grid_idx = tf.range(num_steps + 1)
    # Stores the grid as shape [input_batch, N + 1] where N = num_steps.
    log_spot_grid_1 = tf.expand_dims(
        tf.math.log(spots) + ln_up * num_steps, axis=-1)
    log_spot_grid_2 = tf.expand_dims(ln_dn - ln_up, axis=-1) * grid_idx
    log_spot_grid = log_spot_grid_1 + log_spot_grid_2

    # Adding the new dimension is to ensure that batch shape is at the front.
    payoff_fn = _get_payoff_fn(
        tf.expand_dims(strikes, axis=-1),
        tf.expand_dims(is_call_options, axis=-1))
    value_mod_fn = _get_value_modifier(
        tf.expand_dims(is_american, axis=-1), payoff_fn)

    # Shape [batch shape, num time steps + 1]
    values_grid = payoff_fn(tf.math.exp(log_spot_grid))

    p_up = tf.math.exp((discount_rates - dividend_rates) * dt + ln_up) - 1
    p_up /= tf.math.exp(2 * ln_up) - 1
    p_up = tf.expand_dims(p_up, axis=-1)
    p_dn = 1 - p_up
    discount_factors = tf.expand_dims(
        tf.math.exp(-discount_rates * dt), axis=-1)
    ln_up = tf.expand_dims(ln_up, axis=-1)

    def one_step_back(current_values, current_log_spot_grid):
      next_values = (current_values[..., 1:] * p_dn
                     + current_values[..., :-1] * p_up)
      next_log_spot_grid = current_log_spot_grid[..., :-1] - ln_up
      next_values = value_mod_fn(next_values, tf.math.exp(next_log_spot_grid))
      return discount_factors * next_values, next_log_spot_grid

    def should_continue(current_values, current_log_spot_grid):
      del current_values, current_log_spot_grid
      return True

    batch_shape = values_grid.shape[:-1]
    pv, _ = tf.while_loop(
        should_continue,
        one_step_back, (values_grid, log_spot_grid),
        maximum_iterations=tf.cast(num_steps, dtype=tf.int32),
        shape_invariants=(tf.TensorShape(batch_shape + [None]),
                          tf.TensorShape(batch_shape + [None])))
    return tf.squeeze(pv, axis=-1)


def _get_payoff_fn(strikes, is_call_options):
  """Constructs the payoff functions."""
  option_signs = tf.cast(is_call_options, dtype=strikes.dtype) * 2 - 1

  def payoff(spots):
    """Computes payff for the specified options given the spot grid.

    Args:
      spots: Tensor of shape [batch_size, grid_size, 1]. The spot values at some
        time.

    Returns:
      Payoffs for exercise at the specified strikes.
    """
    return tf.nn.relu((spots - strikes) * option_signs)

  return payoff


def _get_value_modifier(is_american, payoff_fn):
  """Constructs the value modifier for american style exercise."""

  def modifier(values, spots):
    immediate_exercise_value = payoff_fn(spots)
    return tf.where(is_american,
                    tf.math.maximum(immediate_exercise_value, values), values)

  return modifier
