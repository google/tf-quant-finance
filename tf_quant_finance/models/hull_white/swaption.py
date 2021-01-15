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
"""Pricing of Interest rate Swaptions using the Hull-White model."""

import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.experimental.lsm_algorithm import lsm_v2
from tf_quant_finance.math import pde
from tf_quant_finance.math import root_search
from tf_quant_finance.math.interpolation import linear
from tf_quant_finance.models import utils
from tf_quant_finance.models.hull_white import vector_hull_white
from tf_quant_finance.models.hull_white import zero_coupon_bond_option as zcb

__all__ = ['swaption_price', 'bermudan_swaption_price']

# Points smaller than this are merged together in FD time grid
_PDE_TIME_GRID_TOL = 1e-7


def _jamshidian_decomposition(hw_model,
                              expiries,
                              maturities,
                              coefficients,
                              dtype,
                              name=None):
  """Jamshidian decomposition for European swaption valuation.

  Jamshidian decomposition is a widely used technique for the valuation of
  European swaptions (and options on coupon bearing bonds) when the underlying
  models for the term structure are short rate models (such as Hull-White
  model). The method transforms the swaption valuation to the valuation of a
  portfolio of call (put) options on zero-coupon bonds.

  Consider the following swaption payoff(assuming unit notional) at the
  exipration time (under a single curve valuation):

  ```None
  payoff = max(1 - P(T0, TN, r(T0)) - sum_1^N tau_i * X_i * P(T0, Ti, r(T0)), 0)
         = max(1 - sum_0^N alpha_i * P(T0, Ti, r(T0)), 0)
  ```

  where `T0` denotes the swaption expiry, P(T0, Ti, r(T0)) denotes the price
  of the zero coupon bond at `T0` with maturity `Ti` and `r(T0)` is the short
  rate at time `T0`. If `r*` (or breakeven short rate) is the solution of the
  following equation:

  ```None
  1 - sum_0^N alpha_i * P(T0, Ti, r*) = 0            (1)
  ```

  Then the swaption payoff can be expressed as the following (Ref. [1]):

  ```None
  payoff = sum_1^N alpha_i max(P(T0, Ti, r*) - P(T0, Ti), 0)
  ```
  where in the above formulation the swaption payoff is the same as that of
  a portfolio of bond options with strikes `P(T0, Ti, r*)`.

  The function accepts relevant inputs for the above computation and returns
  the strikes of the bond options computed using the Jamshidian decomposition.

  #### References:
    [1]: Leif B. G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling.
    Volume II: Term Structure Models. Chapter 10.


  Args:
    hw_model: An instance of `VectorHullWhiteModel`. The model used for the
      valuation.
    expiries: A real `Tensor` of any shape and dtype. The the time to
      expiration of the swaptions.
    maturities: A real `Tensor` of same shape and dtype as `expiries`. The
      payment times for fixed payments of the underlying swaptions.
    coefficients: A real `Tensor` of shape `expiries.shape + [n]` where `n`
      denotes the number of payments in the fixed leg of the underlying swaps.
    dtype: The default dtype to use when converting values to `Tensor`s.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `jamshidian_decomposition`.

  Returns:
    A real `Tensor` of shape expiries.shape + [dim] containing the forward
    bond prices computed at the breakeven short rate using the Jamshidian
    decomposition. `dim` stands for the dimensionality of the Hull-White
    process.
  """

  name = name or 'jamshidian_decomposition'
  with tf.name_scope(name):
    dim = hw_model.dim()
    coefficients = tf.expand_dims(coefficients, axis=-1)

    def _zero_fun(x):
      # Get P(t0, t, r(t0)).
      p_t0_t = hw_model.discount_bond_price(x, expiries, maturities)
      # return_value.shape = batch_shape + [1] + [dim]
      return_value = tf.reduce_sum(
          coefficients * p_t0_t, axis=-2, keepdims=True) + [1.0]
      return return_value

    swap_shape = expiries.shape.as_list()[:-1] + [1] + [dim]
    lower_bound = -1 * tf.ones(swap_shape, dtype=dtype)
    upper_bound = 1 * tf.ones(swap_shape, dtype=dtype)
    # Solve Eq.(1)
    brent_results = root_search.brentq(_zero_fun, lower_bound, upper_bound)
    breakeven_short_rate = brent_results.estimated_root
    return hw_model.discount_bond_price(breakeven_short_rate, expiries,
                                        maturities)


def _prepare_swaption_indices(tensor_shape):
  """Indices for `gather_nd` for analytic valuation.

  For a `Tensor` x of shape `tensor_shape` = [n] + batch_shape + [n], this
  function returns indices for tf.gather_nd to get `x[i,...,i]`

  Args:
    tensor_shape: A list of length `k` representing shape of the `Tensor`.

  Returns:
    A `Tensor` of shape (num_elements, k) where num_elements= n * batch_size
    of dtype tf.int64.
  """

  tensor_shape = np.array(tensor_shape, dtype=np.int64)
  batch_shape = tensor_shape[1:-1]
  batch_size = np.prod(batch_shape)
  index_list = []
  for i in range(len(tensor_shape)):
    index = np.arange(0, tensor_shape[i], dtype=np.int64)
    if i == 0 or i == len(tensor_shape) - 1:
      index = tf.tile(index, [batch_size])
    else:
      index = np.tile(
          np.repeat(index, np.prod(tensor_shape[i+1:])),
          [np.prod(tensor_shape[1:i])])
    index_list.append(index)

  return tf.stack(index_list, axis=-1)


def _prepare_indices_ijjk(idx0, idx1, idx2, idx3):
  """Prepares indices to get x[i, j, j, k]."""
  # For a 4-D `Tensor` x, creates indices for tf.gather_nd to retrieve
  # x[i, j, j, k].
  len0 = tf.shape(idx0)[0]
  len1 = tf.shape(idx1)[0]
  len3 = tf.shape(idx3)[0]
  idx0 = tf.repeat(idx0, len1 * len3)
  idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
  idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
  idx3 = tf.tile(idx3, [len0 * len1])

  return tf.stack([idx0, idx1, idx2, idx3], axis=-1)


def _prepare_indices_ijj(idx0, idx1, idx2):
  """Prepares indices to get x[i, j, j]."""
  # For a 3-D `Tensor` x, creates indices for tf.gather_nd to retrieve
  # x[i, j, j].
  len0 = tf.shape(idx0)[0]
  len1 = tf.shape(idx1)[0]
  idx0 = tf.repeat(idx0, len1)
  idx1 = tf.tile(idx1, [len0])
  idx2 = tf.tile(idx2, [len0])

  return tf.stack([idx0, idx1, idx2], axis=-1)


def _map_payoff_to_sim_times(indices, payoff, num_samples):
  """Maps the swaption payoffs to short rate simulation times.

  Swaption payoffs are calculated on bermudan swaption's expiries. However, for
  the LSM algorithm, we need short rate simulations and swaption payoffs at
  the union of all exercise times in the batch of swaptions. This function
  takes the payoff of individual swaption at their respective exercise times
  and maps it to all simulation times. This is done by setting the payoff to
  -1 whenever the simulation time is not equal to the swaption exercise time.

  Args:
    indices: A `Tensor` of shape `batch_shape + num_exercise_times` containing
      the index of exercise time in the vector of simulation times.
    payoff: A real tensor of shape
      `[num_samples] + batch_shape + num_exercise_times` containing the
      exercise value of the underlying swap on each exercise time.
    num_samples: A scalar `Tensor` specifying the number of samples on which
      swaption payoff is computed.

  Returns:
    A tuple of `Tensors`. The first tensor is a integer `Tensor` of shape
    `[num_samples] + batch_shape + [num_simulation_times]` and contains `1`
    if the corresponding simulation time is one of the exercise times for the
    swaption. The second `Tensor` is a real `Tensor` of same shape and contains
    the exercise value of the swaption if the corresponding simulation time is
    an exercise time for the swaption or -1 otherwise.
  """
  indices = tf.expand_dims(indices, axis=0)
  indices = tf.repeat(indices, num_samples, axis=0)
  index_list = list()
  tensor_shape = np.array(indices.shape.as_list())
  output_shape = indices.shape.as_list()[:-1] + [
      tf.math.reduce_max(indices) + 1
  ]
  num_elements = np.prod(tensor_shape)
  for dim, _ in enumerate(tensor_shape[:-1]):
    idx = tf.range(0, tensor_shape[dim], dtype=indices.dtype)
    idx = tf.tile(tf.repeat(idx, np.prod(tensor_shape[dim + 1:])),
                  [np.prod(tensor_shape[:dim])])
    index_list.append(idx)

  index_list.append(tf.reshape(indices, [-1]))
  # We need to transform `payoff` from the initial shape of
  # [num_samples, batch_shape, num_exercise_times] to a new `Tensor` with
  # shape = [num_samples, batch_shape, num_exercise_times] such that
  # payoff_new[..., indices] = payoff
  # We achieve this by first creating a `payoff_new` as a SparseTensor with
  # nonzero values at appropriate indices based on the payoff_new.shape and
  # then converting the sparse tenson to dense tensor.
  sparse_indices = tf.cast(tf.stack(index_list, axis=-1), dtype=np.int64)
  is_exercise_time = tf.sparse.to_dense(
      tf.sparse.SparseTensor(sparse_indices, tf.ones(shape=num_elements),
                             output_shape),
      validate_indices=False)
  payoff = tf.sparse.to_dense(
      tf.sparse.SparseTensor(sparse_indices, tf.reshape(payoff, [-1]),
                             output_shape),
      validate_indices=False)
  return is_exercise_time, payoff


def _analytic_valuation(expiries, floating_leg_start_times,
                        floating_leg_end_times, fixed_leg_payment_times,
                        fixed_leg_daycount_fractions, fixed_leg_coupon,
                        reference_rate_fn, dim, mean_reversion, volatility,
                        notional, is_payer_swaption, output_shape,
                        dtype, name):
  """Helper function for analytic valuation."""
  # The below inputs are needed for midcurve swaptions
  del floating_leg_start_times, floating_leg_end_times
  with tf.name_scope(name):
    is_call_options = tf.where(is_payer_swaption,
                               tf.convert_to_tensor(False, dtype=tf.bool),
                               tf.convert_to_tensor(True, dtype=tf.bool))

    model = vector_hull_white.VectorHullWhiteModel(
        dim,
        mean_reversion,
        volatility,
        initial_discount_rate_fn=reference_rate_fn,
        dtype=dtype)
    coefficients = fixed_leg_daycount_fractions * fixed_leg_coupon
    jamshidian_coefficients = tf.concat([
        -coefficients[..., :-1],
        tf.expand_dims(-1.0 - coefficients[..., -1], axis=-1)], axis=-1)

    breakeven_bond_option_strikes = _jamshidian_decomposition(
        model, expiries,
        fixed_leg_payment_times, jamshidian_coefficients, dtype,
        name=name + '_jamshidian_decomposition')

    bond_strike_rank = breakeven_bond_option_strikes.shape.rank
    perm = [bond_strike_rank-1] + [x for x in range(0, bond_strike_rank - 1)]
    breakeven_bond_option_strikes = tf.transpose(
        breakeven_bond_option_strikes, perm=perm)
    bond_option_prices = zcb.bond_option_price(
        strikes=breakeven_bond_option_strikes,
        expiries=expiries,
        maturities=fixed_leg_payment_times,
        discount_rate_fn=reference_rate_fn,
        dim=dim,
        mean_reversion=mean_reversion,
        volatility=volatility,
        is_call_options=is_call_options,
        use_analytic_pricing=True,
        dtype=dtype,
        name=name + '_bond_option')

    # Now compute P(T0, TN) + sum_i (c_i * tau_i * P(T0, Ti))
    # bond_option_prices.shape = [dim] + batch_shape + [m] + [dim], where `m`
    # denotes the number of fixed payments for the underlying swaps.
    swaption_values = (
        tf.reduce_sum(
            bond_option_prices * tf.expand_dims(coefficients, axis=-1),
            axis=-2) + bond_option_prices[..., -1, :])
    swaption_shape = swaption_values.shape
    gather_index = _prepare_swaption_indices(swaption_shape.as_list())
    swaption_values = tf.reshape(
        tf.gather_nd(swaption_values, gather_index), output_shape)
    return notional * swaption_values


def _bermudan_swaption_fd(batch_shape, model, exercise_times,
                          unique_exercise_times, fixed_leg_payment_times,
                          fixed_leg_daycount_fractions, fixed_leg_coupon,
                          notional, is_payer_swaption, time_step_fd,
                          num_grid_points_fd, name, dtype):
  """Price Bermudan swaptions using finite difference."""
  with tf.name_scope(name):
    longest_exercise_time = unique_exercise_times[-1]
    if time_step_fd is None:
      time_step_fd = longest_exercise_time / 100.0
    short_rate_min = -0.2
    short_rate_max = 0.2
    # grid.shape=(num_grid_points,2)
    grid = pde.grids.uniform_grid(
        minimums=[short_rate_min],
        maximums=[short_rate_max],
        sizes=[num_grid_points_fd],
        dtype=dtype)

    pde_time_grid = tf.concat([unique_exercise_times, tf.range(
        0.0, longest_exercise_time, time_step_fd, dtype=dtype)], axis=0)
    # This time grid is now sorted and contains the Bermudan exercise times
    pde_time_grid = tf.sort(pde_time_grid, name='sort_pde_time_grid')
    pde_time_grid_dt = pde_time_grid[1:] - pde_time_grid[:-1]
    pde_time_grid_dt = tf.concat([[100.0], pde_time_grid_dt], axis=-1)
    # Remove duplicates.
    mask = tf.math.greater(pde_time_grid_dt, _PDE_TIME_GRID_TOL)
    pde_time_grid = tf.boolean_mask(pde_time_grid, mask)
    pde_time_grid_dt = tf.boolean_mask(pde_time_grid_dt, mask)

    maturities = fixed_leg_payment_times
    maturities_shape = maturities.shape

    unique_maturities, _ = tf.unique(tf.reshape(maturities, shape=[-1]))
    unique_maturities = tf.sort(unique_maturities, name='sort_maturities')

    num_exercise_times = tf.shape(pde_time_grid)[-1]
    num_maturities = tf.shape(unique_maturities)[-1]

    short_rates = tf.reshape(grid[0], grid[0].shape + [1, 1])
    broadcasted_exercise_times = tf.reshape(
        pde_time_grid, [1] + pde_time_grid.shape + [1])
    broadcasted_maturities = tf.reshape(
        unique_maturities, [1, 1] + unique_maturities.shape)

    # Reshape `short_rate`, `exercise_times` and `maturities` to
    # (num_grid_points, num_exercise_times, num_maturities)
    short_rates = tf.broadcast_to(
        short_rates, grid[0].shape + [num_exercise_times, num_maturities])
    broadcasted_exercise_times = tf.broadcast_to(
        broadcasted_exercise_times,
        grid[0].shape + [num_exercise_times, num_maturities])
    broadcasted_maturities = tf.broadcast_to(
        broadcasted_maturities,
        grid[0].shape + [num_exercise_times, num_maturities])

    # Zero-coupon bond curve
    zcb_curve = model.discount_bond_price(
        tf.expand_dims(short_rates, axis=-1),
        broadcasted_exercise_times, broadcasted_maturities)[..., 0]

    exercise_times_index = tf.searchsorted(
        pde_time_grid, tf.reshape(exercise_times, [-1]))
    maturities_index = tf.searchsorted(
        unique_maturities, tf.reshape(maturities, [-1]))

    # gather_index.shape = (num_grid_points*np.cumprod(maturities_shape), 3)
    gather_index = _prepare_indices_ijj(
        tf.range(0, num_grid_points_fd), exercise_times_index,
        maturities_index)
    zcb_curve = tf.gather_nd(zcb_curve, gather_index)
    # zcb_curve.shape = [num_grid_points_fd] + [maturities_shape]
    zcb_curve = tf.reshape(zcb_curve, [num_grid_points_fd] + maturities_shape)
    # Shape after reduce_sum=(num_grid_points, batch_shape, num_exercise_times)
    fixed_leg = tf.math.reduce_sum(
        fixed_leg_coupon * fixed_leg_daycount_fractions * zcb_curve, axis=-1)
    float_leg = 1.0 - zcb_curve[..., -1]
    payoff_at_exercise = float_leg - fixed_leg
    payoff_at_exercise = tf.where(is_payer_swaption, payoff_at_exercise,
                                  -payoff_at_exercise)

    unrepeated_exercise_times = exercise_times[..., -1]
    exercise_times_index = tf.searchsorted(
        pde_time_grid, tf.reshape(unrepeated_exercise_times, [-1]))
    _, payoff_swap = _map_payoff_to_sim_times(
        tf.reshape(exercise_times_index, unrepeated_exercise_times.shape),
        payoff_at_exercise, num_grid_points_fd)

    # payoff_swap.shape=(num_grid_points_fd, batch_shape, num_exercise_times).
    # Transpose so that the [num_grid_points_fd] is the last dimension; this is
    # needed for broadcasting inside the PDE solver.
    payoff_swap = tf.transpose(payoff_swap)

    def _get_index(t, tensor_to_search):
      t = tf.expand_dims(t, axis=-1)
      index = tf.searchsorted(tensor_to_search, t - _PDE_TIME_GRID_TOL, 'right')
      y = tf.gather(tensor_to_search, index)
      return tf.where(tf.math.abs(t - y) < _PDE_TIME_GRID_TOL, index, -1)[0]

    def _second_order_coeff_fn(t, grid):
      del grid
      return [[model.volatility(t)**2 / 2]]

    def _first_order_coeff_fn(t, grid):
      s = grid[0]
      return [model.drift_fn()(t, s)]

    def _zeroth_order_coeff_fn(t, grid):
      del t
      return -grid[0]

    @pde.boundary_conditions.dirichlet
    def _lower_boundary_fn(t, grid):
      del grid
      index = _get_index(t, pde_time_grid)
      result = tf.where(index > -1, payoff_swap[index, ..., 0], 0.0)
      return tf.where(is_payer_swaption, 0.0, result)

    @pde.boundary_conditions.dirichlet
    def _upper_boundary_fn(t, grid):
      del grid
      index = _get_index(t, pde_time_grid)
      result = tf.where(index > -1, payoff_swap[index, ..., 0], 0.0)
      return tf.where(is_payer_swaption, result, 0.0)

    def _final_value():
      return tf.nn.relu(payoff_swap[-1])

    def _values_transform_fn(t, grid, value_grid):
      index = _get_index(t, pde_time_grid)
      v_star = tf.where(
          index > -1, tf.nn.relu(payoff_swap[index]), 0.0)
      return grid, tf.maximum(value_grid, v_star)

    def _pde_time_step(t):
      index = _get_index(t, pde_time_grid)
      dt = pde_time_grid_dt[index]
      return dt

    res = pde.fd_solvers.solve_backward(
        longest_exercise_time,
        0.0,
        grid,
        values_grid=_final_value(),
        time_step=_pde_time_step,
        boundary_conditions=[(_lower_boundary_fn, _upper_boundary_fn)],
        values_transform_fn=_values_transform_fn,
        second_order_coeff_fn=_second_order_coeff_fn,
        first_order_coeff_fn=_first_order_coeff_fn,
        zeroth_order_coeff_fn=_zeroth_order_coeff_fn,
        dtype=dtype)

    r0 = model.instant_forward_rate(0.0)
    option_value = linear.interpolate(r0, res[1], res[0])
    return tf.reshape(notional * tf.transpose(option_value), batch_shape)


def swaption_price(*,
                   expiries,
                   floating_leg_start_times,
                   floating_leg_end_times,
                   fixed_leg_payment_times,
                   floating_leg_daycount_fractions,
                   fixed_leg_daycount_fractions,
                   fixed_leg_coupon,
                   reference_rate_fn,
                   dim,
                   mean_reversion,
                   volatility,
                   notional=None,
                   is_payer_swaption=None,
                   use_analytic_pricing=True,
                   num_samples=1,
                   random_type=None,
                   seed=None,
                   skip=0,
                   time_step=None,
                   dtype=None,
                   name=None):
  """Calculates the price of European Swaptions using the Hull-White model.

  A European Swaption is a contract that gives the holder an option to enter a
  swap contract at a future date at a prespecified fixed rate. A swaption that
  grants the holder to pay fixed rate and receive floating rate is called a
  payer swaption while the swaption that grants the holder to receive fixed and
  pay floating payments is called the receiver swaption. Typically the start
  date (or the inception date) of the swap concides with the expiry of the
  swaption. Mid-curve swaptions are currently not supported (b/160061740).

  Analytic pricing of swaptions is performed using the Jamshidian decomposition
  [1].

  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007.

  #### Example
  The example shows how value a batch of 1y x 1y and 1y x 2y swaptions using the
  Hull-White model.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  expiries = [1.0, 1.0]
  float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],
                            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]
  float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
  fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
  float_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
  fixed_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
  fixed_leg_coupon = [[0.011, 0.011, 0.011, 0.011, 0.0, 0.0, 0.0, 0.0],
                      [0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011]]
  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  price = tff.models.hull_white.swaption_price(
      expiries=expiries,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.02],
      dtype=dtype)
  # Expected value: [[0.7163243383624043], [1.4031415262337608]] # shape = (2,1)
  ````

  Args:
    expiries: A real `Tensor` of any shape and dtype. The time to
      expiration of the swaptions. The shape of this input determines the number
      (and shape) of swaptions to be priced and the shape of the output.
    floating_leg_start_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual begins for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual ends for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes
      the number of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg.
    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `floating_leg_start_times`. The daycount fractions
      for each payment in the floating leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions
      for each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape
      as `fixed_leg_payment_times`. The fixed rate for each payment in the
      fixed leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape + [dim]`. Returns
      the continuously compounded zero rate at the present time for the input
      expiry time.
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
      `strikes`specifying the notional amount for the underlying swap.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the swaption is a payer (if True) or a receiver
      (if False) swaption. If not supplied, payer swaptions are assumed.
    use_analytic_pricing: A Python boolean specifying if analytic valuation
      should be performed. Analytic valuation is only supported for constant
      `mean_reversion` and piecewise constant `volatility`. If the input is
      `False`, then valuation using Monte-Carlo simulations is performed.
      Default value: The default value is `True`.
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
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `hw_swaption_price`.

  Returns:
    A `Tensor` of real dtype and shape  expiries.shape + [dim] containing the
    computed swaption prices. For swaptions that have. reset in the past
    (expiries<0), the function sets the corresponding option prices to 0.0.
  """
  # TODO(b/160061740): Extend the functionality to support mid-curve swaptions.
  name = name or 'hw_swaption_price'
  del floating_leg_daycount_fractions
  with tf.name_scope(name):
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    dtype = dtype or expiries.dtype
    float_leg_start_times = tf.convert_to_tensor(
        floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
    float_leg_end_times = tf.convert_to_tensor(
        floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions, dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    notional = tf.expand_dims(
        tf.broadcast_to(notional, expiries.shape), axis=-1)
    if is_payer_swaption is None:
      is_payer_swaption = True
    is_payer_swaption = tf.convert_to_tensor(
        is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')

    output_shape = expiries.shape.as_list() + [dim]
    # Add a dimension corresponding to multiple cashflows in a swap
    if expiries.shape.rank == fixed_leg_payment_times.shape.rank - 1:
      expiries = tf.expand_dims(expiries, axis=-1)
    elif expiries.shape.rank < fixed_leg_payment_times.shape.rank - 1:
      raise ValueError('Swaption expiries not specified for all swaptions '
                       'in the batch. Expected rank {} but received {}.'.format(
                           fixed_leg_payment_times.shape.rank - 1,
                           expiries.shape.rank))

    # Expected shape: batch_shape + [m], same as fixed_leg_payment_times.shape
    # We need to explicitly use tf.repeat because we need to price
    # batch_shape + [m] bond options with different strikes along the last
    # dimension.
    expiries = tf.repeat(
        expiries, tf.shape(fixed_leg_payment_times)[-1], axis=-1)

    if use_analytic_pricing:
      return _analytic_valuation(expiries, float_leg_start_times,
                                 float_leg_end_times, fixed_leg_payment_times,
                                 fixed_leg_daycount_fractions,
                                 fixed_leg_coupon, reference_rate_fn,
                                 dim, mean_reversion, volatility, notional,
                                 is_payer_swaption, output_shape, dtype,
                                 name + '_analytic_valuation')

    # Monte-Carlo pricing
    model = vector_hull_white.VectorHullWhiteModel(
        dim,
        mean_reversion,
        volatility,
        initial_discount_rate_fn=reference_rate_fn,
        dtype=dtype)

    if time_step is None:
      raise ValueError('`time_step` must be provided for simulation '
                       'based bond option valuation.')

    sim_times, _ = tf.unique(tf.reshape(expiries, shape=[-1]))
    longest_expiry = tf.reduce_max(sim_times)
    sim_times, _ = tf.unique(tf.concat([sim_times, tf.range(
        time_step, longest_expiry, time_step)], axis=0))
    sim_times = tf.sort(sim_times, name='sort_sim_times')

    maturities = fixed_leg_payment_times
    swaptionlet_shape = maturities.shape
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

    dt = tf.concat(
        [tf.convert_to_tensor([0.0], dtype=dtype),
         sim_times[1:] - sim_times[:-1]], axis=0)
    dt = tf.expand_dims(tf.expand_dims(dt, axis=-1), axis=0)
    discount_factors_builder = tf.math.exp(-r_t * dt)
    # Transpose before (and after) because we want the cumprod along axis=1
    # and `matvec` operates on the last axis.
    discount_factors_builder = tf.transpose(
        utils.cumprod_using_matvec(
            tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])

    # make discount factors the same shape as `p_t_tau`. This involves adding
    # an extra dimenstion (corresponding to `curve_times`).
    discount_factors_builder = tf.expand_dims(
        discount_factors_builder,
        axis=1)
    # tf.repeat is needed because we will use gather_nd later on this tensor.
    discount_factors_simulated = tf.repeat(
        discount_factors_builder, tf.shape(p_t_tau)[1], axis=1)

    # `sim_times` and `curve_times` are sorted for simulation. We need to
    # select the indices corresponding to our input.
    sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
    curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

    gather_index = _prepare_indices_ijjk(
        tf.range(0, num_samples), curve_time_index, sim_time_index,
        tf.range(0, dim))

    # The shape after `gather_nd` will be `(num_samples*num_swaptionlets*dim,)`
    payoff_discount_factors_builder = tf.gather_nd(
        discount_factors_simulated, gather_index)
    # Reshape to `[num_samples] + swaptionlet.shape + [dim]`
    payoff_discount_factors = tf.reshape(
        payoff_discount_factors_builder,
        [num_samples] + swaptionlet_shape + [dim])
    payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
    payoff_bond_price = tf.reshape(
        payoff_bond_price_builder, [num_samples] + swaptionlet_shape + [dim])

    # Add an axis corresponding to `dim`
    fixed_leg_pv = tf.expand_dims(
        fixed_leg_coupon * fixed_leg_daycount_fractions,
        axis=-1) * payoff_bond_price
    # Sum fixed coupon payments within each swap
    fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
    float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
    payoff_swap = payoff_discount_factors[..., -1, :] * (
        float_leg_pv - fixed_leg_pv)
    payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
    payoff_swaption = tf.math.maximum(payoff_swap, 0.0)
    option_value = tf.reshape(
        tf.math.reduce_mean(payoff_swaption, axis=0), output_shape)

    return notional * option_value


def bermudan_swaption_price(*,
                            exercise_times,
                            floating_leg_start_times,
                            floating_leg_end_times,
                            fixed_leg_payment_times,
                            floating_leg_daycount_fractions,
                            fixed_leg_daycount_fractions,
                            fixed_leg_coupon,
                            reference_rate_fn,
                            dim,
                            mean_reversion,
                            volatility,
                            notional=None,
                            is_payer_swaption=None,
                            use_finite_difference=False,
                            lsm_basis=None,
                            num_samples=100,
                            random_type=None,
                            seed=None,
                            skip=0,
                            time_step=None,
                            time_step_finite_difference=None,
                            num_grid_points_finite_difference=100,
                            dtype=None,
                            name=None):
  """Calculates the price of Bermudan Swaptions using the Hull-White model.

  A Bermudan Swaption is a contract that gives the holder an option to enter a
  swap contract on a set of future exercise dates. The exercise dates are
  typically the fixing dates (or a subset thereof) of the underlying swap. If
  `T_N` denotes the final payoff date and `T_i, i = {1,...,n}` denote the set
  of exercise dates, then if the option is exercised at `T_i`, the holder is
  left with a swap with first fixing date equal to `T_i` and maturity `T_N`.

  Simulation based pricing of Bermudan swaptions is performed using the least
  squares Monte-carlo approach [1].

  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007.

  #### Example
  The example shows how value a batch of 5-no-call-1 and 5-no-call-2
  swaptions using the Hull-White model.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  exercise_swaption_1 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
  exercise_swaption_2 = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]
  exercise_times = [exercise_swaption_1, exercise_swaption_2]

  float_leg_start_times_1y = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
  float_leg_start_times_18m = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  float_leg_start_times_2y = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]
  float_leg_start_times_30m = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0]
  float_leg_start_times_3y = [3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0]
  float_leg_start_times_42m = [3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0]
  float_leg_start_times_4y = [4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
  float_leg_start_times_54m = [4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
  float_leg_start_times_5y = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

  float_leg_start_times_swaption_1 = [float_leg_start_times_1y,
                                      float_leg_start_times_18m,
                                      float_leg_start_times_2y,
                                      float_leg_start_times_30m,
                                      float_leg_start_times_3y,
                                      float_leg_start_times_42m,
                                      float_leg_start_times_4y,
                                      float_leg_start_times_54m]

  float_leg_start_times_swaption_2 = [float_leg_start_times_2y,
                                      float_leg_start_times_30m,
                                      float_leg_start_times_3y,
                                      float_leg_start_times_42m,
                                      float_leg_start_times_4y,
                                      float_leg_start_times_54m,
                                      float_leg_start_times_5y,
                                      float_leg_start_times_5y]
  float_leg_start_times = [float_leg_start_times_swaption_1,
                         float_leg_start_times_swaption_2]

  float_leg_end_times = np.clip(np.array(float_leg_start_times) + 0.5, 0.0, 5.0)

  fixed_leg_payment_times = float_leg_end_times
  float_leg_daycount_fractions = (np.array(float_leg_end_times) -
                                  np.array(float_leg_start_times))
  fixed_leg_daycount_fractions = float_leg_daycount_fractions
  fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  price = bermudan_swaption_price(
      exercise_times=exercise_times,
      floating_leg_start_times=float_leg_start_times,
      floating_leg_end_times=float_leg_end_times,
      fixed_leg_payment_times=fixed_leg_payment_times,
      floating_leg_daycount_fractions=float_leg_daycount_fractions,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.01],
      num_samples=1000000,
      time_step=0.1,
      random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
      seed=0,
      dtype=dtype)
  # Expected value: [1.8913050118443016, 1.6618681421434984] # shape = (2,)
  ````

  Args:
    exercise_times: A real `Tensor` of any shape `batch_shape + [num_exercise]`
      `and real dtype. The times corresponding to exercise dates of the
      swaptions. `num_exercise` corresponds to the number of exercise dates for
      the Bermudan swaption. The shape of this input determines the number (and
      shape) of Bermudan swaptions to be priced and the shape of the output.
    floating_leg_start_times: A real `Tensor` of the same dtype as
      `exercise_times`. The times when accrual begins for each payment in the
      floating leg upon exercise of the option. The shape of this input should
      be `exercise_times.shape + [m]` where `m` denotes the number of floating
      payments in each leg of the underlying swap until the swap maturity.
    floating_leg_end_times: A real `Tensor` of the same dtype as
      `exercise_times`. The times when accrual ends for each payment in the
      floating leg upon exercise of the option. The shape of this input should
      be `exercise_times.shape + [m]` where `m` denotes the number of floating
      payments in each leg of the underlying swap until the swap maturity.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as
      `exercise_times`. The payment times for each payment in the fixed leg.
      The shape of this input should be `exercise_times.shape + [n]` where `n`
      denotes the number of fixed payments in each leg of the underlying swap
      until the swap maturity.
    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `floating_leg_start_times`. The daycount fractions
      for each payment in the floating leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions
      for each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape
      as `fixed_leg_payment_times`. The fixed rate for each payment in the
      fixed leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape + [dim]`. Returns
      the continuously compounded zero rate at the present time for the input
      expiry time.
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
      `strikes`specifying the notional amount for the underlying swap.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the swaption is a payer (if True) or a receiver
      (if False) swaption. If not supplied, payer swaptions are assumed.
    use_finite_difference: A Python boolean specifying if the valuation should
      be performed using the finite difference and PDE.
      Default value: `False`, in which case valuation is performed using least
      squares monte-carlo method.
    lsm_basis: A Python callable specifying the basis to be used in the LSM
      algorithm. The callable must accept a `Tensor`s of shape
      `[num_samples, dim]` and output `Tensor`s of shape `[m, num_samples]`
      where `m` is the nimber of basis functions used. This input is only used
      for valuation using LSM.
      Default value: `None`, in which case a polynomial basis of order 2 is
      used.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation. This input is only used for valuation
      using LSM.
      Default value: The default value is 100.
    random_type: Enum value of `RandomType`. The type of (quasi)-random
      number generator to use to generate the simulation paths. This input is
      only used for valuation using LSM.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of
      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
      `HALTON_RANDOMIZED` the seed should be an Python integer. For
      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
      `Tensor` of shape `[2]`. This input is only used for valuation using LSM.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored. This input is only
      used for valuation using LSM.
      Default value: `0`.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation.
      This input is only used for valuation using LSM.
      Default value: `None`.
    time_step_finite_difference: Scalar real `Tensor`. Spacing between time
      grid points in finite difference discretization. This input is only
      relevant for valuation using finite difference.
      Default value: `None`, in which case a `time_step` corresponding to 100
      discrete steps is used.
    num_grid_points_finite_difference: Scalar real `Tensor`. Number of spatial
      grid points for discretization. This input is only relevant for valuation
      using finite difference.
      Default value: 100.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
      TensorFlow are used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name
      `hw_bermudan_swaption_price`.

  Returns:
    A `Tensor` of real dtype and shape  batch_shape + [dim] containing the
    computed swaption prices.

  Raises:
    (a) `ValueError` if exercise_times.rank is less than
    floating_leg_start_times.rank - 1, which would mean exercise times are not
    specified for all swaptions.
    (b) `ValueError` if `time_step` is not specified for Monte-Carlo
    simulations.
    (c) `ValueError` if `dim` > 1.
  """
  if dim > 1:
    raise ValueError('dim > 1 is currently not supported.')

  name = name or 'hw_bermudan_swaption_price'
  del floating_leg_daycount_fractions, floating_leg_start_times
  del floating_leg_end_times
  with tf.name_scope(name):
    exercise_times = tf.convert_to_tensor(
        exercise_times, dtype=dtype, name='exercise_times')
    dtype = dtype or exercise_times.dtype
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions, dtype=dtype,
        name='fixed_leg_daycount_fractions')
    fixed_leg_coupon = tf.convert_to_tensor(
        fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    if is_payer_swaption is None:
      is_payer_swaption = True
    is_payer_swaption = tf.convert_to_tensor(
        is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')

    if lsm_basis is None:
      basis_fn = lsm_v2.make_polynomial_basis(2)
    else:
      basis_fn = lsm_basis

    batch_shape = exercise_times.shape.as_list()[:-1] or [1]
    unique_exercise_times, exercise_time_index = tf.unique(
        tf.reshape(exercise_times, shape=[-1]))
    exercise_time_index = tf.reshape(
        exercise_time_index, shape=exercise_times.shape)

    # Add a dimension corresponding to multiple cashflows in a swap
    if exercise_times.shape.rank == fixed_leg_payment_times.shape.rank - 1:
      exercise_times = tf.expand_dims(exercise_times, axis=-1)
    elif exercise_times.shape.rank < fixed_leg_payment_times.shape.rank - 1:
      raise ValueError('Swaption exercise times not specified for all '
                       'swaptions in the batch. Expected rank '
                       '{} but received {}.'.format(
                           fixed_leg_payment_times.shape.rank - 1,
                           exercise_times.shape.rank))

    exercise_times = tf.repeat(
        exercise_times, tf.shape(fixed_leg_payment_times)[-1], axis=-1)

    model = vector_hull_white.VectorHullWhiteModel(
        dim,
        mean_reversion,
        volatility,
        initial_discount_rate_fn=reference_rate_fn,
        dtype=dtype)

    if use_finite_difference:
      return _bermudan_swaption_fd(batch_shape,
                                   model,
                                   exercise_times,
                                   unique_exercise_times,
                                   fixed_leg_payment_times,
                                   fixed_leg_daycount_fractions,
                                   fixed_leg_coupon,
                                   notional,
                                   is_payer_swaption,
                                   time_step_finite_difference,
                                   num_grid_points_finite_difference,
                                   name + '_fd',
                                   dtype)
    # Monte-Carlo pricing
    if time_step is None:
      raise ValueError('`time_step` must be provided for LSM valuation.')

    sim_times = unique_exercise_times
    longest_exercise_time = sim_times[-1]
    sim_times, _ = tf.unique(tf.concat([sim_times, tf.range(
        time_step, longest_exercise_time, time_step)], axis=0))
    sim_times = tf.sort(sim_times, name='sort_sim_times')

    maturities = fixed_leg_payment_times
    maturities_shape = maturities.shape
    tau = maturities - exercise_times

    curve_times_builder, _ = tf.unique(tf.reshape(tau, shape=[-1]))
    curve_times = tf.sort(curve_times_builder, name='sort_curve_times')

    # Simulate short rates and discount factors.
    p_t_tau, r_t = model.sample_discount_curve_paths(
        times=sim_times,
        curve_times=curve_times,
        num_samples=num_samples,
        random_type=random_type,
        seed=seed,
        skip=skip)

    dt = tf.concat(
        [tf.convert_to_tensor([0.0], dtype=dtype),
         sim_times[1:] - sim_times[:-1]], axis=0)
    dt = tf.expand_dims(tf.expand_dims(dt, axis=-1), axis=0)
    discount_factors_builder = tf.math.exp(-r_t * dt)
    # Transpose before (and after) because we want the cumprod along axis=1
    # and `matvec` operates on the last axis.
    discount_factors_builder = tf.transpose(
        utils.cumprod_using_matvec(
            tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])

    # make discount factors the same shape as `p_t_tau`. This involves adding
    # an extra dimenstion (corresponding to `curve_times`).
    discount_factors_builder = tf.expand_dims(
        discount_factors_builder,
        axis=1)
    # tf.repeat is needed because we will use gather_nd later on this tensor.
    discount_factors_simulated = tf.repeat(
        discount_factors_builder, tf.shape(p_t_tau)[1], axis=1)

    # `sim_times` and `curve_times` are sorted for simulation. We need to
    # select the indices corresponding to our input.
    sim_time_index = tf.searchsorted(
        sim_times, tf.reshape(exercise_times, [-1]))
    curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))

    gather_index = _prepare_indices_ijjk(
        tf.range(0, num_samples), curve_time_index, sim_time_index,
        tf.range(0, dim))

    # TODO(b/167421126): Replace `tf.gather_nd` with `tf.gather`.
    payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
    payoff_bond_price = tf.reshape(
        payoff_bond_price_builder, [num_samples] + maturities_shape + [dim])

    # Add an axis corresponding to `dim`
    fixed_leg_pv = tf.expand_dims(
        fixed_leg_coupon * fixed_leg_daycount_fractions,
        axis=-1) * payoff_bond_price
    # Sum fixed coupon payments within each swap to calculate the swap payoff
    # at each exercise time.
    fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
    float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
    payoff_swap = float_leg_pv - fixed_leg_pv
    payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)

    # Get the short rate simulations for the set of unique exercise times
    sim_time_index = tf.searchsorted(sim_times, unique_exercise_times)
    short_rate = tf.gather(r_t, sim_time_index, axis=1)

    # Currently the payoffs are computed on exercise times of each option.
    # They need to be mapped to the short rate simulation times, which is a
    # union of all exercise times.
    is_exercise_time, payoff_swap = _map_payoff_to_sim_times(
        exercise_time_index, payoff_swap, num_samples)

    # Transpose so that `time_index` is the leading dimension
    # (for XLA compatibility)
    perm = [is_exercise_time.shape.rank - 1] + list(
        range(is_exercise_time.shape.rank - 1))
    is_exercise_time = tf.transpose(is_exercise_time, perm=perm)
    payoff_swap = tf.transpose(payoff_swap, perm=perm)

    # Time to call LSM
    def _payoff_fn(rt, time_index):
      del rt
      result = tf.where(is_exercise_time[time_index] > 0,
                        tf.nn.relu(payoff_swap[time_index]), 0.0)
      return tf.reshape(result, shape=[num_samples] + batch_shape)

    discount_factors_simulated = tf.gather(
        discount_factors_simulated, sim_time_index, axis=2)

    option_value = lsm_v2.least_square_mc(
        short_rate, tf.range(0, tf.shape(short_rate)[1]),
        _payoff_fn,
        basis_fn,
        discount_factors=discount_factors_simulated[:, -1:, :, 0],
        dtype=dtype)

    return notional * option_value
