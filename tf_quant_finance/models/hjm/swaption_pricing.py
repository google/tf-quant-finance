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
"""Pricing of the Interest rate Swaption using the HJM model."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math import pde
from tf_quant_finance.models import utils
from tf_quant_finance.models import valuation_method as vm
from tf_quant_finance.models.hjm import gaussian_hjm
from tf_quant_finance.models.hjm import quasi_gaussian_hjm
from tf_quant_finance.models.hjm import swaption_util


# Points smaller than this are merged together in FD time grid
_PDE_TIME_GRID_TOL = 1e-7


def price(*,
          expiries,
          fixed_leg_payment_times,
          fixed_leg_daycount_fractions,
          fixed_leg_coupon,
          reference_rate_fn,
          num_hjm_factors,
          mean_reversion,
          volatility,
          times=None,
          time_step=None,
          num_time_steps=None,
          curve_times=None,
          corr_matrix=None,
          notional=None,
          is_payer_swaption=None,
          valuation_method=vm.ValuationMethod.MONTE_CARLO,
          num_samples=1,
          random_type=None,
          seed=None,
          skip=0,
          time_step_finite_difference=None,
          num_time_steps_finite_difference=None,
          num_grid_points_finite_difference=101,
          dtype=None,
          name=None):
  """Calculates the price of European swaptions using the HJM model.

  A European Swaption is a contract that gives the holder an option to enter a
  swap contract at a future date at a prespecified fixed rate. A swaption that
  grants the holder the right to pay fixed rate and receive floating rate is
  called a payer swaption while the swaption that grants the holder the right to
  receive fixed and pay floating payments is called the receiver swaption.
  Typically the start date (or the inception date) of the swap coincides with
  the expiry of the swaption. Mid-curve swaptions are currently not supported
  (b/160061740).

  This implementation uses the HJM model to numerically value the swaption via
  Monte-Carlo. For more information on the formulation of the HJM model, see
  quasi_gaussian_hjm.py.

  #### Example

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  # Price 1y x 1y swaption with quarterly payments using Monte Carlo
  # simulations.
  expiries = np.array([1.0])
  fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])
  fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)
  fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)
  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  mean_reversion = [0.03]
  volatility = [0.02]

  price = tff.models.hjm.swaption_price(
      expiries=expiries,
      fixed_leg_payment_times=fixed_leg_payment_times,
      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
      fixed_leg_coupon=fixed_leg_coupon,
      reference_rate_fn=zero_rate_fn,
      notional=100.,
      num_hjm_factors=1,
      mean_reversion=mean_reversion,
      volatility=volatility,
      valuation_method=tff.model.ValuationMethod.MONTE_CARLO,
      num_samples=500000,
      time_step=0.1,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
      seed=[1, 2])
  # Expected value: [[0.716]]
  ````


  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007. Section 6.7, page 237.

  Args:
    expiries: A real `Tensor` of any shape and dtype. The time to expiration of
      the swaptions. The shape of this input determines the number (and shape)
      of swaptions to be priced and the shape of the output.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg. The `fixed_leg_payment_times` should be
      greater-than or equal-to the corresponding expiries.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions for
      each payment in the fixed leg.
    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape as
      `fixed_leg_payment_times`. The fixed rate for each payment in the fixed
      leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape +
      [num_hjm_factors]`. Returns the continuously compounded zero rate at the
      present time for the input expiry time.
    num_hjm_factors: A Python scalar which corresponds to the number of factors
      in the HJM model to be used for pricing.
    mean_reversion: A real positive `Tensor` of shape `[num_hjm_factors]`.
      Corresponds to the mean reversion rate of each factor.
    volatility: A real positive `Tensor` of the same `dtype` and shape as
      `mean_reversion` or a callable with the following properties: (a)  The
        callable should accept a scalar `Tensor` `t` and a 1-D `Tensor` `r(t)`
        of shape `[num_samples]` and returns a 2-D `Tensor` of shape
        `[num_samples, num_hjm_factors]`. The variable `t`  stands for time and
        `r(t)` is the short rate at time `t`.  The function returns the
        instantaneous volatility `sigma(t) = sigma(t, r(t))`. When `volatility`
        is specified as a real `Tensor`, each factor is assumed to have a
        constant instantaneous volatility  and the  model is effectively a
        Gaussian HJM model. Corresponds to the instantaneous volatility of each
        factor.
    times: An optional rank 1 `Tensor` of increasing positive real values. The
      times at which Monte Carlo simulations are performed. Relevant when
      swaption valuation is done using Monte Calro simulations.
      Default value: `None` in which case simulation times are computed based
      on either `time_step` or `num_time_steps` inputs.
    time_step: Optional scalar real `Tensor`. Maximal distance between time
      grid points in Euler scheme. Relevant when Euler scheme is used for
      simulation. This input or `num_time_steps` are required when valuation
      method is Monte Carlo.
      Default Value: `None`.
    num_time_steps: An optional scalar integer `Tensor` - a total number of
      time steps during Monte Carlo simulations. The maximal distance betwen
      points in grid is bounded by
      `times[-1] / (num_time_steps - times.shape[0])`.
      Either this or `time_step` should be supplied when the valuation method
      is Monte Carlo.
      Default value: `None`.
    curve_times: An optional rank 1 `Tensor` of positive and increasing real
      values. The maturities at which spot discount curve is computed during
      simulations.
      Default value: `None` in which case `curve_times` is computed based on
      swaption expiries and `fixed_leg_payments_times` inputs.
    corr_matrix: A `Tensor` of shape `[num_hjm_factors, num_hjm_factors]` and
      the same `dtype` as `mean_reversion`. Specifies the correlation between
      HJM factors.
      Default value: `None` in which case the factors are assumed to be
        uncorrelated.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the underlying swaps.
       Default value: None in which case the notional is set to 1.
    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.
      Indicates whether the swaption is a payer (if True) or a receiver (if
      False) swaption. If not supplied, payer swaptions are assumed.
    valuation_method: An enum of type `ValuationMethod` specifying
      the method to be used for swaption valuation. Currently the valuation is
      supported using `MONTE_CARLO` and `FINITE_DIFFERENCE` methods. Valuation
      using finite difference is only supported for Gaussian HJM models, i.e.
      for models with constant mean-reversion rate and time-dependent
      volatility.
      Default value: `ValuationMethod.MONTE_CARLO`, in which case
      swaption valuation is done using Monte Carlo simulations.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation. This input is ignored during analytic
      valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random number
      generator to use to generate the simulation paths. This input is relevant
      only for Monte-Carlo valuation and ignored during analytic valuation.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,
      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,
      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be an Python
      integer. For `STATELESS` and  `STATELESS_ANTITHETIC` must be supplied as
      an integer `Tensor` of shape `[2]`. This input is relevant only for
      Monte-Carlo valuation and ignored during analytic valuation.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    time_step_finite_difference: Optional scalar real `Tensor`. Spacing between
      time grid points in finite difference discretization. This input is only
      relevant for valuation using finite difference.
      Default value: `None`. If `num_time_steps_finite_difference` is also
      unspecified then a `time_step` corresponding to 100 discretization steps
      is used.
    num_time_steps_finite_difference: Optional scalar real `Tensor`. Number of
      time grid points in finite difference discretization. This input is only
      relevant for valuation using finite difference.
      Default value: `None`. If `time_step_finite_difference` is also
      unspecified, then 100 time steps are used.
    num_grid_points_finite_difference: Optional scalar real `Tensor`. Number of
      spatial grid points per dimension. Currently, we construct an uniform grid
      for spatial discretization. This input is only relevant for valuation
      using finite difference.
      Default value: 101.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this function.
      Default value: `None` which maps to the default name `hjm_swaption_price`.

  Returns:
    A `Tensor` of real dtype and shape expiries.shape + [1]
    containing the computed swaption prices. For swaptions that have reset in
    the past (expiries<0), the function sets the corresponding option prices to
    0.0.
  """

  # TODO(b/160061740): Extend the functionality to support mid-curve swaptions.
  name = name or 'hjm_swaption_price'
  with tf.name_scope(name):
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    dtype = dtype or expiries.dtype
    fixed_leg_payment_times = tf.convert_to_tensor(
        fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
    fixed_leg_daycount_fractions = tf.convert_to_tensor(
        fixed_leg_daycount_fractions,
        dtype=dtype,
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

    output_shape = expiries.shape.as_list() + [1]
    # Add a dimension corresponding to multiple cashflows in a swap
    if expiries.shape.rank == fixed_leg_payment_times.shape.rank - 1:
      expiries = tf.expand_dims(expiries, axis=-1)
    elif expiries.shape.rank < fixed_leg_payment_times.shape.rank - 1:
      raise ValueError('Swaption expiries not specified for all swaptions '
                       'in the batch. Expected rank {} but received {}.'.format(
                           fixed_leg_payment_times.shape.rank - 1,
                           expiries.shape.rank))

    # Expected shape: batch_shape + [m], where m is the number of fixed leg
    # payments per underlying swap. This is the same as
    # fixed_leg_payment_times.shape
    #
    # We need to explicitly use tf.repeat because we need to price
    # batch_shape + [m] bond options with different strikes along the last
    # dimension.
    expiries = tf.repeat(
        expiries, tf.shape(fixed_leg_payment_times)[-1], axis=-1)

    if valuation_method == vm.ValuationMethod.FINITE_DIFFERENCE:
      model = gaussian_hjm.GaussianHJM(
          num_hjm_factors,
          mean_reversion=mean_reversion,
          volatility=volatility,
          initial_discount_rate_fn=reference_rate_fn,
          corr_matrix=corr_matrix,
          dtype=dtype)

      batch_shape = expiries.shape.as_list()[:-1] or [1]
      return _bermudan_swaption_fd(
          batch_shape,
          model,
          # Add a dimension to denote ONE exercise date
          tf.expand_dims(expiries, axis=-2),
          fixed_leg_payment_times,
          fixed_leg_daycount_fractions,
          fixed_leg_coupon,
          notional,
          is_payer_swaption,
          time_step_finite_difference,
          num_time_steps_finite_difference,
          num_grid_points_finite_difference,
          name + '_fd',
          dtype)
    elif valuation_method == vm.ValuationMethod.MONTE_CARLO:
      # Monte-Carlo pricing
      model = quasi_gaussian_hjm.QuasiGaussianHJM(
          num_hjm_factors,
          mean_reversion=mean_reversion,
          volatility=volatility,
          initial_discount_rate_fn=reference_rate_fn,
          corr_matrix=corr_matrix,
          dtype=dtype)

      return _european_swaption_mc(
          output_shape, model, expiries, fixed_leg_payment_times,
          fixed_leg_daycount_fractions, fixed_leg_coupon, notional,
          is_payer_swaption, times, time_step, num_time_steps, curve_times,
          num_samples, random_type, skip, seed, dtype, name + '_mc')
    else:
      raise ValueError('Swaption Valuation using {} is not supported'.format(
          str(valuation_method)))


def _european_swaption_mc(output_shape, model, expiries,
                          fixed_leg_payment_times, fixed_leg_daycount_fractions,
                          fixed_leg_coupon, notional, is_payer_swaption, times,
                          time_step, num_time_steps, curve_times, num_samples,
                          random_type, skip, seed, dtype, name):
  """Price European swaptions using Monte-Carlo."""
  with tf.name_scope(name):
    if (times is None) and (time_step is None) and (num_time_steps is None):
      raise ValueError(
          'One of `times`, `time_step` or `num_time_steps` must be '
          'provided for simulation based swaption valuation.')

    def _sample_discount_curve_path_fn(times, curve_times, num_samples):
      p_t_tau, r_t, df = model.sample_discount_curve_paths(
          times=times,
          curve_times=curve_times,
          num_samples=num_samples,
          random_type=random_type,
          time_step=time_step,
          num_time_steps=num_time_steps,
          seed=seed,
          skip=skip)
      p_t_tau = tf.expand_dims(p_t_tau, axis=-1)
      r_t = tf.expand_dims(r_t, axis=-1)
      df = tf.expand_dims(df, axis=-1)
      return p_t_tau, r_t, df

    payoff_discount_factors, payoff_bond_price = (
        swaption_util.discount_factors_and_bond_prices_from_samples(
            expiries=expiries,
            payment_times=fixed_leg_payment_times,
            sample_discount_curve_paths_fn=_sample_discount_curve_path_fn,
            num_samples=num_samples,
            times=times,
            curve_times=curve_times,
            dtype=dtype))

    # Add an axis corresponding to `dim`
    fixed_leg_pv = tf.expand_dims(
        fixed_leg_coupon * fixed_leg_daycount_fractions,
        axis=-1) * payoff_bond_price

    # Sum fixed coupon payments within each swap.
    # Here, axis=-2 is the payments axis - i.e. summing over all payments; and
    # the last axis is the `dim` axis, as explained in comment above
    # `fixed_leg_pv` (Note that for HJM the dim of this axis is 1 always).
    fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
    float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
    payoff_swap = payoff_discount_factors[..., -1, :] * (
        float_leg_pv - fixed_leg_pv)
    payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
    payoff_swaption = tf.math.maximum(payoff_swap, 0.0)
    option_value = tf.reshape(
        tf.math.reduce_mean(payoff_swaption, axis=0), output_shape)

    return notional * option_value


def _bermudan_swaption_fd(batch_shape, model, exercise_times,
                          fixed_leg_payment_times, fixed_leg_daycount_fractions,
                          fixed_leg_coupon, notional, is_payer_swaption,
                          time_step_fd, num_time_steps_fd, num_grid_points_fd,
                          name, dtype):
  """Price Bermudan swaptions using finite difference."""
  with tf.name_scope(name):
    dim = model.dim()
    x_min = -0.5
    x_max = 0.5
    # grid.shape = (num_grid_points,2)
    grid = pde.grids.uniform_grid(
        minimums=[x_min] * dim,
        maximums=[x_max] * dim,
        sizes=[num_grid_points_fd] * dim,
        dtype=dtype)

    # TODO(b/186876306): Remove dynamic shapes.
    pde_time_grid, pde_time_grid_dt = _create_pde_time_grid(
        exercise_times, time_step_fd, num_time_steps_fd, dtype)
    maturities, unique_maturities, maturities_shape = (
        _create_term_structure_maturities(fixed_leg_payment_times))

    num_maturities = tf.shape(unique_maturities)[-1]
    x_meshgrid = _coord_grid_to_mesh_grid(grid)
    meshgrid_shape = tf.shape(x_meshgrid)
    broadcasted_maturities = tf.expand_dims(unique_maturities, axis=0)

    num_grid_points = tf.math.reduce_prod(meshgrid_shape[1:])
    shape_to_broadcast = tf.concat([meshgrid_shape, [num_maturities]], axis=0)

    # Reshape `state_x`, `maturities` to (num_grid_points, num_maturities)
    state_x = tf.expand_dims(x_meshgrid, axis=-1)
    state_x = tf.broadcast_to(state_x, shape_to_broadcast)
    broadcasted_maturities = tf.broadcast_to(broadcasted_maturities,
                                             shape_to_broadcast[1:])

    def _get_swap_payoff(payoff_time):
      broadcasted_exercise_times = tf.broadcast_to(payoff_time,
                                                   shape_to_broadcast[1:])

      # Zero-coupon bond curve
      zcb_curve = model.discount_bond_price(
          tf.transpose(
              tf.reshape(state_x, [dim, num_grid_points * num_maturities])),
          tf.reshape(broadcasted_exercise_times, [-1]),
          tf.reshape(broadcasted_maturities, [-1]))
      zcb_curve = tf.reshape(zcb_curve, [num_grid_points, num_maturities])

      maturities_index = tf.searchsorted(unique_maturities,
                                         tf.reshape(maturities, [-1]))

      zcb_curve = tf.gather(zcb_curve, maturities_index, axis=-1)
      # zcb_curve.shape = [num_grid_points] + [maturities_shape]
      zcb_curve = tf.reshape(
          zcb_curve, tf.concat([[num_grid_points], maturities_shape], axis=0))

      # Shape after reduce_sum =
      # (num_grid_points, batch_shape)
      fixed_leg = tf.math.reduce_sum(
          fixed_leg_coupon * fixed_leg_daycount_fractions * zcb_curve, axis=-1)
      float_leg = 1.0 - zcb_curve[..., -1]
      payoff_swap = float_leg - fixed_leg
      payoff_swap = tf.where(is_payer_swaption, payoff_swap, -payoff_swap)
      return tf.reshape(
          tf.transpose(payoff_swap),
          tf.concat([batch_shape, meshgrid_shape[1:]], axis=0))

    def _get_index(t, tensor_to_search):
      t = tf.expand_dims(t, axis=-1)
      index = tf.searchsorted(tensor_to_search, t - _PDE_TIME_GRID_TOL, 'right')
      y = tf.gather(tensor_to_search, index)
      return tf.where(tf.math.abs(t - y) < _PDE_TIME_GRID_TOL, index, -1)[0]

    sum_x_meshgrid = tf.math.reduce_sum(x_meshgrid, axis=0)

    def _is_exercise_time(t):
      return tf.reduce_any(
          tf.where(
              tf.math.abs(exercise_times[..., -1] - t) < _PDE_TIME_GRID_TOL,
              True, False),
          axis=-1)

    def _discounting_fn(t, grid):
      del grid
      f_0_t = (model._instant_forward_rate_fn(t))  # pylint: disable=protected-access
      return sum_x_meshgrid + f_0_t

    def _final_value():
      t = pde_time_grid[-1]
      payoff_swap = tf.nn.relu(_get_swap_payoff(t))
      is_ex_time = _is_exercise_time(t)
      return tf.where(
          tf.reshape(is_ex_time, tf.concat([batch_shape, [1] * dim], axis=0)),
          payoff_swap, 0.0)

    def _values_transform_fn(t, grid, value_grid):
      zero = tf.zeros_like(value_grid)
      is_ex_time = _is_exercise_time(t)

      def _at_least_one_swaption_pays():
        payoff_swap = tf.nn.relu(_get_swap_payoff(t))
        return tf.where(
            tf.reshape(is_ex_time, tf.concat([batch_shape, [1] * dim], axis=0)),
            payoff_swap, zero)

      v_star = tf.cond(
          tf.math.reduce_any(is_ex_time), _at_least_one_swaption_pays,
          lambda: zero)
      return grid, tf.maximum(value_grid, v_star)

    def _pde_time_step(t):
      index = _get_index(t, pde_time_grid)
      return pde_time_grid_dt[index]

    # Use default boundary conditions, d^2V/dx_i^2 = 0
    boundary_conditions = [(None, None) for i in range(dim)]
    # res[0] contains the swaption prices.
    # res[0].shape = batch_shape + [num_grid_points] * dim
    res = model.fd_solver_backward(
        pde_time_grid[-1],
        0.0,
        grid,
        values_grid=_final_value(),
        time_step=_pde_time_step,
        boundary_conditions=boundary_conditions,
        values_transform_fn=_values_transform_fn,
        discounting=_discounting_fn,
        dtype=dtype)

    idx = tf.searchsorted(
        tf.convert_to_tensor(grid),
        tf.expand_dims(tf.convert_to_tensor([0.0] * dim, dtype=dtype), axis=-1))
    # idx.shape = (dim, 1)
    idx = tf.squeeze(idx) if dim > 1 else tf.reshape(idx, shape=[1])
    slices = [slice(None)] + [slice(i, i + 1) for i in tf.unstack(idx)]
    # shape = batch_shape + [1] * dim
    option_value = res[0][slices]
    # shape = batch_shape
    option_value = tf.squeeze(option_value, axis=list(range(-dim, 0)))

    # output_shape = batch_shape + [1]
    return notional * tf.expand_dims(
        tf.reshape(option_value, batch_shape), axis=-1)


def _coord_grid_to_mesh_grid(coord_grid):
  if len(coord_grid) == 1:
    return tf.expand_dims(coord_grid[0], 0)
  x_meshgrid = tf.stack(values=tf.meshgrid(*coord_grid, indexing='ij'), axis=-1)
  perm = [len(coord_grid)] + list(range(len(coord_grid)))
  return tf.transpose(x_meshgrid, perm=perm)


def _create_pde_time_grid(exercise_times, time_step_fd, num_time_steps_fd,
                          dtype):
  """Create PDE time grid."""
  with tf.name_scope('create_pde_time_grid'):
    exercise_times, _ = tf.unique(tf.reshape(exercise_times, shape=[-1]))
    if num_time_steps_fd is not None:
      num_time_steps_fd = tf.convert_to_tensor(
          num_time_steps_fd, dtype=tf.int32, name='num_time_steps_fd')
      time_step_fd = tf.math.reduce_max(exercise_times) / tf.cast(
          num_time_steps_fd, dtype=dtype)
    if time_step_fd is None and num_time_steps_fd is None:
      num_time_steps_fd = 100

    pde_time_grid, _, _ = utils.prepare_grid(
        times=exercise_times,
        time_step=time_step_fd,
        dtype=dtype,
        num_time_steps=num_time_steps_fd)
    pde_time_grid_dt = pde_time_grid[1:] - pde_time_grid[:-1]
    pde_time_grid_dt = tf.concat([[100.0], pde_time_grid_dt], axis=-1)

    return pde_time_grid, pde_time_grid_dt


def _create_term_structure_maturities(fixed_leg_payment_times):
  """Create maturities needed for termstructure simulations."""

  with tf.name_scope('create_termstructure_maturities'):
    maturities = fixed_leg_payment_times
    maturities_shape = tf.shape(maturities)

    # We should eventually remove tf.unique, but keeping it for now because
    # PDE solvers are not xla compatible in TFF currently.
    unique_maturities, _ = tf.unique(tf.reshape(maturities, shape=[-1]))
    unique_maturities = tf.sort(unique_maturities, name='sort_maturities')

    return maturities, unique_maturities, maturities_shape
