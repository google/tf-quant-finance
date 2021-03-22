# Lint as: python3
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
"""Multi-Factor Gaussian HJM Model."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math import gradient
from tf_quant_finance.math import piecewise
from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import utils
from tf_quant_finance.models.hjm import quasi_gaussian_hjm


class GaussianHJM(quasi_gaussian_hjm.QuasiGaussianHJM):
  r"""Gaussian HJM model for term-structure modeling.

  Heath-Jarrow-Morton (HJM) model for the interest rate term-structre
  modelling specifies the dynamics of the instantaneus forward rate `f(t,T)`
  with maturity `T` at time `t` as follows:

  ```None
    df(t,T) = mu(t,T) dt + sum_i sigma_i(t,  T) * dW_i(t),
    1 <= i <= n,
  ```
  where `mu(t,T)` and `sigma_i(t,T)` denote the drift and volatility
  for the forward rate and `W_i` are Brownian motions with instantaneous
  correlation `Rho`. The model above represents an `n-factor` HJM model.
  The Gaussian HJM model assumes that the volatility `sigma_i(t,T)` is a
  deterministic function of time (t). Under the risk-neutral measure, the
  drift `mu(t,T)` is computed as

  ```
    mu(t,T) = sum_i sigma_i(t,T)  int_t^T sigma_(t,u) du
  ```
  Using the separability condition, the HJM model above can be formulated as
  the following Markovian model:

  ```None
    sigma(t,T) = sigma(t) * h(T)    (Separability condition)
  ```
  A common choice for the function h(t) is `h(t) = exp(-kt)`. Using the above
  parameterization of sigma(t,T), we obtain the following Markovian
  formulation of the HJM model [1]:

  ```None
    HJM Model
    dx_i(t) = (sum_j [y_ij(t)] - k_i * x_i(t)) dt + sigma_i(t) dW_i
    dy_ij(t) = (rho_ij * sigma_i(t)*sigma_j(t) - (k_i + k_j) * y_ij(t)) dt
    r(t) = sum_i x_i(t) + f(0, t)
  ```
  where `x` is an `n`-dimensional vector and `y` is an `nxn` dimensional
  matrix. For Gaussian HJM model, the quantity `y_ij(t)` can be computed
  analytically as follows:

  ```None
    y_ij(t) = rho_ij * exp(-k_i * t) * exp(-k_j * t) *
              int_0^t exp((k_i+k_j) * s) * sigma_i(s) * sigma_j(s) ds
  ```

  The Gaussian HJM class implements the model outlined above by simulating the
  state `x(t)` while analytically computing `y(t)`.

  The price at time `t` of a zero-coupon bond maturing at `T` is given by
  (Ref. [1]):

  ```None
  P(t,T) = P(0,T) / P(0,t) *
           exp(-x(t) * G(t,T) - 0.5 * y(t) * G(t,T)^2)
  ```

  The HJM model implementation supports constant mean-reversion rate `k` and
  `sigma(t)` can be an arbitrary function of `t`. We use Euler discretization
  to simulate the HJM model.

  #### Example. Simulate a 4-factor HJM process.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  def discount_fn(x):
    return 0.01 * tf.ones_like(x, dtype=dtype)

  process = tff.experimental.hjm.GaussianHJM(
      dim=4,
      mean_reversion=[0.03, 0.01, 0.02, 0.005],  # constant mean-reversion
      volatility=[0.01, 0.011, 0.015, 0.008],  # constant volatility
      initial_discount_rate_fn=discount_fn,
      dtype=dtype)
  times = np.array([0.1, 1.0, 2.0, 3.0])
  short_rate_paths, discount_paths, _, _ = process.sample_paths(
      times,
      num_samples=100000,
      time_step=0.1,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
      seed=[1, 2],
      skip=1000000)
  ```

  #### References:
    [1]: Leif B. G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling.
    Volume II: Term Structure Models.
  """

  def __init__(self,
               dim,
               mean_reversion,
               volatility,
               initial_discount_rate_fn,
               corr_matrix=None,
               dtype=None,
               name=None):
    """Initializes the HJM model.

    Args:
      dim: A Python scalar which corresponds to the number of factors comprising
        the model.
      mean_reversion: A real positive `Tensor` of shape `[dim]`. Corresponds to
        the mean reversion rate of each factor.
      volatility: A real positive `Tensor` of the same `dtype` and shape as
        `mean_reversion` or a callable with the following properties: (a)  The
          callable should accept a scalar `Tensor` `t` and returns a 1-D
          `Tensor` of shape `[dim]`. The function returns instantaneous
          volatility `sigma(t)`. When `volatility` is specified is a real
          `Tensor`, each factor is assumed to have a constant instantaneous
          volatility. Corresponds to the instantaneous volatility of each
          factor.
      initial_discount_rate_fn: A Python callable that accepts expiry time as a
        real `Tensor` of the same `dtype` as `mean_reversion` and returns a
        `Tensor` of shape `input_shape + dim`. Corresponds to the zero coupon
        bond yield at the present time for the input expiry time.
      corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as
        `mean_reversion`. Corresponds to the correlation matrix `Rho`.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: `None` which means that default dtypes inferred by
          TensorFlow are used.
      name: Python string. The name to give to the ops created by this class.
        Default value: `None` which maps to the default name
          `gaussian_hjm_model`.
    """
    self._name = name or 'gaussian_hjm_model'
    with tf.name_scope(self._name):
      self._dtype = dtype or None
      self._dim = dim
      self._factors = dim

      def _instant_forward_rate_fn(t):
        t = tf.convert_to_tensor(t, dtype=self._dtype)

        def _log_zero_coupon_bond(x):
          r = tf.convert_to_tensor(
              initial_discount_rate_fn(x), dtype=self._dtype)
          return -r * x

        rate = -gradient.fwd_gradient(
            _log_zero_coupon_bond,
            t,
            use_gradient_tape=True,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return rate

      def _initial_discount_rate_fn(t):
        return tf.convert_to_tensor(
            initial_discount_rate_fn(t), dtype=self._dtype)

      self._instant_forward_rate_fn = _instant_forward_rate_fn
      self._initial_discount_rate_fn = _initial_discount_rate_fn
      self._mean_reversion = tf.convert_to_tensor(
          mean_reversion, dtype=dtype, name='mean_reversion')

      # Setup volatility
      if callable(volatility):
        self._volatility = volatility
      else:
        volatility = tf.convert_to_tensor(volatility, dtype=dtype)
        jump_locations = [[]] * dim
        volatility = tf.expand_dims(volatility, axis=-1)
        self._volatility = piecewise.PiecewiseConstantFunc(
            jump_locations=jump_locations, values=volatility, dtype=dtype)

      if corr_matrix is None:
        corr_matrix = tf.eye(dim, dim, dtype=self._dtype)
      self._rho = tf.convert_to_tensor(corr_matrix, dtype=dtype, name='rho')
      self._sqrt_rho = tf.linalg.cholesky(self._rho)

      # Volatility function
      def _vol_fn(t, state):
        """Volatility function of Gaussian-HJM."""
        del state
        volatility = self._volatility(tf.expand_dims(t, -1))  # shape=(dim, 1)

        return self._sqrt_rho * volatility

      # Drift function
      def _drift_fn(t, state):
        """Drift function of Gaussian-HJM."""
        x = state
        # shape = [self._factors, self._factors]
        y = self.state_y(tf.expand_dims(t, axis=-1))[..., 0]
        drift = tf.math.reduce_sum(y, axis=-1) - self._mean_reversion * x
        return drift

      self._exact_discretization_setup(dim)
      super(quasi_gaussian_hjm.QuasiGaussianHJM,
            self).__init__(dim, _drift_fn, _vol_fn, dtype, name)

  def sample_paths(self,
                   times,
                   num_samples,
                   time_step,
                   random_type=None,
                   seed=None,
                   skip=0,
                   name=None):
    """Returns a sample of short rate paths from the HJM process.

    Uses Euler sampling for simulating the short rate paths.

    Args:
      times: A real positive `Tensor` of shape `(num_times,)`. The times at
        which the path points are to be evaluated.
      num_samples: Positive scalar `int32` `Tensor`. The number of paths to
        draw.
      time_step: Scalar real `Tensor`. Maximal distance between time grid points
        in Euler scheme. Used only when Euler scheme is applied.
        Default value: `None`.
      random_type: Enum value of `RandomType`. The type of (quasi)-random
        number generator to use to generate the paths.
        Default value: `None` which maps to the standard pseudo-random numbers.
      seed: Seed for the random number generator. The seed is
        only relevant if `random_type` is one of
        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
        `HALTON_RANDOMIZED` the seed should be an Python integer. For
        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
        `Tensor` of shape `[2]`.
        Default value: `None` which means no seed is set.
      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
        Halton sequence to skip. Used only when `random_type` is 'SOBOL',
        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
        Default value: `0`.
      name: Python string. The name to give this op.
        Default value: `sample_paths`.

    Returns:
      A tuple containing four elements.

      * The first element is a `Tensor` of
      shape `[num_samples, num_times]` containing the simulated short rate
      paths.
      * The second element is a `Tensor` of shape
      `[num_samples, num_times]` containing the simulated discount factor
      paths.
      * The third element is a `Tensor` of shape
      `[num_samples, num_times, dim]` conating the simulated values of the
      state variable `x`
      * The fourth element is a `Tensor` of shape
      `[num_samples, num_times, dim^2]` conating the simulated values of the
      state variable `y`.

    Raises:
      ValueError:
        (a) If `times` has rank different from `1`.
        (b) If Euler scheme is used by times is not supplied.
    """
    name = name or self._name + '_sample_path'
    with tf.name_scope(name):
      times = tf.convert_to_tensor(times, self._dtype)
      if times.shape.rank != 1:
        raise ValueError('`times` should be a rank 1 Tensor. '
                         'Rank is {} instead.'.format(times.shape.rank))
      return self._sample_paths(times, time_step, num_samples, random_type,
                                skip, seed)

  def state_y(self, t):
    """Computes the state variable `y(t)` for tha Gaussian HJM Model.

    For Gaussian HJM model, the state parameter y(t), can be analytically
    computed as follows:

    y_ij(t) = exp(-k_i * t) * exp(-k_j * t) * (
              int_0^t rho_ij * sigma_i(u) * sigma_j(u) * du)

    Args:
      t: A rank 1 real `Tensor` of shape `[num_times]` specifying the time `t`.

    Returns:
      A real `Tensor` of shape [self._factors, self._factors, num_times]
      containing the computed y_ij(t).
    """
    t = tf.convert_to_tensor(t, dtype=self._dtype)
    t_shape = tf.shape(t)
    t = tf.broadcast_to(t, tf.concat([[self._dim], t_shape], axis=0))
    time_index = tf.searchsorted(self._jump_locations, t)
    # create a matrix k2(i,j) = k(i) + k(j)
    mr2 = tf.expand_dims(self._mean_reversion, axis=-1)
    # Add a dimension corresponding to `num_times`
    mr2 = tf.expand_dims(mr2 + tf.transpose(mr2), axis=-1)

    def _integrate_volatility_squared(vol, l_limit, u_limit):
      # create sigma2_ij = sigma_i * sigma_j
      vol = tf.expand_dims(vol, axis=-2)
      vol_squared = tf.expand_dims(self._rho, axis=-1) * (
          vol * tf.transpose(vol, perm=[1, 0, 2]))
      return vol_squared / mr2 * (tf.math.exp(mr2 * u_limit) - tf.math.exp(
          mr2 * l_limit))

    is_constant_vol = tf.math.equal(tf.shape(self._jump_values_vol)[-1], 0)
    v_squared_between_vol_knots = tf.cond(
        is_constant_vol,
        lambda: tf.zeros(shape=(self._dim, self._dim, 0), dtype=self._dtype),
        lambda: _integrate_volatility_squared(  # pylint: disable=g-long-lambda
            self._jump_values_vol, self._padded_knots, self._jump_locations))
    v_squared_at_vol_knots = tf.concat([
        tf.zeros((self._dim, self._dim, 1), dtype=self._dtype),
        utils.cumsum_using_matvec(v_squared_between_vol_knots)
    ], axis=-1)

    vn = tf.concat([self._zero_padding, self._jump_locations], axis=1)

    v_squared_t = _integrate_volatility_squared(
        self._volatility(t), tf.gather(vn, time_index, batch_dims=1), t)
    v_squared_t += tf.gather(v_squared_at_vol_knots, time_index, batch_dims=-1)

    return tf.math.exp(-mr2 * t) * v_squared_t

  def discount_bond_price(self, state, times, maturities, name=None):
    """Returns zero-coupon bond prices `P(t,T)` conditional on `x(t)`.

    Args:
      state: A `Tensor` of real dtype and shape compatible with
        `(num_times, dim)` specifying the state `x(t)`.
      times: A `Tensor` of real dtype and shape `(num_times,)`. The time `t`
        at which discount bond prices are computed.
      maturities: A `Tensor` of real dtype and shape `(num_times,)`. The time
        to maturity of the discount bonds.
      name: Str. The name to give this op.
        Default value: `discount_bond_prices`.

    Returns:
      A `Tensor` of real dtype and the same shape as `(num_times,)`
      containing the price of zero-coupon bonds.
    """
    name = name or self._name + '_discount_bond_prices'
    with tf.name_scope(name):
      x_t = tf.convert_to_tensor(state, self._dtype)
      times = tf.convert_to_tensor(times, self._dtype)
      maturities = tf.convert_to_tensor(maturities, self._dtype)
      # Flatten it because `PiecewiseConstantFunction` expects the first
      # dimension to be broadcastable to [dim]
      input_shape_times = tf.shape(times)
      # The shape of `mean_reversion` will be (dim,n) where `n` is the number
      # of elements in `times`.
      mean_reversion = self._mean_reversion
      y_t = self.state_y(times)

      times = tf.expand_dims(times, axis=-1)
      maturities = tf.expand_dims(maturities, axis=-1)
      y_t = tf.reshape(tf.transpose(y_t), tf.concat(
          [input_shape_times, [self._dim, self._dim]], axis=0))

      # note that by making `times` and `maturities` of the same shape, we
      # ensure that the shape of the output is `(1, 1, num_times)` instead of
      # `(1, num_maturities, num_times)`
      values = self._bond_reconstitution(  # shape=(1, 1, num_times)
          times, maturities, mean_reversion, x_t, y_t, 1,
          tf.shape(times)[0])
      return values[0][0]

  def _sample_paths(self, times, time_step, num_samples, random_type, skip,
                    seed):
    """Returns a sample of paths from the process."""
    initial_state = tf.zeros((self._dim,), dtype=self._dtype)
    # Note that we need a finer simulation grid (determnied by `dt`) to compute
    # discount factors accurately. The `times` input might not be granular
    # enough for accurate calculations.
    times, _, time_indices = utils.prepare_grid(
        times=times, time_step=time_step, dtype=self._dtype)
    # Add zeros as a starting location
    dt = times[1:] - times[:-1]

    # Shape = (num_samples, num_times, nfactors)
    paths = euler_sampling.sample(
        self._dim,
        self._drift_fn,
        self._volatility_fn,
        times,
        num_samples=num_samples,
        initial_state=initial_state,
        random_type=random_type,
        seed=seed,
        time_step=time_step,
        skip=skip)
    y_paths = self.state_y(times)  # shape=(dim, dim, num_times)
    y_paths = tf.reshape(
        y_paths, tf.concat([[self._dim**2], tf.shape(times)], axis=0))

    # shape=(num_samples, num_times, dim**2)
    y_paths = tf.repeat(tf.expand_dims(tf.transpose(
        y_paths), axis=0), num_samples, axis=0)

    f_0_t = self._instant_forward_rate_fn(times)  # shape=(num_times,)
    rate_paths = tf.math.reduce_sum(
        paths, axis=-1) + f_0_t  # shape=(num_samples, num_times)

    discount_factor_paths = tf.math.exp(-rate_paths[:, :-1] * dt)
    discount_factor_paths = tf.concat(
        [tf.ones((num_samples, 1), dtype=self._dtype), discount_factor_paths],
        axis=1)  # shape=(num_samples, num_times)
    discount_factor_paths = utils.cumprod_using_matvec(discount_factor_paths)
    return (tf.gather(rate_paths, time_indices, axis=1),
            tf.gather(discount_factor_paths, time_indices, axis=1),
            tf.gather(paths, time_indices, axis=1),
            tf.gather(y_paths, time_indices, axis=1))

  def _exact_discretization_setup(self, dim):
    """Initial setup for efficient computations."""
    self._zero_padding = tf.zeros((dim, 1), dtype=self._dtype)
    self._jump_locations = self._volatility.jump_locations()
    self._jump_values_vol = self._volatility(self._jump_locations)
    self._padded_knots = tf.concat(
        [self._zero_padding, self._jump_locations[:, :-1]], axis=1)
