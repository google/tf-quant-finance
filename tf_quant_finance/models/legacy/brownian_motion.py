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
"""N-dimensional Brownian Motion.

Implements the Ito process defined by:

```
  dX_i = a_i(t) dt + Sum[S_{ij}(t) dW_{j}, 1 <= j <= n] for each i in {1,..,n}
```

where `dW_{j}, 1 <= j <= n` are n independent 1D Brownian increments. The
coefficient `a_i` is the drift and the matrix `S_{ij}` is the volatility of the
process.

For more details, see Ref [1].

#### References:
  [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.
"""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.random_ops import multivariate_normal as mvn
from tf_quant_finance.models.legacy import brownian_motion_utils as bmu
from tf_quant_finance.models.legacy import ito_process


class BrownianMotion(ito_process.ItoProcess):
  """The multi dimensional Brownian Motion."""

  def __init__(self,
               dim=1,
               drift=None,
               volatility=None,
               total_drift_fn=None,
               total_covariance_fn=None,
               dtype=None,
               name=None):
    """Initializes the Brownian motion class.

    Represents the Ito process:

    ```None
      dX_i = a_i(t) dt + Sum(S_{ij}(t) dW_j for j in [1 ... n]), 1 <= i <= n

    ```

    `a_i(t)` is the drift rate of this process and the `S_{ij}(t)` is the
    volatility matrix. Associated to these parameters are the integrated
    drift and covariance functions. These are defined as:

    ```None
      total_drift_{i}(t1, t2) = Integrate(a_{i}(t), t1 <= t <= t2)
      total_covariance_{ij}(t1, t2) = Integrate(inst_covariance_{ij}(t),
                                                     t1 <= t <= t2)
      inst_covariance_{ij}(t) = (S.S^T)_{ij}(t)
    ```

    Sampling from the Brownian motion process with time dependent parameters
    can be done efficiently if the total drift and total covariance functions
    are supplied. If the parameters are constant, the total parameters can be
    automatically inferred and it is not worth supplying then explicitly.

    Currently, it is not possible to infer the total drift and covariance from
    the instantaneous values if the latter are functions of time. In this case,
    we use a generic sampling method (Euler-Maruyama) which may be
    inefficient. It is advisable to supply the total covariance and total drift
    in the time dependent case where possible.

    #### Example
    The following is an example of a 1 dimensional brownian motion using default
    arguments of zero drift and unit volatility.

    ```python
    process = bm.BrownianMotion()
    times = np.array([0.2, 0.33, 0.7, 0.9, 1.88])
    num_samples = 10000
    with tf.Session() as sess:
      paths = sess.run(process.sample_paths(
          times,
          num_samples=num_samples,
          initial_state=np.array(0.1),
          seed=1234))

    # Compute the means at the specified times.
    means = np.mean(paths, axis=0)
    print (means)  # Mean values will be near 0.1 for each time

    # Compute the covariances at the given times
    covars = np.cov(paths.reshape([num_samples, 5]), rowvar=False)

    # covars is a 5 x 5 covariance matrix.
    # Expected result is that Covar(X(t), X(t')) = min(t, t')
    expected = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
    print ("Computed Covars: {}, True Covars: {}".format(covars, expected))
    ```

    Args:
      dim: Python int greater than or equal to 1. The dimension of the Brownian
        motion.
        Default value: 1 (i.e. a one dimensional brownian process).
      drift: The drift of the process. The type and shape of the value must be
        one of the following (in increasing order of generality) (a) A real
        scalar `Tensor`. This corresponds to a time and component independent
        drift. Every component of the Brownian motion has the same drift rate
        equal to this value. (b) A real `Tensor` of shape `[dim]`. This
        corresponds to a time independent drift with the `i`th component as the
        drift rate of the `i`th component of the Brownian motion. (c) A Python
        callable accepting a single positive `Tensor` of general shape (referred
        to as `times_shape`) and returning a `Tensor` of shape `times_shape +
        [dim]`. The input argument is the times at which the drift needs to be
        evaluated. This case corresponds to a general time and direction
        dependent drift rate.
        Default value: None which maps to zero drift.
      volatility: The volatility of the process. The type and shape of the
        supplied value must be one of the following (in increasing order of
        generality) (a) A positive real scalar `Tensor`. This corresponds to a
        time independent, diagonal volatility matrix. The `(i, j)` component of
        the full volatility matrix is equal to zero if `i != j` and equal to the
        supplied value otherwise. (b) A positive real `Tensor` of shape `[dim]`.
        This corresponds to a time independent volatility matrix with zero
        correlation. The `(i, j)` component of the full volatility matrix is
        equal to zero `i != j` and equal to the `i`th component of the supplied
        value otherwise. (c) A positive definite real `Tensor` of shape `[dim,
        dim]`. The full time independent volatility matrix. (d) A Python
        callable accepting a single positive `Tensor` of general shape (referred
        to as `times_shape`) and returning a `Tensor` of shape `times_shape +
        [dim, dim]`. The input argument are the times at which the volatility
        needs to be evaluated. This case corresponds to a general time and axis
        dependent volatility matrix.
        Default value: None which maps to a volatility matrix equal to identity.
      total_drift_fn: Optional Python callable to compute the integrated drift
        rate between two times. The callable should accept two real `Tensor`
        arguments. The first argument contains the start times and the second,
        the end times of the time intervals for which the total drift is to be
        computed. Both the `Tensor` arguments are of the same dtype and shape.
        The return value of the callable should be a real `Tensor` of the same
        dtype as the input arguments and of shape `times_shape + [dim]` where
        `times_shape` is the shape of the times `Tensor`. Note that it is an
        error to supply this parameter if the `drift` is not supplied.
        Default value: None.
      total_covariance_fn: A Python callable returning the integrated covariance
        rate between two times. The callable should accept two real `Tensor`
        arguments. The first argument is the start times and the second is the
        end times of the time intervals for which the total covariance is
        needed. Both the `Tensor` arguments are of the same dtype and shape. The
        return value of the callable is a real `Tensor` of the same dtype as the
        input arguments and of shape `times_shape + [dim, dim]` where
        `times_shape` is the shape of the times `Tensor`. Note that it is an
        error to supply this argument if the `volatility` is not supplied.
        Default value: None.
      dtype: The default dtype to use when converting values to `Tensor`s.
        Default value: None which means that default dtypes inferred by
          TensorFlow are used.
      name: str. The name scope under which ops created by the methods of this
        class are nested.
        Default value: None which maps to the default name `brownian_motion`.

    Raises:
      ValueError if the dimension is less than 1 or if total drift is supplied
        but drift is not supplied or if the total covariance is supplied but
        but volatility is not supplied.
    """
    super(BrownianMotion, self).__init__()

    if dim < 1:
      raise ValueError('Dimension must be 1 or greater.')
    if drift is None and total_drift_fn is not None:
      raise ValueError('total_drift_fn must not be supplied if drift'
                       ' is not supplied.')
    if volatility is None and total_covariance_fn is not None:
      raise ValueError('total_covariance_fn must not be supplied if drift'
                       ' is not supplied.')
    self._dim = dim
    self._dtype = dtype
    self._name = name or 'brownian_motion'

    drift_fn, total_drift_fn = bmu.construct_drift_data(drift, total_drift_fn,
                                                        dim, dtype)
    self._drift_fn = drift_fn
    self._total_drift_fn = total_drift_fn

    vol_fn, total_covar_fn = bmu.construct_vol_data(volatility,
                                                    total_covariance_fn, dim,
                                                    dtype)
    self._volatility_fn = vol_fn
    self._total_covariance_fn = total_covar_fn

  # Override
  def dim(self):
    """The dimension of the process."""
    return self._dim

  # Override
  def dtype(self):
    """The data type of process realizations."""
    return self._dtype

  # Override
  def name(self):
    """The name to give to the ops created by this class."""
    return self._name

  # Override
  def drift_fn(self):
    return lambda t, x: self._drift_fn(t)

  # Override
  def volatility_fn(self):
    return lambda t, x: self._volatility_fn(t)

  def total_drift_fn(self):
    """The integrated drift of the process.

    Returns:
      None or a Python callable. None is returned if the input drift was a
      callable and no total drift function was supplied.
      The callable returns the integrated drift rate between two times.
      It accepts two real `Tensor` arguments. The first argument is the
      left end point and the second is the right end point of the time interval
      for which the total drift is needed. Both the `Tensor` arguments are of
      the same dtype and shape. The return value of the callable is
      a real `Tensor` of the same dtype as the input arguments and of shape
      `times_shape + [dim]` where `times_shape` is the shape of the times
      `Tensor`.
    """
    return self._total_drift_fn

  def total_covariance_fn(self):
    """The total covariance of the process between two times.

    Returns:
      A Python callable returning the integrated covariances between two times.
      The callable accepts two real `Tensor` arguments. The first argument
      is the left end point and the second is the right end point of the time
      interval for which the total covariance is needed.

      The shape of the two input arguments and their dtypes must match.
      The output of the callable is a `Tensor` of shape
      `times_shape + [dim, dim]` containing the integrated covariance matrix
      between the start times and end times.
    """
    return self._total_covariance_fn

  # Override
  def sample_paths(self,
                   times,
                   num_samples=1,
                   initial_state=None,
                   random_type=None,
                   seed=None,
                   swap_memory=True,
                   name=None,
                   **kwargs):
    """Returns a sample of paths from the process.

    Generates samples of paths from the process at the specified time points.

    Args:
      times: Rank 1 `Tensor` of increasing positive real values. The times at
        which the path points are to be evaluated.
      num_samples: Positive scalar `int`. The number of paths to draw.
      initial_state: `Tensor` of shape `[dim]`. The initial state of the
        process.
        Default value: None which maps to a zero initial state.
      random_type: Enum value of `RandomType`. The type of (quasi)-random number
        generator to use to generate the paths.
        Default value: None which maps to the standard pseudo-random numbers.
      seed: Python `int`. The random seed to use. If not supplied, no seed is
        set.
      swap_memory: Whether GPU-CPU memory swap is enabled for this op. See
        equivalent flag in `tf.while_loop` documentation for more details.
        Useful when computing a gradient of the op since `tf.while_loop` is used
        to propagate stochastic process in time.
      name: str. The name to give this op. If not supplied, default name of
        `sample_paths` is used.
      **kwargs: parameters, specific to Euler schema: `grid_step` is rank 0 real
        `Tensor` - maximal distance between points in grid in Euler schema. Note
        that Euler sampling is only used if it is not possible to do exact
        sampling because total drift or total covariance are unavailable.

    Returns:
     A real `Tensor` of shape [num_samples, k, n] where `k` is the size of the
        `times`, `n` is the dimension of the process.
    """
    if self._total_drift_fn is None or self._total_covariance_fn is None:
      return super(BrownianMotion, self).sample_paths(
          times,
          num_samples=num_samples,
          initial_state=initial_state,
          random_type=random_type,
          seed=seed,
          name=name,
          **kwargs)

    default_name = self._name + '_sample_path'
    with tf.compat.v1.name_scope(
        name, default_name=default_name, values=[times, initial_state]):
      end_times = tf.convert_to_tensor(times, dtype=self.dtype())
      start_times = tf.concat(
          [tf.zeros([1], dtype=end_times.dtype), end_times[:-1]], axis=0)
      paths = self._exact_sampling(end_times, start_times, num_samples,
                                   initial_state, random_type, seed)
      if initial_state is not None:
        return paths + initial_state
      return paths

  def _exact_sampling(self, end_times, start_times, num_samples, initial_state,
                      random_type, seed):
    """Returns a sample of paths from the process."""
    non_decreasing = tf.debugging.assert_greater_equal(
        end_times, start_times, message='Sampling times must be non-decreasing')
    starts_non_negative = tf.debugging.assert_greater_equal(
        start_times,
        tf.zeros_like(start_times),
        message='Sampling times must not be < 0.')
    with tf.compat.v1.control_dependencies(
        [starts_non_negative, non_decreasing]):
      drifts = self._total_drift_fn(start_times, end_times)
      covars = self._total_covariance_fn(start_times, end_times)
      # path_deltas are of shape [num_samples, size(times), dim].
      path_deltas = mvn.multivariate_normal((num_samples,),
                                            mean=drifts,
                                            covariance_matrix=covars,
                                            random_type=random_type,
                                            seed=seed)
      paths = tf.cumsum(path_deltas, axis=1)
    return paths

  # Override
  def fd_solver_backward(self,
                         final_time,
                         discounting_fn=None,
                         grid_spec=None,
                         time_step=None,
                         time_step_fn=None,
                         values_batch_size=1,
                         name=None,
                         **kwargs):
    """Returns a solver for solving Feynman-Kac PDE associated to the process.

    Represents the PDE

    ```None
      V_t + Sum[a_i(t) V_i, 1<=i<=n] +
        (1/2) Sum[ D_{ij}(t) V_{ij}, 1 <= i,j <= n] - r(t, x) V = 0
    ```

    In the above, `V_t` is the derivative of `V` with respect to `t`,
    `V_i` is the partial derivative with respect to `x_i` and `V_{ij}` the
    (mixed) partial derivative with respect to `x_i` and `x_j`. `D_{ij}` are
    the components of the diffusion tensor:

    ```None
      D_{ij}(t) = (Sigma . Transpose[Sigma])_{ij}(t)
    ```

    This method provides a finite difference solver to solve the above
    differential equation. Whereas the coefficients `mu` and `D` are properties
    of the SDE itself, the function `r(t, x)` may be arbitrarily specified
    by the user (the parameter `discounting_fn` to this method).

    Args:
      final_time: Positive scalar real `Tensor`. The time of the final value.
        The solver is initialized to this final time.
      discounting_fn: Python callable corresponding to the function `r(t, x)` in
        the description above. The callable accepts two positional arguments.
        The first argument is the time at which the discount rate function is
        needed. The second argument contains the values of the state at which
        the discount is to be computed.
        Default value: None which maps to `r(t, x) = 0`.
      grid_spec: An iterable convertible to a tuple containing at least the
        attributes named 'grid', 'dim' and 'sizes'. For a full description of
        the fields and expected types, see `grids.GridSpec` which provides the
        canonical specification of this object.
      time_step: A real positive scalar `Tensor` or None. The fixed
        discretization parameter along the time dimension. Either this argument
        or the `time_step_fn` must be specified. It is an error to specify both.
        Default value: None.
      time_step_fn: A callable accepting an instance of `grids.GridStepperState`
        and returning the size of the next time step as a real scalar tensor.
        This argument allows usage of a non-constant time step while stepping
        back. If not specified, the `time_step` parameter must be specified. It
        is an error to specify both.
        Default value: None.
      values_batch_size: A positive Python int. The batch size of values to be
        propagated simultaneously.
        Default value: 1.
      name: Python str. The name to give this op.
        Default value: None which maps to `fd_solver_backward`.
      **kwargs: Any other keyword args needed.

    Returns:
      An instance of `BackwardGridStepper` configured for solving the
      Feynman-Kac PDE associated to this process.
    """
    # TODO(b/141669934): Implement the method once the boundary conditions
    # specification is complete.
    raise NotImplementedError('Finite difference solver not implemented')


def _prefer_static_shape(tensor):
  """Returns the static shape if fully specified else the dynamic shape."""
  tensor = tf.convert_to_tensor(tensor)
  static_shape = tensor.shape
  if static_shape.is_fully_defined():
    return static_shape
  return tf.shape(tensor)


def _prefer_static_rank(tensor):
  """Returns the static rank if fully specified else the dynamic rank."""
  tensor = tf.convert_to_tensor(tensor)
  if tensor.shape.rank is None:
    return tf.rank(tensor)
  return tensor.shape.rank
