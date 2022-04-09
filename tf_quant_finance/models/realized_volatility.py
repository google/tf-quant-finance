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
"""Implements realized volatility statistics."""

import enum

import tensorflow.compat.v2 as tf

from tf_quant_finance.math import diff_ops


@enum.unique
class PathScale(enum.Enum):
  """Represents what scale a path is on.

  * `ORIGINAL`: Represents a path on its original scale.
  * `LOG`: Represents log scale values of a path.
  """
  ORIGINAL = 1
  LOG = 2


@enum.unique
class ReturnsType(enum.Enum):
  """Represents types of return processes.

  * `ABS`: Represents absolute returns.
  * `LOG`: Represents log returns.
  """
  ABS = 1
  LOG = 2


def realized_volatility(sample_paths,
                        times=None,
                        scaling_factors=None,
                        returns_type=ReturnsType.LOG,
                        path_scale=PathScale.ORIGINAL,
                        axis=-1,
                        dtype=None,
                        name=None):
  r"""Calculates the total realized volatility for each path.

  With `t_i, i=0,...,N` being a discrete sequence of times at which a series
  `S_{t_k}, i=0,...,N` is observed. The logarithmic returns (`ReturnsType.LOG`)
  process is given by:

  ```
  R_k = log(S_{t_{k}} / S_{t_{k-1}})^2
  ```

  Whereas for absolute returns (`ReturnsType.ABS`) it is given by:

  ```
  R_k = |S_{t_k}} - S_{t_{k-1}})| / |S_{t_{k-1}}|
  ```

  Letting `dt_k = t_k - t_{k-1}` the realized variance is then calculated as:

  ```
  V = c * f( \Sum_{k=1}^{N-1} R_k / dt_k )
  ```

  Where `f` is the square root for logarithmic returns and the identity function
  for absolute returns. If `times` is not supplied then it is assumed that
  `dt_k = 1` everywhere. The arbitrary scaling factor `c` enables various
  flavours of averaging or annualization (for examples of which see [1] or
  section 9.7 of [2]).

  #### Examples

  Calculation of realized logarithmic volatility as in [1]:

  ```python
  import tensorflow as tf
  import tf_quant_finance as tff
  dtype=tf.float64
  num_samples = 1000
  num_times = 252
  seed = (1, 2)
  annual_vol = 20
  sigma = annual_vol / (100 * np.sqrt(num_times - 1))
  mu = -0.5*sigma**2

  gbm = tff.models.GeometricBrownianMotion(mu=mu, sigma=sigma, dtype=dtype)
  sample_paths = gbm.sample_paths(
      times=range(num_times),
      num_samples=num_samples,
      seed=seed,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)

  annualization = 100 * np.sqrt( (num_times / (num_times - 1)) )
  tf.math.reduce_mean(
    realized_volatility(sample_paths,
                        scaling_factors=annualization,
                        path_scale=PathScale.ORIGINAL,
                        axis=1))
  # 20.03408344960287
  ```

  Carrying on with the same paths the realized absolute volatility (`RV_d2` in
  [3]) is:

  ```
  scaling = 100 * np.sqrt((np.pi/(2 * (num_times-1))))
  tf.math.reduce_mean(
    realized_volatility(sample_paths,
                        scaling_factors=scaling,
                        returns_type=ReturnsType.ABS,
                        path_scale=PathScale.LOG))
  # 19.811590402553158
  ```

  #### References:
  [1]: CBOE. Summary Product Specifications Chart for S&P 500 Variance Futures.
  2012.
  https://cdn.cboe.com/resources/futures/sp_500_variance_futures_contract.pdf
  [2]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
  guide. Chapter 5. 2011.
  [3]: Zhu, S.P. and Lian, G.H., 2015. Analytically pricing volatility swaps
  under stochastic volatility. Journal of Computational and Applied Mathematics.

  Args:
    sample_paths: A real `Tensor` of shape
      `batch_shape_0 + [N] + batch_shape_1`.
    times: A real `Tensor` of shape compatible with `batch_shape_0 + [N] +
      batch_shape_1`. The times represented on the axis of interest (the `t_k`).
      Default value: None. Resulting in the assumption of unit time increments.
    scaling_factors: An optional real `Tensor` of shape compatible with
      `batch_shape_0 + batch_shape_1`. Any scaling factors to be applied to the
      result (e.g. for annualization).
      Default value: `None`. Resulting in `c=1` in the above calculation.
    returns_type: Value of ReturnsType. Indicates which definition of returns
      should be used.
      Default value: ReturnsType.LOG, representing logarithmic returns.
    path_scale: Value of PathScale. Indicates which space the supplied
      `sample_paths` are in. If required the paths will then be transformed onto
      the appropriate scale.
      Default value: PathScale.ORIGINAL.
    axis: Python int. The axis along which to calculate the statistic.
      Default value: -1 (the final axis).
    dtype: `tf.DType`. If supplied the dtype for the input and output `Tensor`s.
      Default value: `None` leading to use of `sample_paths`.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to 'realized_volatility'.

  Returns:
    Tensor of shape equal to `batch_shape_0 + batch_shape_1` (i.e. with axis
      `axis` having been reduced over).
  """
  with tf.name_scope(name or 'realized_volatility'):
    sample_paths = tf.convert_to_tensor(sample_paths, dtype=dtype,
                                        name='sample_paths')
    dtype = dtype or sample_paths.dtype
    if returns_type == ReturnsType.LOG:
      component_transform = lambda t: tf.pow(t, 2)
      result_transform = tf.math.sqrt
      if path_scale == PathScale.ORIGINAL:
        transformed_paths = tf.math.log(sample_paths)
      elif path_scale == PathScale.LOG:
        transformed_paths = sample_paths
    elif returns_type == ReturnsType.ABS:
      component_transform = tf.math.abs
      result_transform = tf.identity
      if path_scale == PathScale.ORIGINAL:
        transformed_paths = sample_paths
      elif path_scale == PathScale.LOG:
        transformed_paths = tf.math.exp(sample_paths)

    diffs = component_transform(
        diff_ops.diff(transformed_paths, order=1, exclusive=True, axis=axis))
    denominators = 1
    if times is not None:
      times = tf.convert_to_tensor(times, dtype=dtype, name='times')
      denominators = diff_ops.diff(times, order=1, exclusive=True, axis=axis)
    if returns_type == ReturnsType.ABS:
      slices = transformed_paths.shape.rank * [slice(None)]
      slices[axis] = slice(None, -1)
      denominators = denominators * component_transform(
          transformed_paths[slices])
    path_statistics = result_transform(
        tf.math.reduce_sum(diffs / denominators, axis=axis))
    if scaling_factors is not None:
      scaling_factors = tf.convert_to_tensor(
          scaling_factors, dtype=dtype, name='scaling_factors')
      return scaling_factors * path_statistics
    return path_statistics
