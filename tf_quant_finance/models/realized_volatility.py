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
  ORIGINAL = 1
  LOG = 2


@enum.unique
class ReturnsType(enum.Enum):
  ABS = 1
  LOG = 2


@enum.unique
class EstimatorType(enum.Enum):
  STANDARD = 1
  BIPOWER = 2


def realized_volatility(sample_paths,
                         times=None,
                         scaling_factors=None,
                         returns_type=ReturnsType.LOG,
                         path_scale=PathScale.ORIGINAL,
                         estimator_type=EstimatorType.STANDARD,
                         axis=-1,
                         dtype=None,
                         name=None):
  """Calculates the total realized volatility for each path.

  With `t_i, i=0,...,N` being a discrete sequence of times at which a series
  `S_{t_i}, i=0,...,N` is observed. The logarithmic returns (`ReturnsType.LOG`)
  process is given by:

  R_k = log(S_{t_k} / S_{t_{k-1}})^2

  Whereas for absolute returns (`ReturnsType.ABS`) it is given by:

  R_k = |S_{t_k} - S_{t_{k-1}}| / |S_{t_{k-1}}|

  Letting `dt_k = t_k - t_{k-1}` the realized variance using the standard
  estimator (`EstimatorType.STANDARD`) is then calculated as:

  V = c * f( Sum_{k=1}^{N-1} R_k / dt_k )

  Where `f` is the square root for logarithmic returns and the identity
  function for absolute returns.

  If the jump-robust Bipower Variation estimator is selected
  (`EstimatorType.BIPOWER`), the calculation for logarithmic returns becomes:

  V = c * sqrt( (pi / 2) * Sum_{k=2}^{N-1} (|R_k| * |R_{k-1}|) / dt_k )

  This adjacent-product approach isolates the continuous volatility component
  by mitigating the impact of discrete price jumps.

  If `times` is not supplied then it is assumed that `dt_k = 1` everywhere.
  The arbitrary scaling factor `c` enables various flavours of averaging or
  annualization.

  Args:
    sample_paths: A real Tensor of shape
      `batch_shape_0 + [N] + batch_shape_1`.
    times: A real Tensor of shape compatible with
      `batch_shape_0 + [N] + batch_shape_1`. The times represented on the
      axis of interest (the `t_k`).
      Default value: `None`, resulting in the assumption of unit time
      increments.
    scaling_factors: An optional real Tensor of shape compatible with
      `batch_shape_0 + batch_shape_1`. Any scaling factors to be applied to
      the result (e.g. for annualization).
      Default value: `None`, resulting in `c=1` in the above calculation.
    returns_type: Value of `ReturnsType`. Indicates which definition of
      returns should be used.
      Default value: `ReturnsType.LOG`, representing logarithmic returns.
    path_scale: Value of `PathScale`. Indicates which space the supplied
      `sample_paths` are in. If required the paths will then be transformed
      onto the appropriate scale.
      Default value: `PathScale.ORIGINAL`.
    estimator_type: Value of `EstimatorType`. Indicates which statistical
      estimator should be used.
      Default value: `EstimatorType.STANDARD`.
    axis: Python int. The axis along which to calculate the statistic.
      Default value: -1 (the final axis).
    dtype: `tf.DType`. If supplied the dtype for the input and output
      Tensors.
      Default value: `None`, leading to the dtype of `sample_paths`.
    name: Python str. The name to give to the ops created by this function.
      Default value: `None` which maps to `'realized_volatility'`.

  Returns:
    Tensor of shape equal to `batch_shape_0 + batch_shape_1` (i.e. with the
    `axis` dimension reduced over).
  """
  with tf.name_scope(name or 'realized_volatility'):
    sample_paths = tf.convert_to_tensor(
        sample_paths, dtype=dtype, name='sample_paths')
    dtype = dtype or sample_paths.dtype

    if returns_type == ReturnsType.LOG:
      component_transform = lambda t: tf.pow(t, 2)
      result_transform = tf.math.sqrt
      if path_scale == PathScale.ORIGINAL:
        transformed_paths = tf.math.log(sample_paths)
      elif path_scale == PathScale.LOG:
        transformed_paths = sample_paths
      else:
        raise ValueError(f"Unsupported path_scale: {path_scale}")
    elif returns_type == ReturnsType.ABS:
      component_transform = tf.math.abs
      result_transform = tf.identity
      if path_scale == PathScale.ORIGINAL:
        transformed_paths = sample_paths
      elif path_scale == PathScale.LOG:
        transformed_paths = tf.math.exp(sample_paths)
      else:
        raise ValueError(f"Unsupported path_scale: {path_scale}")
    else:
      raise ValueError(f"Unsupported returns_type: {returns_type}")

    if estimator_type == EstimatorType.BIPOWER:
      if returns_type != ReturnsType.LOG:
        raise ValueError("Bipower Variation requires ReturnsType.LOG.")

      raw_returns = tf.math.abs(
          diff_ops.diff(
              transformed_paths, order=1, exclusive=True, axis=axis))

      slices_k = [slice(None)] * raw_returns.shape.rank
      slices_k_minus_1 = [slice(None)] * raw_returns.shape.rank
      slices_k[axis] = slice(1, None)
      slices_k_minus_1[axis] = slice(0, -1)

      bipower_terms = (raw_returns[tuple(slices_k)] *
                        raw_returns[tuple(slices_k_minus_1)])
      denominators = 1
      if times is not None:
        times = tf.convert_to_tensor(times, dtype=dtype, name='times')
        dt = diff_ops.diff(times, order=1, exclusive=True, axis=axis)
        denominators = dt[tuple(slices_k)]

      constant_factor = tf.constant(1.5707963267948966, dtype=dtype)
      realized_bipower_var = constant_factor * tf.math.reduce_sum(
          bipower_terms / denominators, axis=axis)
      path_statistics = tf.math.sqrt(realized_bipower_var)

    else:
      diffs = component_transform(
          diff_ops.diff(
              transformed_paths, order=1, exclusive=True, axis=axis))
      denominators = 1
      if times is not None:
        times = tf.convert_to_tensor(times, dtype=dtype, name='times')
        denominators = diff_ops.diff(times, order=1, exclusive=True, axis=axis)
      if returns_type == ReturnsType.ABS:
        slices = [slice(None)] * transformed_paths.shape.rank
        slices[axis] = slice(None, -1)
        denominators = denominators * component_transform(
            transformed_paths[tuple(slices)])
      path_statistics = result_transform(
          tf.math.reduce_sum(diffs / denominators, axis=axis))

    if scaling_factors is not None:
      scaling_factors = tf.convert_to_tensor(
          scaling_factors, dtype=dtype, name='scaling_factors')
      return scaling_factors * path_statistics

    return path_statistics