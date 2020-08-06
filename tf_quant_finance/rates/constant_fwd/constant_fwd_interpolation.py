# Lint as: python3
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

"""Constant forward interpolation method."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.interpolation.linear import linear_interpolation


def interpolate(interpolation_times,
                reference_times,
                reference_yields,
                dtype=None,
                name=None):
  """Performs the constant forward interpolation for supplied points.

  Given an interest rate yield curve whose maturities and the corresponding
  (continuously compounded) yields are in `reference_times` and
  `reference_yields`, this function returns interpolated yields at
  `interpolation_times` using the constant forward interpolation.

  Let `t_i, i=1,...,n` and `y_i, i=1,...,n` denote the reference_times and
  reference_yields respectively. If `t` is a maturity for which the
  interpolation is desired such that t_{i-1} <= t <= t_i, then constant forward
  interpolation produces the corresponding yield, `y_t`, such that the forward
  rate in the interval `[t_{i-1},t]` is the same as the forward rate in the
  interval `[t_{i-1},t_i]`. Mathematically, this is the same as linearly
  interpolating `t*y_t` using the curve `t_i, t_i*y_i`.

  `reference_times` must be strictly increasing but `reference_yields` don't
  need to be because we don't require the rate curve to be monotonic.

  #### Examples

  ```python
  interpolation_times = [1, 3, 6, 7, 8, 15, 18, 25, 30]
  # `reference_times` must be increasing, but `reference_yields` don't need to
  # be.
  reference_times = [0.0, 2.0, 6.0, 8.0, 18.0, 30.0]
  reference_yields = [0.01, 0.02, 0.015, 0.014, 0.02, 0.025]
  result = interpolate(interpolation_times, reference_times, reference_yields)
  ```

  Args:
    interpolation_times: The times at which interpolation is desired. A N-D
      `Tensor` of real dtype where the first N-1 dimensions represent the
      batching dimensions.
    reference_times: Maturities in the input yield curve. A N-D `Tensor` of
      real dtype where the first N-1 dimensions represent the batching
      dimensions. Should be sorted in increasing order.
    reference_yields: Continuously compounded yields corresponding to
      `reference_times`. A N-D `Tensor` of real dtype. Should have the
      compatible shape as `x_data`.
    dtype: Optional tf.dtype for `interpolation_times`, reference_times`,
      and `reference_yields`. If not specified, the dtype of the inputs will be
      used.
    name: Python str. The name prefixed to the ops created by this function. If
      not supplied, the default name 'constant_fwd_interpolation' is used.

  Returns:
    A N-D `Tensor` of real dtype with the same shape as `interpolations_times`
      containing the interpolated yields.
  """
  with tf.compat.v1.name_scope(
      name,
      default_name='constant_fwd_interpolation',
      values=[interpolation_times, reference_times, reference_yields]):
    interpolation_times = tf.convert_to_tensor(interpolation_times, dtype=dtype)
    dtype = dtype or interpolation_times.dtype
    reference_times = tf.convert_to_tensor(reference_times, dtype=dtype)
    reference_yields = tf.convert_to_tensor(reference_yields, dtype=dtype)

    # Currently only flat extrapolation is being supported. We achieve this by
    # clipping the interpolation times between minimum and maximum times of the
    # input curve.
    reference_times_min = tf.reduce_min(
        reference_times, axis=-1, keepdims=True)
    reference_times_max = tf.reduce_max(
        reference_times, axis=-1, keepdims=True)
    interpolation_times = tf.minimum(
        reference_times_max,
        tf.maximum(reference_times_min, interpolation_times))

    interpolated_prod = linear_interpolation.interpolate(
        interpolation_times, reference_times,
        reference_times * reference_yields, dtype=dtype)
    interpolated = interpolated_prod / interpolation_times
    return interpolated
