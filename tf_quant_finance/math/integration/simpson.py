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

"""Composite Simpson's algorithm for numeric integration."""


import tensorflow.compat.v2 as tf


def simpson(func, lower, upper, num_points=1001, dtype=None, name=None):
  """Evaluates definite integral using composite Simpson's 1/3 rule.

  Integrates `func` using composite Simpson's 1/3 rule [1].

  Evaluates function at points of evenly spaced grid of `num_points` points,
  then uses obtained values to interpolate `func` with quadratic polynomials
  and integrates these polynomials.

  #### References
  [1] Weisstein, Eric W. "Simpson's Rule." From MathWorld - A Wolfram Web
      Resource. http://mathworld.wolfram.com/SimpsonsRule.html

  #### Example
  ```python
    f = lambda x: x*x
    a = tf.constant(0.0)
    b = tf.constant(3.0)
    integrate(f, a, b, num_points=1001) # 9.0
  ```

  Args:
    func: Python callable representing a function to be integrated. It must be a
      callable of a single `Tensor` parameter and return a `Tensor` of the same
      shape and dtype as its input. It will be called with a `Tesnor` of shape
      `lower.shape + [n]` (where n is integer number of points) and of the same
      `dtype` as `lower`.
    lower: `Tensor` or Python float representing the lower limits of
      integration. `func` will be integrated between each pair of points defined
      by `lower` and `upper`.
    upper: `Tensor` of the same shape and dtype as `lower` or Python float
      representing the upper limits of intergation.
    num_points: Scalar int32 `Tensor`. Number of points at which function `func`
      will be evaluated. Must be odd and at least 3.
      Default value: 1001.
    dtype: Optional `tf.Dtype`. If supplied, the dtype for the `lower` and
      `upper`. Result will have the same dtype.
      Default value: None which maps to dtype of `lower`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'integrate_simpson_composite'.

  Returns:
    `Tensor` of shape `func_batch_shape + limits_batch_shape`, containing
      value of the definite integral.

  """
  with tf.compat.v1.name_scope(
      name, default_name='integrate_simpson_composite', values=[lower, upper]):
    lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
    dtype = lower.dtype
    upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
    num_points = tf.convert_to_tensor(
        num_points, dtype=tf.int32, name='num_points')

    assertions = [
        tf.debugging.assert_greater_equal(num_points, 3),
        tf.debugging.assert_equal(num_points % 2, 1),
    ]

    with tf.compat.v1.control_dependencies(assertions):
      dx = (upper - lower) / (tf.cast(num_points, dtype=dtype) - 1)
      dx_expand = tf.expand_dims(dx, -1)
      lower_exp = tf.expand_dims(lower, -1)
      grid = lower_exp + dx_expand * tf.cast(tf.range(num_points), dtype=dtype)
      weights_first = tf.constant([1.0], dtype=dtype)
      weights_mid = tf.tile(
          tf.constant([4.0, 2.0], dtype=dtype), [(num_points - 3) // 2])
      weights_last = tf.constant([4.0, 1.0], dtype=dtype)
      weights = tf.concat([weights_first, weights_mid, weights_last], axis=0)

    return tf.reduce_sum(func(grid) * weights, axis=-1) * dx / 3
