# Lint as: python3
# Copyright 2020 Google LLC
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
"""Payoff functions."""

import functools
import tensorflow.compat.v2 as tf


def make_basket_put_payoff(strike_price, dtype=None, name=None):
  """Produces a callable from samples to payoff of a simple basket put option.

  Args:
    strike_price: A `Tensor` of `dtype` consistent with `samples` and shape
      `[num_samples, num_strikes]`.
    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`
      If supplied, represents the `dtype` for the 'strike_price' as well as
      for the input argument of the output payoff callable.
      Default value: `None`, which means that the `dtype` inferred by TensorFlow
      is used.
    name: Python `str` name prefixed to Ops created by the callable created
      by this function.
      Default value: `None` which is mapped to the default name 'put_valuer'

  Returns:
    A callable from `Tensor` of shape `[num_samples, num_exercise_times, dim]`
    and a scalar `Tensor` representing current time to a `Tensor` of shape
    `[num_samples, num_strikes]`.
  """
  strike_price = tf.convert_to_tensor(strike_price, dtype=dtype,
                                      name="strike_price")
  put_valuer = functools.partial(_put_valuer, strike_price=strike_price,
                                 dtype=dtype, name=name)

  return put_valuer


def _put_valuer(sample_paths, time_index, strike_price, dtype=None, name=None):
  """Produces a callable from samples to payoff of a simple basket put option.

  Args:
    sample_paths: A `Tensor` of either `flaot32` or `float64` dtype and of
      shape `[num_samples, num_times, dim]`.
    time_index: An integer scalar `Tensor` that corresponds to the time
      coordinate at which the basis function is computed.
    strike_price: A `Tensor` of the same `dtype` as `sample_paths` and shape
      `[num_samples, num_strikes]`.
    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`
      If supplied, represents the `dtype` for the 'strike_price' as well as
      for the input argument of the output payoff callable.
      Default value: `None`, which means that the `dtype` inferred by TensorFlow
      is used.
    name: Python `str` name prefixed to Ops created by the callable created
      by this function.
      Default value: `None` which is mapped to the default name 'put_valuer'

  Returns:
    A callable from `Tensor` of shape `[num_samples, num_exercise_times, dim]`
    and a scalar `Tensor` representing current time to a `Tensor` of shape
    `[num_samples, num_strikes]`.
  """
  name = name or "put_valuer"
  with tf.name_scope(name):
    strike_price = tf.convert_to_tensor(strike_price, dtype=dtype,
                                        name="strike_price")
    sample_paths = tf.convert_to_tensor(sample_paths, dtype=dtype,
                                        name="sample_paths")
    num_samples, _, dim = sample_paths.shape.as_list()

    slice_sample_paths = tf.slice(sample_paths, [0, time_index, 0],
                                  [num_samples, 1, dim])
    slice_sample_paths = tf.squeeze(slice_sample_paths, 1)
    average = tf.math.reduce_mean(slice_sample_paths, axis=-1, keepdims=True)
    return tf.nn.relu(strike_price - average)
