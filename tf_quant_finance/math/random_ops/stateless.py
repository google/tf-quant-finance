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

"""Stateless random ops.

Implement some of the stateless ops, which produce random numbers as a
deterministic function of seed.
"""


import tensorflow.compat.v2 as tf


def stateless_random_shuffle(input_tensor, seed, name=None):
  """Produces stateless random shuffle of the 1st dimension of an input Tensor.

  This is a stateless version of `tf.random_shuffle`. If run twice with the same
  seed, produces the same result.

  Example
  ```python
  identity_shuffle = tf.range(100)
  random_shuffle = stateless_random_shuffle(identity_shuffle, seed=(42, 2))
  ```

  Args:
    input_tensor: float32, float64, int32 or int64 1-D Tensor.
    seed: int32 or int64 Tensor of shape [2].
    name: Python `str` name prefixed to ops created by this function.

  Returns:
    A Tensor of the same shape and dtype as `input_tensor`.
  """
  with tf.compat.v1.name_scope(name,
                               default_name='stateless_random_shuffle',
                               values=[input_tensor, seed]):
    input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
    seed = tf.convert_to_tensor(seed, name='random_seed')
    uniforms = tf.random.stateless_uniform(
        shape=[tf.shape(input_tensor)[0]], seed=seed, dtype=tf.float64)
  return tf.gather(input_tensor, tf.argsort(uniforms, stable=True, axis=0))
