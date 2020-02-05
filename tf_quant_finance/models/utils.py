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
"""Common methods for model building."""

import tensorflow.compat.v2 as tf
from tf_quant_finance.math import random_ops as random


def generate_mc_normal_draws(num_normal_draws,
                             num_time_steps,
                             num_sample_paths,
                             random_type,
                             skip=0,
                             seed=None,
                             dtype=None,
                             name=None):
  """Generates normal random samples to be consumed by a Monte Carlo algorithm.

  Many of Monte Carlo (MC) algorithms can be re-written so that all necessary
  random (or quasi-random) variables are drawn in advance as a `Tensor` of
  shape `[num_time_steps, num_samples, num_normal_draws]`, where
  `num_time_steps` is the number of time steps Monte Carlo algorithm performs,
  `num_sample_paths` is a number of sample paths of the Monte Carlo algorithm
  and `num_normal_draws` is a number of independent normal draws per sample
  paths.
  For example, in order to use quasi-random numbers in a Monte Carlo algorithm,
  the samples have to be drawn in advance.
  The function generates a `Tensor`, say, `x` in a format such that for a
  quasi-`random_type` `x[i]` is correspond to different dimensions of the
  quasi-random sequence, so that it can be used in a Monte Carlo algorithm

  Args:
    num_normal_draws: A scalar int32 `Tensor`. The number of independent normal
      draws at each time step for each sample path. Should be a graph
      compilation constant.
    num_time_steps: A scalar int32 `Tensor`. The number of time steps at which
      to draw the independent normal samples. Should be a graph compilation
      constant.
    num_sample_paths: A scalar int32 `Tensor`. The number of trajectories (e.g.,
      Monte Carlo paths) for which to draw the independent normal samples.
      Should be a graph compilation constant.
    random_type: Enum value of `tff.math.random.RandomType`. The type of
      (quasi)-random number generator to use to generate the paths.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[PSEUDO, PSEUDO_ANTITHETIC, HALTON_RANDOMIZED]`.
    dtype: The `dtype` of the output `Tensor`.
      Default value: `None` which maps to `float32`.
    name: Python string. The name to give this op.
      Default value: `None` which maps to `generate_mc_normal_draws`.

  Returns:
   A `Tensor` of shape `[num_time_steps, num_sample_paths, num_normal_draws]`.
  """
  if name is None:
    name = 'generate_mc_normal_draws'
  if skip is None:
    skip = 0
  with tf.name_scope(name):
    if dtype is None:
      dtype = tf.float32
    # In case of quasi-random draws, the total dimension of the draws should be
    # `num_time_steps * dim`
    total_dimension = tf.zeros([num_time_steps * num_normal_draws], dtype=dtype,
                               name='total_dimension')
    normal_draws = random.mv_normal_sample(
        [num_sample_paths], mean=total_dimension,
        random_type=random_type,
        seed=seed,
        skip=skip)
    # Reshape and transpose
    normal_draws = tf.reshape(
        normal_draws, [num_sample_paths, num_time_steps, num_normal_draws])
    # Shape [steps_num, num_samples, dim]
    normal_draws = tf.transpose(normal_draws, [1, 0, 2])
    return normal_draws
