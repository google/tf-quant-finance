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
      seed: Seed for the random number generator. The seed is
        only relevant if `random_type` is one of
        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
        `HALTON_RANDOMIZED` the seed should be an Python integer. For
        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer
        `Tensor` of shape `[2]`.
        Default value: `None` which means no seed is set.
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


def maybe_update_along_axis(*,
                            tensor,
                            new_tensor,
                            axis,
                            ind,
                            do_update,
                            dtype=None,
                            name=None):
  """Replace `tensor` entries with `new_tensor` along a given axis.

  This updates elements of `tensor` that correspond to the elements returned by
  `numpy.take(updated, ind, axis)` with the corresponding elements of
  `new_tensor`.

  # Example
  ```python
  tensor = tf.ones([5, 4, 3, 2])
  new_tensor = tf.zeros([5, 4, 3, 2])
  updated_tensor = maybe_update_along_axis(tensor=tensor,
                                           new_tensor=new_tensor,
                                           axis=1,
                                           ind=2,
                                           do_update=True)
  # Returns a `Tensor` of ones where
  # `updated_tensor[:, 2, :, :].numpy() == 0`
  ```
  If the `do_update` is set to `False`, then the update does not happen unless
  the number of dimensions along the `axis` is equal to 1. This functionality
  is useful when, for example, aggregating samples of an Ito process.

  Args:
    tensor: A `Tensor` of any shape and `dtype`.
    new_tensor: A `Tensor` of the same `dtype` as `tensor` and of shape
      broadcastable with `tensor`.
    axis: A Python integer. The axis of `tensor` along which the elements have
      to be updated.
    ind: An int32 scalar `Tensor` that denotes an index on the `axis` which
      defines the updated slice of `tensor` (see example above).
    do_update: A bool scalar `Tensor`. If `False`, the output is the same as
      `tensor`, unless  the dimension of the `tensor` along the `axis` is equal
      to 1.
    dtype: The `dtype` of the input `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give this op.
      Default value: `None` which maps to `maybe_update_along_axis`.

  Returns:
    A `Tensor` of the same shape and `dtype` as `tensor`.
  """
  name = name or 'maybe_update_along_axis'
  with tf.name_scope(name):
    tensor = tf.convert_to_tensor(tensor, dtype=dtype,
                                  name='tensor')
    dtype = tensor.dtype
    new_tensor = tf.convert_to_tensor(new_tensor, dtype=dtype,
                                      name='new_tensor')
    ind = tf.convert_to_tensor(ind, name='ind')
    do_update = tf.convert_to_tensor(do_update, name='do_update')
    size_along_axis = tensor.shape.as_list()[axis]
    def _write_update_to_result():
      one_hot = tf.one_hot(ind, depth=size_along_axis)
      mask_shape = len(tensor.shape) * [1]
      mask_shape[axis] = size_along_axis
      mask = tf.reshape(one_hot > 0, mask_shape)
      return tf.where(mask, new_tensor, tensor)
    # Update only if size_along_axis > 1.
    if size_along_axis > 1:
      return tf.cond(do_update,
                     _write_update_to_result,
                     lambda: tensor)
    else:
      return new_tensor


def prepare_grid(*, times, time_step, dtype):
  """Prepares grid of times for path generation.

  Args:
    times:  Rank 1 `Tensor` of increasing positive real values. The times at
      which the path points are to be evaluated.
    time_step: Rank 0 real `Tensor`. Maximal distance between points in
      resulting grid.
    dtype: `tf.Dtype` of the input and output `Tensor`s.

  Returns:
    Tuple `(all_times, mask, time_points)`.
    `all_times` is a 1-D real `Tensor` containing all points from 'times` and
    the uniform grid of points between `[0, times[-1]]` with grid size equal to
    `time_step`. The `Tensor` is sorted in ascending order and may contain
    duplicates.
    `mask` is a boolean 1-D `Tensor` of the same shape as 'all_times', showing
    which elements of 'all_times' correspond to THE values from `times`.
    Guarantees that times[0]=0 and mask[0]=False.
    `time_indices`. An integer `Tensor` of the same shape as `times` indicating
    `times` indices in `all_times`.
  """
  grid = tf.range(0.0, times[-1], time_step, dtype=dtype)
  all_times = tf.concat([times, grid], axis=0)
  # Remove duplicate points
  # all_times = tf.unique(all_times).y
  # Sort sequence. Identify the time indices of interest
  # TODO(b/169400743): use tf.sort instead of argsort and casting when XLA
  # float64 support is extended for tf.sort
  args = tf.argsort(tf.cast(all_times, dtype=tf.float32))
  all_times = tf.gather(all_times, args)
  # Remove duplicate points
  duplicate_tol = 1e-10 if dtype == tf.float64 else 1e-6
  dt = all_times[1:] - all_times[:-1]
  dt = tf.concat([[1.0], dt], axis=-1)
  duplicate_mask = tf.math.greater(dt, duplicate_tol)
  all_times = tf.boolean_mask(all_times, duplicate_mask)

  time_indices = tf.searchsorted(all_times, times, out_type=tf.int32)
  # Create a boolean mask to identify the iterations that have to be recorded.
  mask_sparse = tf.sparse.SparseTensor(
      indices=tf.expand_dims(
          tf.cast(time_indices, dtype=tf.int64), axis=1),
      values=tf.fill(tf.shape(times), True),
      dense_shape=tf.shape(all_times, out_type=tf.int64))
  mask = tf.sparse.to_dense(mask_sparse)
  # all_times = tf.concat([[0.0], all_times], axis=0)
  # mask = tf.concat([[False], mask], axis=0)
  # time_indices = time_indices + 1
  return all_times, mask, time_indices


def block_diagonal_to_dense(*matrices):
  """Given a sequence of matrices, creates a block-diagonal dense matrix."""
  operators = [tf.linalg.LinearOperatorFullMatrix(m) for m in matrices]
  return tf.linalg.LinearOperatorBlockDiag(operators).to_dense()


def cumsum_using_matvec(input_tensor):
  """Computes cumsum using matrix algebra."""
  dtype = input_tensor.dtype
  axis_length = input_tensor.shape.as_list()[-1]
  ones = tf.ones([axis_length, axis_length], dtype=dtype)
  lower_triangular = tf.linalg.band_part(ones, -1, 0)
  cumsum = tf.linalg.matvec(lower_triangular, input_tensor)
  return cumsum


def cumprod_using_matvec(input_tensor):
  """Computes cumprod using matrix algebra."""
  dtype = input_tensor.dtype
  axis_length = input_tensor.shape.as_list()[-1]
  ones = tf.ones([axis_length, axis_length], dtype=dtype)
  lower_triangular = tf.linalg.band_part(ones, -1, 0)
  cumsum = tf.linalg.matvec(lower_triangular, tf.math.log(input_tensor))
  return tf.math.exp(cumsum)
