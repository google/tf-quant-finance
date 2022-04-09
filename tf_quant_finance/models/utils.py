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
                             batch_shape=None,
                             skip=0,
                             seed=None,
                             dtype=None,
                             name=None):
  """Generates normal random samples to be consumed by a Monte Carlo algorithm.

  Many of Monte Carlo (MC) algorithms can be re-written so that all necessary
  random (or quasi-random) variables are drawn in advance as a `Tensor` of
  shape `batch_shape + [num_time_steps, num_samples, num_normal_draws]`, where
  `batch_shape` is the shape of the independent batches of the Monte Carlo
  algorithm, `num_time_steps` is the number of time steps Monte Carlo algorithm
  performs within each batch, `num_sample_paths` is a number of sample paths of
  the Monte Carlo algorithm and `num_normal_draws` is a number of independent
  normal draws per sample path.
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
    batch_shape: This input can be either of type `tf.TensorShape` or a 1-d
      `Tensor` of type `tf.int32` specifying the dimensions of independent
      batches of normal samples to be drawn.
      Default value: `None` which correspond to a single batch of shape
      `tf.TensorShape([])`.
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
   A `Tensor` of shape
   `[num_time_steps] + batch_shape + [num_sample_paths, num_normal_draws]`.
  """
  if name is None:
    name = 'generate_mc_normal_draws'
  if skip is None:
    skip = 0
  with tf.name_scope(name):
    if dtype is None:
      dtype = tf.float32
    if batch_shape is None:
      batch_shape = tf.TensorShape([])

    # In case of quasi-random draws, the total dimension of the draws should be
    # `num_time_steps * dim`
    total_dimension = tf.zeros(
        [num_time_steps * num_normal_draws], dtype=dtype,
        name='total_dimension')
    if random_type in [random.RandomType.PSEUDO_ANTITHETIC,
                       random.RandomType.STATELESS_ANTITHETIC]:
      # Put `num_sample_paths` to the front for antithetic samplers
      sample_shape = tf.concat([[num_sample_paths], batch_shape], axis=0)
      is_antithetic = True
    else:
      # Note that for QMC sequences `num_sample_paths` should follow
      # `batch_shape`
      sample_shape = tf.concat([batch_shape, [num_sample_paths]], axis=0)
      is_antithetic = False
    normal_draws = random.mv_normal_sample(
        sample_shape,
        mean=total_dimension,
        random_type=random_type,
        seed=seed,
        skip=skip)
    # Reshape and transpose
    normal_draws = tf.reshape(
        normal_draws,
        tf.concat([sample_shape, [num_time_steps, num_normal_draws]], axis=0))
    # Shape [steps_num] + batch_shape + [num_samples, dim]
    normal_draws_rank = normal_draws.shape.rank
    if is_antithetic and normal_draws_rank > 3:
      # Permutation for the case when the batch_shape is present
      perm = [normal_draws_rank-2] + list(
          range(1, normal_draws_rank-2)) + [0, normal_draws_rank-1]
    else:
      perm = [normal_draws_rank-2] + list(
          range(normal_draws_rank-2)) + [normal_draws_rank-1]
    normal_draws = tf.transpose(normal_draws, perm=perm)
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
      size_along_axis_dynamic = tf.shape(tensor)[axis]
      one_hot = tf.one_hot(ind, depth=size_along_axis_dynamic)
      mask_size = tensor.shape.rank
      mask_shape = tf.pad(
          [size_along_axis_dynamic],
          paddings=[[axis, mask_size - axis - 1]], constant_values=1)
      mask = tf.reshape(one_hot > 0, mask_shape)
      return tf.where(mask, new_tensor, tensor)
    # Update only if size_along_axis > 1 or if the shape is dynamic
    if size_along_axis is None or size_along_axis > 1:
      return tf.cond(do_update,
                     _write_update_to_result,
                     lambda: tensor)
    else:
      return new_tensor


def prepare_grid(*, times, time_step, dtype, tolerance=None,
                 num_time_steps=None, times_grid=None):
  """Prepares grid of times for path generation.

  Args:
    times:  Rank 1 `Tensor` of increasing positive real values. The times at
      which the path points are to be evaluated.
    time_step: Rank 0 real `Tensor`. Maximal distance between points in
      resulting grid.
    dtype: `tf.Dtype` of the input and output `Tensor`s.
    tolerance: A non-negative scalar `Tensor` specifying the minimum tolerance
      for discernible times on the time grid. Times that are closer than the
      tolerance are perceived to be the same.
      Default value: `None` which maps to `1-e6` if the for single precision
        `dtype` and `1e-10` for double precision `dtype`.
    num_time_steps: Number of points on the grid. If suppied, a uniform grid
      is constructed for `[time_step, times[-1] - time_step]` consisting of
      max(0, num_time_steps - len(times)) points that is then concatenated with
      times. This parameter guarantees the number of points on the time grid
      is `max(len(times), num_time_steps)` and that `times` are included to the
      grid.
      Default value: `None`, which means that a uniform grid is created.
       containing all points from 'times` and the uniform grid of points between
       `[0, times[-1]]` with grid size equal to `time_step`.
    times_grid: An optional rank 1 `Tensor` representing time discretization
      grid. If `times` are not on the grid, then the nearest points from the
      grid are used.
      Default value: `None`, which means that times grid is computed using
      `time_step` and `num_time_steps`.

  Returns:
    Tuple `(all_times, mask, time_indices)`.
    `all_times` is a 1-D real `Tensor`. If `num_time_steps` is supplied the
      shape of the output is `max(num_time_steps, len(times))`. Otherwise
      consists of all points from 'times` and the uniform grid of points between
      `[0, times[-1]]` with grid size equal to `time_step`.
    `mask` is a boolean 1-D `Tensor` of the same shape as 'all_times', showing
      which elements of 'all_times' correspond to THE values from `times`.
      Guarantees that times[0]=0 and mask[0]=False.
    `time_indices`. An integer `Tensor` of the same shape as `times` indicating
    `times` indices in `all_times`.
  """
  if tolerance is None:
    tolerance = 1e-10 if dtype == tf.float64 else 1e-6
  tolerance = tf.convert_to_tensor(tolerance, dtype=dtype)
  if times_grid is None:
    if num_time_steps is None:
      all_times, time_indices = _grid_from_time_step(
          times=times, time_step=time_step, dtype=dtype, tolerance=tolerance)
    else:
      all_times, time_indices = _grid_from_num_times(
          times=times, time_step=time_step, num_time_steps=num_time_steps)
  else:
    all_times = times_grid
    time_indices = tf.searchsorted(times_grid, times)
    # Adjust indices to bring `times` closer to `times_grid`.
    times_diff_1 = tf.gather(times_grid, time_indices) - times
    times_diff_2 = tf.gather(
        times_grid, tf.math.maximum(time_indices-1, 0)) - times
    time_indices = tf.where(
        tf.math.abs(times_diff_2) > tf.math.abs(times_diff_1),
        time_indices,
        tf.math.maximum(time_indices - 1, 0))
  # Create a boolean mask to identify the iterations that have to be recorded.
  # Use `tf.scatter_nd` because it handles duplicates. Also we first create
  # an int64 Tensor and then create a boolean mask because scatter_nd with
  # booleans is currently not supported on GPUs.
  mask = tf.scatter_nd(
      indices=tf.expand_dims(tf.cast(time_indices, dtype=tf.int64), axis=1),
      updates=tf.fill(tf.shape(times), 1),
      shape=tf.shape(all_times, out_type=tf.int64))
  mask = tf.where(mask > 0, True, False)

  return all_times, mask, time_indices


def _grid_from_time_step(*, times, time_step, dtype, tolerance):
  """Creates a time grid from an input time step."""
  grid = tf.range(0.0, times[-1], time_step, dtype=dtype)
  all_times = tf.concat([times, grid], axis=0)
  all_times = tf.sort(all_times)

  # Remove duplicate points
  dt = all_times[1:] - all_times[:-1]
  dt = tf.concat([[1.0], dt], axis=-1)
  duplicate_mask = tf.math.greater(dt, tolerance)
  all_times = tf.boolean_mask(all_times, duplicate_mask)
  time_indices = tf.searchsorted(all_times, times, out_type=tf.int32)
  time_indices = tf.math.minimum(time_indices, tf.shape(all_times)[0] - 1)

  # Move `time_indices` to the left, if the requested `times` are removed from
  # `all_times` during deduplication
  time_indices = tf.where(
      tf.gather(all_times, time_indices) - times > tolerance,
      time_indices - 1,
      time_indices)

  return all_times, time_indices


def _grid_from_num_times(*, times, time_step, num_time_steps):
  """Creates a time grid for the requeste number of time steps."""
  # Build a uniform grid for the timestep of size
  # max(0, num_time_steps - tf.shape(times)[0])
  uniform_grid = tf.linspace(
      time_step, times[-1] - time_step,
      tf.math.maximum(num_time_steps - tf.shape(times)[0], 0))
  grid = tf.sort(tf.concat([uniform_grid, times], 0))
  # Add zero to the time grid
  all_times = tf.concat([[0], grid], 0)
  time_indices = tf.searchsorted(all_times, times, out_type=tf.int32)
  return all_times, time_indices


def block_diagonal_to_dense(*matrices):
  """Given a sequence of matrices, creates a block-diagonal dense matrix."""
  operators = [tf.linalg.LinearOperatorFullMatrix(m) for m in matrices]
  return tf.linalg.LinearOperatorBlockDiag(operators).to_dense()


def cumsum_using_matvec(input_tensor):
  """Computes cumsum using matrix algebra."""
  dtype = input_tensor.dtype
  axis_length = tf.shape(input_tensor)[-1]
  ones = tf.ones([axis_length, axis_length], dtype=dtype)
  lower_triangular = tf.linalg.band_part(ones, -1, 0)
  cumsum = tf.linalg.matvec(lower_triangular, input_tensor)
  return cumsum


def cumprod_using_matvec(input_tensor):
  """Computes cumprod using matrix algebra."""
  dtype = input_tensor.dtype
  axis_length = tf.shape(input_tensor)[-1]
  ones = tf.ones([axis_length, axis_length], dtype=dtype)
  lower_triangular = tf.linalg.band_part(ones, -1, 0)
  cumsum = tf.linalg.matvec(lower_triangular, tf.math.log(input_tensor))
  return tf.math.exp(cumsum)


def convert_to_tensor_with_default(value, default, dtype=None, name=None):
  """Converts the given `value` to a `Tensor` or returns the `default` value.

  Converts the input `value` to a `Tensor` or returns `default` converted to a
  `Tensor` if `value == None`.

  Args:
    value: An object whose type has a registered Tensor conversion function.
    default: The value to return if `value == None`.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of value.
    name: Optional name to use if a new Tensor is created.

  Returns:
    A Tensor based on value.
  """
  rtn_val = default if value is None else value
  return tf.convert_to_tensor(rtn_val, dtype=dtype, name=name)
