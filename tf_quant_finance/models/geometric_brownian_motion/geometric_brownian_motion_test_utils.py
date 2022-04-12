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

"""Utility functions for the Geometric Brownian Motion models."""

import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff


def convert_to_ndarray(test_obj, a):
  """Converts the input `a` into an ndarray.

  Args:
    test_obj: An object which has the `evaluate` method. Used to evaluate `a` if
      `a` is a Tensor.
    a: Object to be converted to an ndarray.

  Returns:
    An ndarray containing the values of `a`.
  """
  # TODO(b/177990397): This function should be independent of the test framework
  # If a is tensor-like then convert it to ndarray
  if tf.is_tensor(a):
    a = test_obj.evaluate(a)
  if not isinstance(a, np.ndarray):
    return np.array(a)
  return a


def arrays_all_close(test_obj, a, b, atol, msg=None):
  """Check if two arrays are within a tolerance specified per element.

  Checks that a, b and atol have the same shape and that
    `abs(a_i - b_i) <= atol_i` for all elements of `a` and `b`.
  This function differs from np.testing.assert_allclose() as
  np.testing.assert_allclose() applies the same `atol` to all of the elements,
  whereas this function takes a `ndarray` specifying a `atol` for each element.

  Args:
    test_obj: An object which has the `evaluate` method. Used to evaluate `a` if
      `a` is a Tensor.
    a: The expected numpy `ndarray`, or anything that can be converted into a
       numpy `ndarray` (including Tensor), or any arbitrarily nested of
       structure of these.
    b: The actual numpy `ndarray`, or anything that can be converted into a
       numpy `ndarray` (including Tensor), or any arbitrarily nested of
       structure of these.
    atol: absolute tolerance as a numpy `ndarray` of the same shape as `a` and
       `b`.
    msg: Optional message to include in the error message.
  Raises:
    ValueError: If `a`, `b` and `atol` do not have the same shape.
    AssertionError: If any of the elements are outside the tolerance.
  """
  # TODO(b/177990397): This function should be independent of the test framework
  a = convert_to_ndarray(test_obj, a)
  b = convert_to_ndarray(test_obj, b)
  atol = convert_to_ndarray(test_obj, atol)
  # Check the shapes are the same.
  if a.shape != b.shape:
    raise ValueError("Mismatched shapes a.shape() = {}".format(a.shape) +
                     ", b.shape= {}".format(b.shape) +
                     ", atol.shape = {}".format(atol.shape) +
                     ". ({}).".format(msg))
  abs_diff = np.abs(a - b)
  if np.any(abs_diff >= atol):
    raise ValueError("Expected and actual values differ by more than the " +
                     "tolerance.\n a = {}".format(a) +
                     "\n b = {}".format(b) +
                     "\n abs_diff = {}".format(abs_diff) +
                     "\n atol = {}".format(atol) +
                     "\n When {}.".format(msg))
  return


def generate_sample_paths(mu, sigma, times, initial_state, supply_draws,
                          num_samples, dtype):
  """Returns the sample paths for the process with the given parameters.

  Args:
    mu: Scalar real `Tensor` broadcastable to [`batch_shape`, 1] or an instance
      of left-continuous `PiecewiseConstantFunc` of [`batch_shape`]
      dimensions. Where `batch_shape` is the larger of `mu.shape` and
      `sigma.shape`. Corresponds to the mean drift of the Ito process.
    sigma: Scalar real `Tensor` broadcastable to [`batch_shape`, 1] or an
      instance of left-continuous `PiecewiseConstantFunc` of the same `dtype`
      and `batch_shape` as set by `mu`. Where `batch_shape` is the larger of
      `mu.shape` and `sigma.shape`. Corresponds to the volatility of the
      process and should be positive.
    times: A `Tensor` of positive real values of a shape [`T`, `num_times`],
      where `T` is either empty or a shape which is broadcastable to
      `batch_shape` (as defined by the shape of `mu` or `sigma`. The times at
      which the path points are to be evaluated.
    initial_state: A `Tensor` of the same `dtype` as `times` and of shape
      broadcastable to `[batch_shape, num_samples]`. Represents the initial
      state of the Ito process.
    supply_draws: Boolean set to true if the `normal_draws` should be generated
      and then passed to the pricing function.
    num_samples: Positive scalar `int`. The number of paths to draw.
    dtype: The default dtype to use when converting values to `Tensor`s.

  Returns:
    A Tensor containing the the sample paths of shape
    [batch_shape, num_samples, num_times, 1].
  """
  process = tff.models.GeometricBrownianMotion(mu, sigma, dtype=dtype)
  normal_draws = None
  if supply_draws:
    total_dimension = tf.zeros(times.shape[-1], dtype=dtype)
    # Shape [num_samples, times.shape[-1]]
    normal_draws = tff.math.random.mv_normal_sample(
        [num_samples], mean=total_dimension,
        random_type=tff.math.random.RandomType.SOBOL,
        seed=[4, 2])
    # Shape [num_samples, times.shape[-1], 1]
    normal_draws = tf.expand_dims(normal_draws, axis=-1)
  return process.sample_paths(
      times=times,
      initial_state=initial_state,
      random_type=tff.math.random.RandomType.STATELESS,
      num_samples=num_samples,
      normal_draws=normal_draws,
      seed=[1234, 5])


def calculate_mean_and_variance_from_sample_paths(samples, num_samples, dtype):
  """Returns the mean and variance of log(`samples`).

  Args:
    samples: A real `Tensor` of shape [batch_shape, `num_samples`, num_times, 1]
      containing the samples of random paths drawn from an Ito process.
    num_samples: A scalar integer. The number of sample paths in `samples`.
    dtype: The default dtype to use when converting values to `Tensor`s.

  Returns:
    A tuple of (mean, variance, standard_error of the mean,
    standard_error of the variance) of the log of the samples.  Where the
    components of the tuple have shape [batch_shape, num_times].
  """
  log_s = tf.math.log(samples)
  mean = tf.reduce_mean(log_s, axis=-3, keepdims=True)
  var = tf.reduce_mean((log_s - mean)**2, axis=-3, keepdims=True)
  mean = tf.squeeze(mean, axis=[-1, -3])
  var = tf.squeeze(var, axis=[-1, -3])

  # Standard error of the mean formula taken from
  # https://en.wikipedia.org/wiki/Standard_error.
  std_err_mean = tf.math.sqrt(var / num_samples)
  # Standard error of the sample variance (\sigma_{S^2}) is given by:
  # \sigma_{S^2} = S^2.\sqrt(2 / (n-1)).
  # Taken from 'Standard Errors of Mean, Variance, and Standard Deviation
  # Estimators' Ahn, Fessler 2003.
  # (https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf)
  std_err_var = var * tf.math.sqrt(
      tf.constant(2.0, dtype=dtype) /
      (tf.constant(num_samples, dtype=dtype) - tf.constant(1.0, dtype=dtype)))
  return (mean, var, std_err_mean, std_err_var)


def calculate_sample_paths_mean_and_variance(test_obj, mu, sigma, times,
                                             initial_state, supply_draws,
                                             num_samples, dtype):
  """Returns the mean and variance of the log of the sample paths for a process.

  Generates a set of sample paths for a univariate geometric brownian motion
  and calculates the mean and variance of the log of the paths. Also returns the
  standard error of the mean and variance.

  Args:
    test_obj: An object which has the `evaluate` method. Used to evaluate `a` if
      `a` is a Tensor.
    mu: Scalar real `Tensor` broadcastable to [`batch_shape`] or an instance
      of left-continuous `PiecewiseConstantFunc` of [`batch_shape`]
      dimensions. Where `batch_shape` is the larger of `mu.shape` and
      `sigma.shape`. Corresponds to the mean drift of the Ito process.
    sigma: Scalar real `Tensor` broadcastable to [`batch_shape`] or an
      instance of left-continuous `PiecewiseConstantFunc` of the same `dtype`
      and `batch_shape` as set by `mu`. Where `batch_shape` is the larger of
      `mu.shape` and `sigma.shape`. Corresponds to the volatility of the
      process and should be positive.
    times: Rank 1 `Tensor` of positive real values. The times at which the
      path points are to be evaluated.
    initial_state: A `Tensor` of the same `dtype` as `times` and of shape
      broadcastable to `[batch_shape, num_samples]`. Represents the initial
      state of the Ito process.
    supply_draws: Boolean set to true if the `normal_draws` should be generated
      and then passed to the pricing function.
    num_samples: Positive scalar `int`. The number of paths to draw.
    dtype: The default dtype to use when converting values to `Tensor`s.

  Returns:
    A tuple of (mean, variance, standard_error of the mean,
      standard_error of the variance).
  """
  # TODO(b/177990397): This function should be independent of the test framework
  samples = generate_sample_paths(mu, sigma, times, initial_state, supply_draws,
                                  num_samples, dtype)
  return test_obj.evaluate(
      calculate_mean_and_variance_from_sample_paths(samples, num_samples,
                                                    dtype))
