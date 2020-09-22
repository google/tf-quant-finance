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
"""Utilities for equity American options."""

from typing import Tuple

import tensorflow.compat.v2 as tf

from tf_quant_finance import math
from tf_quant_finance import models
from tf_quant_finance.experimental import lsm_algorithm
from tf_quant_finance.experimental.pricing_platform.framework.core import types


def bs_lsm_price(
    spots: types.FloatTensor,
    expiry_times: types.FloatTensor,
    strikes: types.FloatTensor,
    volatility: types.FloatTensor,
    discount_factors: types.FloatTensor,
    num_samples: int = 100000,
    num_exercise_times: int = 100,
    basis_fn=None,
    seed: Tuple[int, int] = (1, 2),
    is_call_option: types.BoolTensor = True,
    num_calibration_samples: int = None,
    dtype: types.Dtype = None,
    name: str = None):
  """Computes American option price via LSM under Black-Scholes model.

  Args:
    spots: A rank 1 real `Tensor` with spot prices.
    expiry_times: A `Tensor` of the same shape and dtype as `spots` representing
      expiry values of the options.
    strikes: A `Tensor` of the same shape and dtype as `spots` representing
      strike price of the options.
    volatility: A `Tensor` of the same shape and dtype as `spots` representing
      volatility values of the options.
    discount_factors: A `Tensor` of the same shape and dtype as `spots`
      representing discount factors at the expiry times.
    num_samples: Number of Monte Carlo samples.
    num_exercise_times: Number of excercise times for American options.
    basis_fn: Callable from a `Tensor` of the same shape
      `[num_samples, num_exercice_times, 1]` (corresponding to Monte Carlo
      samples) and a positive integer `Tenor` (representing a current
      time index) to a `Tensor` of shape `[basis_size, num_samples]` of the same
      dtype as `spots`. The result being the design matrix used in
      regression of the continuation value of options.
      This is the same argument as in `lsm_algorithm.least_square_mc`.
    seed: A tuple of 2 integers setting global and local seed of the Monte Carlo
      sampler
    is_call_option: A bool `Tensor`.
    num_calibration_samples: An optional integer less or equal to `num_samples`.
      The number of sampled trajectories used for the LSM regression step.
      Default value: `None`, which means that all samples are used for
        regression.
    dtype: `tf.Dtype` of the input and output real `Tensor`s.
      Default value: `None` which maps to `float64`.
    name: Python str. The name to give to the ops created by this class.
      Default value: `None` which maps to 'forward_rate_agreement'.
  Returns:
    A `Tensor` of the same shape and dtyoe as `spots` representing american
    option prices.
  """
  dtype = dtype or tf.float64
  name = name or "bs_lsm_price"
  with tf.name_scope(name):

    strikes = tf.convert_to_tensor(strikes, dtype=dtype,
                                   name="strikes")
    spots = tf.convert_to_tensor(spots, dtype=dtype,
                                 name="spots")
    volatility = tf.convert_to_tensor(volatility, dtype=dtype,
                                      name="volatility")
    expiry_times = tf.convert_to_tensor(expiry_times, dtype=dtype,
                                        name="expiry_times")
    discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype,
                                            name="discount_factors")
    risk_free_rate = -tf.math.log(discount_factors) / expiry_times
    # Normalize expiry times
    var = volatility**2
    expiry_times = expiry_times * var

    gbm = models.GeometricBrownianMotion(
        mu=0.0, sigma=1.0,
        dtype=dtype)
    max_time = tf.reduce_max(expiry_times)

    # Get a grid of 100 exercise times + all expiry times
    unique_expiry = tf.unique(expiry_times).y
    times = tf.sort(tf.concat([tf.linspace(tf.constant(0.0, dtype),
                                           max_time,
                                           num_exercise_times),
                               unique_expiry], axis=0))
    # Samples for all options
    samples = gbm.sample_paths(
        times,
        initial_state=1.0,
        num_samples=num_samples,
        seed=seed,
        random_type=math.random.RandomType.STATELESS_ANTITHETIC)
    indices = tf.searchsorted(times, expiry_times)
    indices_ext = tf.expand_dims(indices, axis=-1)
    # Payoff function takes all the samples of shape
    # [num_paths, num_times, dim] and returns a `Tensor` of
    # shape [num_paths, num_strikes]. This corresponds to a
    # payoff at the present time.
    def _payoff_fn(sample_paths, time_index):
      current_samples = tf.transpose(sample_paths, [1, 2, 0])[time_index]
      r = tf.math.exp(tf.expand_dims(risk_free_rate / var, axis=-1)
                      * times[time_index])
      s = tf.expand_dims(spots, axis=-1)
      call_put = tf.expand_dims(is_call_option, axis=-1)
      payoff = tf.expand_dims(strikes, -1) - r * s * current_samples
      payoff = tf.where(call_put, tf.nn.relu(-payoff), tf.nn.relu(payoff))
      # Since the pricing is happening on the grid,
      # For options, which have already expired, the payoff is set to `0`
      # to indicate that one should not exercise the option after it has expired
      res = tf.where(time_index > indices_ext,
                     tf.constant(0, dtype=dtype),
                     payoff)
      return tf.transpose(res)

    if basis_fn is None:
      # Polynomial basis with 2 functions
      basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)

    # Set up Longstaff-Schwartz algorithm
    def lsm_price(sample_paths):
      num_times = int(times.shape[0])
      # This is Longstaff-Schwartz algorithm
      return lsm_algorithm.least_square_mc_v2(
          sample_paths=sample_paths,
          exercise_times=list(range(num_times)),
          payoff_fn=_payoff_fn,
          basis_fn=basis_fn,
          discount_factors=tf.math.exp(
              -tf.reshape(risk_free_rate / var, [1, -1, 1]) * times),
          num_calibration_samples=num_calibration_samples)
    return lsm_price(samples)
