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
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.models import longstaff_schwartz


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
    is_call_option: A bool `Tensor` of the same shape as `spots` specifying
      which options are put/call.
      Default value: `True` which means all options are call.
    num_calibration_samples: An optional integer less or equal to `num_samples`.
      The number of sampled trajectories used for the LSM regression step.
      Default value: `None`, which means that all samples are used for
        regression.
    dtype: `tf.Dtype` of the input and output real `Tensor`s.
      Default value: `None` which maps to `float64`.
    name: Python str. The name to give to the ops created by this class.
      Default value: `None` which maps to 'bs_lsm_price'.
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
    call_put = tf.convert_to_tensor(is_call_option, dtype=tf.bool,
                                    name="discount_factors")
    call_put = tf.expand_dims(call_put, axis=-1)
    risk_free_rate = -tf.math.log(discount_factors) / expiry_times
    # Normalize expiry times
    scaled_expiries = volatility * tf.math.sqrt(expiry_times)
    expiry_times_reshape = tf.reshape(expiry_times, [-1, 1, 1, 1])
    gbm = models.GeometricBrownianMotion(
        mu=0.0, sigma=1.0,
        dtype=dtype)
    times = tf.linspace(tf.constant(0.0, dtype), 1.0, num_exercise_times)
    # Samples from Geometric Brownian motion to be used for all options
    # Distributed as exp(-1/2 * times + W(times))
    samples = gbm.sample_paths(
        times,
        initial_state=1.0,
        num_samples=num_samples,
        seed=seed,
        random_type=math.random.RandomType.STATELESS_ANTITHETIC)
    # Rescale samples
    # Shape [batch_size, 1, 1, 1]
    volatility_expand = tf.reshape(volatility, [-1, 1, 1, 1])
    # Shape [1, 1, num_exercise_times, 1]
    times_expand = tf.reshape(times, [1, 1, -1, 1])
    # Shape [1, num_samples, num_exercise_times, dim]
    # Distributed as exp(W(times))
    samples = tf.math.exp(0.5 * times_expand) * samples
    # Shape [batch_size, num_samples, num_exercise_times, dim]
    # Distributed as exp(volatility * W(t))
    samples = samples**tf.reshape(scaled_expiries, [-1, 1, 1, 1])
    # Distributed as exp(-volatility**2 / 2 * t + volatility * W(t))
    samples *= tf.math.exp(-times_expand * expiry_times_reshape
                           * volatility_expand**2 / 2)
    # Shape [batch_size, 1, 1, 1]
    spots = tf.reshape(spots, [-1, 1, 1, 1])
    # Shape [batch_size, 1, num_exercise_times, 1]
    rates_exp = tf.math.exp(tf.reshape(risk_free_rate, [-1, 1, 1, 1])
                            * times_expand * expiry_times_reshape)
    # Shape [batch_size, num_samples, num_exercise_times, dim]
    # Distributed as spot * exp((r -volatility**2 / 2) * t + volatility * W(t))
    samples = spots * rates_exp * samples
    # Payoff function takes all the samples of shape
    # [batch_size, num_paths, num_times, dim] and returns a `Tensor` of
    # shape [num_paths, batch_size]. This corresponds to a
    # payoff at the present time.
    def _payoff_fn(sample_paths, time_index):
      current_samples = tf.squeeze(sample_paths, axis=-1)
      current_samples = tf.transpose(current_samples, [2, 0, 1])[time_index]
      # Shape [batch_size, num_samples]
      payoff = tf.expand_dims(strikes, -1) - current_samples
      payoff = tf.where(call_put, tf.nn.relu(-payoff), tf.nn.relu(payoff))
      return tf.transpose(payoff)

    if basis_fn is None:
      # Polynomial basis with 2 functions
      basis_fn = longstaff_schwartz.make_polynomial_basis(2)

    # Set up Longstaff-Schwartz algorithm
    def lsm_price(sample_paths):
      exercise_times = tf.range(tf.shape(times)[0])
      # This is Longstaff-Schwartz algorithm
      return longstaff_schwartz.least_square_mc(
          sample_paths=sample_paths,
          exercise_times=exercise_times,
          payoff_fn=_payoff_fn,
          basis_fn=basis_fn,
          discount_factors=tf.math.exp(
              -tf.reshape(risk_free_rate, [1, -1, 1]) * times
              * tf.reshape(expiry_times, [1, -1, 1])),
          num_calibration_samples=num_calibration_samples)
    return lsm_price(samples)
