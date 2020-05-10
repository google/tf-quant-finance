# Lint as: python3
# Copyright 2020 Google LLC
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

"""Methods for brownian bridges.

These can be used in Monte-Carlo simulation for payoff with continuous barrier.
Indeed, the Monte-Carlo simulation is inherently discrete in time, and to
improve convergence (w.r.t. the number of time steps) for payoff with continuous
barrier, adjustment with brownian bridge can be made.

## References

[1] Emmanuel Gobet. Advanced Monte Carlo methods for barrier and related
exotic options.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669
"""

import tensorflow.compat.v2 as tf


def brownian_bridge_double(*,
                           x_start,
                           x_end,
                           variance,
                           upper_barrier,
                           lower_barrier,
                           n_cutoff=3,
                           dtype=None,
                           name=None):
  """Computes probability of not touching the barriers for a 1D Brownian Bridge.

  The Brownian bridge starts at `x_start`, ends at `x_end` and has a variance
  `variance`. The no-touch probabilities are calculated assuming that `x_start`
  and `x_end` are within the barriers 'lower_barrier' and 'upper_barrier'.
  This can be used in Monte Carlo pricing for adjusting probability of
  touching the barriers from discrete case to continuous case.
  Typically in practice, the tensors `x_start`, `x_end` and `variance` should be
  of rank 2 (with time steps and paths being the 2 dimensions).

  #### Example

  ```python
  x_start = np.asarray([[4.5, 4.5, 4.5], [4.5, 4.6, 4.7]])
  x_end = np.asarray([[5.0, 4.9, 4.8], [4.8, 4.9, 5.0]])
  variance = np.asarray([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
  upper_barrier = 5.1
  lower_barrier = 4.4

  no_touch_proba = brownian_bridge_double(
    x_start=x_start,
    x_end=x_end,
    variance=variance,
    upper_barrier=upper_barrier,
    lower_barrier=lower_barrier,
    n_cutoff=3,
    )
  # Expected print output of no_touch_proba:
  #[[0.45842169 0.21510919 0.52704599]
  #[0.09394963 0.73302813 0.22595022]]
  ```

  #### References

  [1] Emmanuel Gobet. Advanced Monte Carlo methods for barrier and related
  exotic options.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669

  Args:
    x_start: A real `Tensor` of any shape and dtype.
    x_end: A real `Tensor` of the same dtype and compatible shape as
      `x_start`.
    variance: A real `Tensor` of the same dtype and compatible shape as
      `x_start`.
    upper_barrier: A scalar `Tensor` of the same dtype as `x_start`. Stands for
      the upper boundary for the Brownian Bridge.
    lower_barrier: A scalar `Tensor` of the same dtype as `x_start`. Stands for
      lower the boundary for the Brownian Bridge.
    n_cutoff: A positive scalar int32 `Tensor`. This controls when to cutoff
      the sum which would otherwise have an infinite number of terms.
      Default value: 3.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name
      `brownian_bridge_double`.

  Returns:
      A `Tensor` of the same shape as the input data which is the probability
      of not touching the upper and lower barrier.
  """
  with tf.name_scope(name or 'brownian_bridge_double'):
    x_start = tf.convert_to_tensor(x_start, dtype=dtype, name='x_start')
    dtype = x_start.dtype
    variance = tf.convert_to_tensor(variance, dtype=dtype, name='variance')
    x_end = tf.convert_to_tensor(x_end, dtype=dtype, name='x_end')
    barrier_diff = upper_barrier - lower_barrier

    x_start = tf.expand_dims(x_start, -1)
    x_end = tf.expand_dims(x_end, -1)
    variance = tf.expand_dims(variance, -1)

    k = tf.expand_dims(tf.range(-n_cutoff, n_cutoff + 1, dtype=dtype), 0)

    a = k * barrier_diff * (k * barrier_diff + (x_end - x_start))
    b = (k * barrier_diff + x_start - upper_barrier)
    b *= k * barrier_diff + (x_end - upper_barrier)

    # TODO(b/152731702): replace with a numericall stable procedure.
    output = tf.math.exp(- 2 * a / variance) - tf.math.exp(-2 * b / variance)

    return tf.reduce_sum(output, axis=-1)


def brownian_bridge_single(*,
                           x_start,
                           x_end,
                           variance,
                           barrier,
                           dtype=None,
                           name=None):
  """Computes proba of not touching the barrier for a 1D Brownian Bridge.

  The Brownian bridge starts at `x_start`, ends at `x_end` and has a variance
  `variance`. The no-touch probabilities are calculated assuming that `x_start`
  and `x_end` are the same side of the barrier (either both above or both
  below).
  This can be used in Monte Carlo pricing for adjusting probability of
  touching the barrier from discrete case to continuous case.
  Typically in practise, the tensors `x_start`, `x_end` and `variance` should be
  bi-dimensional (with time steps and paths being the 2 dimensions).

  #### Example

  ```python
  x_start = np.asarray([[4.5, 4.5, 4.5], [4.5, 4.6, 4.7]])
  x_end = np.asarray([[5.0, 4.9, 4.8], [4.8, 4.9, 5.0]])
  variance = np.asarray([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
  barrier = 5.1

  no_touch_proba = brownian_bridge_single(
    x_start=x_start,
    x_end=x_end,
    variance=variance,
    barrier=barrier)
  # Expected print output of no_touch_proba:
  # [[0.69880579 0.69880579 0.97267628]
  #  [0.69880579 0.86466472 0.32967995]]
  ```

  #### References

  [1] Emmanuel Gobet. Advanced Monte Carlo methods for barrier and related
  exotic options.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669

  Args:
    x_start: A real `Tensor` of any shape and dtype.
    x_end: A real `Tensor` of the same dtype and compatible shape as
      `x_start`.
    variance: A real `Tensor` of the same dtype and compatible shape as
      `x_start`.
    barrier: A scalar `Tensor` of the same dtype as `x_start`. Stands for the
      boundary for the Brownian Bridge.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by
      TensorFlow.
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name
      `brownian_bridge_single`.

  Returns:
      A `Tensor` of the same shape as the input data which is the probability
      of not touching the barrier.
  """
  with tf.name_scope(name or 'brownian_bridge_single'):
    x_start = tf.convert_to_tensor(x_start, dtype=dtype, name='x_start')
    dtype = x_start.dtype
    variance = tf.convert_to_tensor(variance, dtype=dtype, name='variance')
    x_end = tf.convert_to_tensor(x_end, dtype=dtype, name='x_end')

    a = (x_start - barrier) * (x_end - barrier)
    return 1 - tf.math.exp(-2 * a / variance)

