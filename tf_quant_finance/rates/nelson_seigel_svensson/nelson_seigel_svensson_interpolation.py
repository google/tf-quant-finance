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
"""Nelson Seigel Svensson interpolation method."""

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils

__all__ = ['interpolate', 'SvenssonParameters']


@tff_utils.dataclass
class SvenssonParameters:
  """Nelson Seigel Svensson interpolation parameters.

  Attributes:
    beta_0: A real `Tensor` of arbitrary shape `batch_shape`.
    beta_1: A real `Tensor` of arbitrary shape `batch_shape`.
    beta_2: A real `Tensor` of arbitrary shape `batch_shape`.
    beta_3: A real `Tensor` of arbitrary shape `batch_shape`.
    tau_1: A real `Tensor` of arbitrary shape `batch_shape`.
    tau_2: A real `Tensor` of arbitrary shape `batch_shape`.
  """
  beta_0: types.RealTensor
  beta_1: types.RealTensor
  beta_2: types.RealTensor
  beta_3: types.RealTensor
  tau_1: types.RealTensor
  tau_2: types.RealTensor


def interpolate(interpolation_times: types.RealTensor,
                svensson_parameters: SvenssonParameters,
                dtype: tf.DType = None,
                name: str = None) -> types.RealTensor:
  """Performs Nelson Seigel Svensson interpolation for supplied points.

  Given a set of interpolation times and the parameters for the nelson seigel
  svensson model, this function returns the interpolated values for the yield
  curve. We assume that the parameters are already computed using a fitting
  technique.
  ```None
  r(T) = beta_0 +
         beta_1 * (1-exp(-T/tau_1))/(T/tau_1) +
         beta_2 * ((1-exp(-T/tau_1))/(T/tau_1) - exp(-T/tau_1)) +
         beta_3 * ((1-exp(-T/tau_2))/(T/tau_2) - exp_(-T/tau_2))
  ```

  Where `T` represents interpolation times and
  `beta_i`'s and `tau_i`'s are paramters for the model.

  #### Example
  ```python
  import tf_quant_finance as tff
  interpolation_times = [5., 10., 15., 20.]
  svensson_parameters =
  tff.rates.nelson_svensson.interpolate.SvenssonParameters(
        beta_0=0.05, beta_1=-0.01, beta_2=0.3, beta_3=0.02,
        tau_1=1.5, tau_2=20.0)
  result = interpolate(interpolation_times, svensson_parameters)
  # Expected_result
  # [0.12531, 0.09667, 0.08361, 0.07703]
  ```

  #### References:
    [1]: Robert Müller. A technical note on the Svensson model as applied to
    the Swiss term structure.
    BIS Papers No 25, Mar 2015.
    https://www.bis.org/publ/bppdf/bispap25l.pdf

  Args:
    interpolation_times: The times at which interpolation is desired. A N-D
      `Tensor` of real dtype where the first N-1 dimensions represent the
      batching dimensions.
    svensson_parameters: An instance of `SvenssonParameters`. All parameters
      within should be real tensors.
    dtype: Optional tf.dtype for `interpolation_times`. If not specified, the
      dtype of the inputs will be used.
    name: Python str. The name prefixed to the ops created by this function. If
      not supplied, the default name 'nelson_svensson_interpolation' is used.

  Returns:
    A N-D `Tensor` of real dtype with the same shape as `interpolations_times`
      containing the interpolated yields.
  """
  name = name or 'nelson_svensson_interpolation'
  with tf.compat.v1.name_scope(name):
    interpolation_times = tf.convert_to_tensor(interpolation_times, dtype=dtype)
    dtype = dtype or interpolation_times.dtype

    yield_part0 = svensson_parameters.beta_0
    yield_part1 = svensson_parameters.beta_1 * (
        _integrated_exp_term(interpolation_times, svensson_parameters.tau_1))
    yield_part2 = svensson_parameters.beta_2 * (
        _integrated_exp_term(interpolation_times, svensson_parameters.tau_1) -
        _exp_term(interpolation_times, svensson_parameters.tau_1))
    yield_part3 = svensson_parameters.beta_3 * (
        _integrated_exp_term(interpolation_times, svensson_parameters.tau_2) -
        _exp_term(interpolation_times, svensson_parameters.tau_2))

    interpolated_yields = yield_part0 + yield_part1 + yield_part2 + yield_part3

  return interpolated_yields


def _exp_term(x, y):
  return tf.math.exp(-tf.math.divide_no_nan(x, y))


def _integrated_exp_term(x, y):
  return (1 - _exp_term(x, y)) / (tf.math.divide_no_nan(x, y))
