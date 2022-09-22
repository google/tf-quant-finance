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

"""Algorithms for numeric integration in TensorFlow."""


import enum
from typing import Callable, Optional
import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.math.integration.gauss_legendre import gauss_legendre
from tf_quant_finance.math.integration.simpson import simpson


@enum.unique
class IntegrationMethod(enum.Enum):
  """Specifies which algorithm to use for the numeric integration.

  * `COMPOSITE_SIMPSONS_RULE`: Composite Simpson's 1/3 rule.
  * `GAUSS_LEGENDRE`: Gauss-Legendre quadrature method.
  """
  COMPOSITE_SIMPSONS_RULE = 1
  GAUSS_LEGENDRE = 2


def integrate(
    func: Callable[[types.FloatTensor], types.FloatTensor],
    lower: types.FloatTensor,
    upper: types.FloatTensor,
    method: IntegrationMethod = IntegrationMethod.COMPOSITE_SIMPSONS_RULE,
    dtype: Optional[tf.DType] = None,
    name: Optional[str] = None,
    **kwargs) -> types.FloatTensor:
  """Evaluates definite integral.

  #### Example
  ```python
    f = lambda x: x*x
    a = tf.constant(0.0)
    b = tf.constant(3.0)
    integrate(f, a, b) # 9.0
  ```

  Args:
    func: Represents a function to be integrated. It must be a callable of a
      single `Tensor` parameter and return a `Tensor` of the same shape and
      dtype as its input. It will be called with a `Tesnor` of shape
      `lower.shape + [n]` (where n is integer number of points) and of the same
      `dtype` as `lower`.
    lower: Represents the lower limits of integration. `func` will be integrated
      between each pair of points defined by `lower` and `upper`.
    upper: Same shape and dtype as `lower` representing the upper limits of
      intergation.
    method: Integration method. Instance of IntegrationMethod enum. Default is
      IntegrationMethod.COMPOSITE_SIMPSONS_RULE.
    dtype: Dtype of result. Must be real dtype. Defaults to dtype of `lower`.
    name: The name to give to the ops created by this function.
      Default value: None which maps to 'integrate'.
    **kwargs: Additional parameters for specific integration method.

  Returns:
    `Tensor` of the same shape and dtype as `lower`, containing the value of the
    definite integral.

  Raises: ValueError if `method` was not recognized.
  """
  with tf.compat.v1.name_scope(
      name, default_name='integrate', values=[lower, upper]):
    if method == IntegrationMethod.COMPOSITE_SIMPSONS_RULE:
      return simpson(func, lower, upper, dtype=dtype, **kwargs)
    elif method == IntegrationMethod.GAUSS_LEGENDRE:
      return gauss_legendre(func, lower, upper, dtype=dtype, **kwargs)
    else:
      raise ValueError('Unknown method: %s.' % method)
