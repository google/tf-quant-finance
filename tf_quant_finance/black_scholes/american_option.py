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
"""Baron-Adesi Whaley approximation of the Black Scholes equation to calculate
 the price of a batch of American options."""


import tensorflow as tf
import tf_quant_finance as tff


def option_price(volatilities,
                 strikes,
                 expiries,
                 spots=None,
                 forwards=None,
                 risk_free_rates=None,
                 cost_of_carries=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
  """Computes the price for a batch of call or put options, using an approximate
  pricing formula, the Baron-Adesi Whaley approximation

  #### Example

  ```python
  forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
  volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
  expiries = 1.0
  computed_prices = option_price(
      volatilities,
      strikes,
      expiries,
      forwards=forwards,
      dtype=tf.float64)
  # Expected print output of computed prices:
  # [ 0.          2.          2.04806848  1.00020297  2.07303131]
  ```

  ## References:
  [1] Baron-Adesi, Whaley, Efficient Analytic Approximation of American Option
    Values, The Journal of Finance, Vol XLII, No. 2, June 1987

  Args:
  volatilities: Real `Tensor` of any shape and dtype. The volatilities to
    expiry of the options to price.
  strikes: A real `Tensor` of the same dtype and compatible shape as
    `volatilities`. The strikes of the options to be priced.
  expiries: A real `Tensor` of same dtype and compatible shape as
    `volatilities`. The expiry of each option. The units should be such that
    `expiry * volatility**2` is dimensionless.
  spots: A real `Tensor` of any shape that broadcasts to the shape of the
    `volatilities`. The current spot price of the underlying. Either this
    argument or the `forwards` (but not both) must be supplied.
  forwards: A real `Tensor` of any shape that broadcasts to the shape of
    `volatilities`. The forwards to maturity. Either this argument or the
    `spots` must be supplied but both must not be supplied.
  risk_free_rates: An optional real `Tensor` of same dtype as the
    `volatilities`. If not None, discount factors are calculated using it:
    e^(-rT), where r is the risk free rate. If not None, `spots` is supplied
    and `cost_of_carries` is not, also used to compute forwards to expiry.
    Default value: None, equivalent to r = 0 and discount factors = 1 when
    discount_factors also not given.
  cost_of_carries: An optional real `Tensor` of same dtype as the
    `volatilities`. If not None, and `spots` is supplied, used to calculate
    forwards from spots: F = e^(bT) * S. If None, value assumed
    to be equal to the risk free rate.
    Default value: None, equivalent to b = r.
  is_call_options: A boolean `Tensor` of a shape compatible with
    `volatilities`. Indicates whether the option is a call (if True) or a put
    (if False). If not supplied, call options are assumed.
  dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
    of any supplied non-`Tensor` arguments to `Tensor`.
    Default value: None which maps to the default dtype inferred by TensorFlow
      (float32).
  name: str. The name for the ops created by this function.
    Default value: None which is mapped to the default name `option_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
    ValueError: If both `risk_free_rates` and `discount_factors` is supplied.
    ValueError: If `cost_of_carries` is supplied without `discount_factors`.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')
  if cost_of_carries and not risk_free_rates:
    raise ValueError('cost_of_carries may only be supplied alongside the '
                     'risk_free_rates')

  