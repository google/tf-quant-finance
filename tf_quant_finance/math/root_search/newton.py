# Lint as: python3
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
"""Root finder functions using newton method."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.math.root_search import utils


# TODO(b/179451420): Refactor BrentResults as RootSearchResults and return it
# for newton method as well.
def root_finder(value_and_grad_func,
                initial_values,
                max_iterations=20,
                tolerance=2e-7,
                relative_tolerance=None,
                dtype=None,
                name='root_finder'):
  """Finds roots of a scalar function using Newton's method.

  This method uses Newton's algorithm to find values `x` such that `f(x)=0` for
  some real-valued differentiable function `f`. Given an initial value `x_0` the
  values are iteratively updated as:

    `x_{n+1} = x_n - f(x_n) / f'(x_n),`

  for further details on Newton's method, see [1]. The implementation accepts
  array-like arguments and assumes that each cell corresponds to an independent
  scalar model.

  #### Examples
  ```python
  # Set up the problem of finding the square roots of three numbers.
  constants = np.array([4.0, 9.0, 16.0])
  initial_values = np.ones(len(constants))
  def objective_and_gradient(values):
    objective = values**2 - constants
    gradient = 2.0 * values
    return objective, gradient

  # Obtain and evaluate a tensor containing the roots.
  roots = tff.math.root_search.newton_root(objective_and_gradient,
    initial_values)
  print(root_values)  # Expected output: [ 2.  3.  4.]
  print(converged)  # Expected output: [ True  True  True]
  print(failed)  # Expected output: [False False False]
  ```

  #### References
  [1] Luenberger, D.G., 1984. 'Linear and Nonlinear Programming'. Reading, MA:
  Addison-Wesley.

  Args:
    value_and_grad_func: A python callable that takes a `Tensor` of the same
      shape and dtype as the `initial_values` and which returns a two-`tuple` of
      `Tensors`, namely the objective function and the gradient evaluated at the
      passed parameters.
    initial_values: A real `Tensor` of any shape. The initial values of the
      parameters to use (`x_0` in the notation above).
    max_iterations: positive `int`. The maximum number of
      iterations of Newton's method.
      Default value: 20.
    tolerance: positive scalar `Tensor`. The absolute tolerance for the root
      search. Search is judged to have converged  if
      `|f(x_n) - f(x_n-1)|` < |x_n| * `relative_tolerance` + `tolerance`
      (using the notation above), or if `x_n` becomes `nan`. When an element is
      judged to have converged it will no longer be updated. If all elements
      converge before `max_iterations` is reached then the root finder will
      return early. If None, it would be set according to the `dtype`,
      which is 4 * np.finfo(dtype.as_numpy_dtype(0)).eps.
      Default value: 2e-7.
    relative_tolerance: positive `double`, default 0. See the document for
      `tolerance`.
      Default value: None.
    dtype: optional `tf.DType`. If supplied the `initial_values` will be coerced
      to this data type.
      Default value: None.
    name: `str`, to be prefixed to the name of
      TensorFlow ops created by this function.
      Default value: 'root_finder'.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `initial_values`. It
    contains the found roots (same dtype as `initial_values`), a boolean
    `Tensor` indicating whether the corresponding root results in an objective
    function value less than the tolerance, and a boolean `Tensor` which is true
    where the corresponding 'root' is not finite.
  """
  if tolerance is None:
    tolerance = utils.default_relative_root_tolerance(dtype)
  if relative_tolerance is None:
    relative_tolerance = utils.default_relative_root_tolerance(dtype)

  with tf.compat.v1.name_scope(
      name,
      default_name='newton_root_finder',
      values=[initial_values, tolerance]):

    initial_values = tf.convert_to_tensor(
        initial_values, dtype=dtype, name='initial_values')

    def _condition(counter, parameters, converged, failed):
      del parameters
      early_stop = tf.reduce_all(converged | failed)
      return ~((counter >= max_iterations) | early_stop)

    def _updater(counter, parameters, converged, failed):
      """Updates each parameter via Newton's method."""
      values, gradients = value_and_grad_func(parameters)
      deltas = tf.math.divide(values, gradients)

      converged = tf.abs(
          deltas) < relative_tolerance * tf.abs(values) + tolerance

      # Used to zero out updates to cells that have converged.
      update_mask = tf.cast(~converged, dtype=parameters.dtype)
      increment = -update_mask * deltas
      updated_parameters = parameters + increment
      failed = ~tf.math.is_finite(updated_parameters)

      return counter + 1, updated_parameters, converged, failed

    starting_position = (tf.constant(0, dtype=tf.int32), initial_values,
                         tf.zeros_like(initial_values, dtype=tf.bool),
                         tf.math.is_nan(initial_values))

    return tf.while_loop(_condition, _updater, starting_position,
                         maximum_iterations=max_iterations)[1:]

