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

"""The Levenberg marquardt optimization algorithm.

References:
  MINPACK Library
    https://www.netlib.org/minpack/lmder.f

  Jorge J. Moré, "The Levenberg-Marquardt algorithm: Implementation and theory"
    https://www.osti.gov/biblio/7256021-levenberg-marquardt-algorithm-implementation-theory

  J. J. Moré, B. S. Garbow, and K. E. Hillstrom, "User Guide for MINPACK-1"
    http://cds.cern.ch/record/126569/files/CM-P00068642.pdf
"""

OptimizerResult = collections.namedtuple(
    "OptimizerResult",
    [
        # Scalar boolean tensor indicating whether the minimum
        # was found within tolerance.
        "converged",
        # Scalar boolean tensor indicating whether a line search
        # step failed to find a suitable step size satisfying Wolfe
        # conditions. In the absence of any constraints on the
        # number of objective evaluations permitted, this value will
        # be the complement of `converged`. However, if there is
        # a constraint and the search stopped due to available
        # evaluations being exhausted, both `failed` and `converged`
        # will be simultaneously False.
        "failed",
        # The number of iterations.
        "num_iterations",
        # The total number of objective evaluations performed.
        "num_objective_evaluations",
        # A tensor containing the last argument value found during the search.
        # If the search converged, then this value is the argmin of the
        # objective function (within some tolerance).
        "position",
        # A tensor containing the value of the objective
        # function at the `position`. If the search
        # converged, then this is the (local) minimum of
        # the objective function.
        "objective_value",
        # A tensor containing the jocobian, where each row (i) is the gradient
        # of the position on the ith data point.
        "jacobian",
    ],
)


# Internal state of optimizer.
_OptimizerState = collections.namedtuple(
    "_OptimizerState",
    [
        # Fields from OptimizerResult.
        "converged",
        "failed",
        "num_iterations",
        "num_objective_evaluations",
        # Position (x_k).
        "position",
        # Objective (f_k).
        "objective_value",
        # Jacobian (j_k).
        "jacobian",
        # Levenberg-Marquardt parameter (lm_param_k).
        "lm_parameter",
        # Scaling factors on the position (d_k).
        "scaling_factors",
        # Step size (delta_k).
        "step_size",
        # Reduction ratio (Actual / Predicted) (rho_k).
        "reduction_ratio",
        # Actual reduction
        "actual_reduction",
        # Predicted reduction
        "predicted_reduction",
        # Previous iteration state. If unsuccessful, reuse the position and
        # the jacobian with a new step size
        "prev_iteration_state",
    ],
)


def lm_fit(
    value_and_gradients_function,
    x_data,
    y_data,
    initial_position,
    x_tolerance=1e-8,
    f_tolerance=1e-8,
    g_tolerance=1e-8,
    factor=0.1,
    max_iterations=50,
    parallel_iterations=1,
    stopping_condition=None,
    name=None,
):
    """Fit the function with Levenberg Marquardt algorithm.

  ### Reference:
  MINPACK Library
    https://www.netlib.org/minpack/lmder.f

  Jorge J. Moré, "The Levenberg-Marquardt algorithm: Implementation and theory"
    https://www.osti.gov/biblio/7256021-levenberg-marquardt-algorithm-implementation-theory

  J. J. Moré, B. S. Garbow, and K. E. Hillstrom, "User Guide for MINPACK-1"
    http://cds.cern.ch/record/126569/files/CM-P00068642.pdf


  ### Usage:
  The following example demonstrates this optimizer attempting to minimize the
  least squares loss objective of a simple quadratic function.

  ```python
    x_data = np.array([-1.0, 0.0, 1.0])
    y_data = np.array([-4.0, 0.0, 2.0])

    # The objective function
    def quadratic(x, p):
      return p[0] * (x ** x) + p[1] * x + p[2]

    start = tf.constant([1.0, 1.0, 1.0])
    optim_results = lm_fit(
      quadratic, x_data, y_data, initial_position=start, tolerance=1e-8)

    with tf.Session() as session:
      results = session.run(optim_results)
      # Check that the search converged
      assert(results.converged)
      # Check that the argmin is close to the actual value.
      np.testing.assert_allclose(results.position, minimum)
  ```

  Args:
    value_and_gradients_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
      the value of the function and its gradient at that point. The function to
      be minimized. The input should be of shape `[..., n]`, where `n` is the
      size of the domain of input points, and all others are batching
      dimensions. The first component of the return value should be a real
      `Tensor` of matching shape `[...]`. The second component (the gradient)
      should also be of shape `[..., n]` like the input value to the function.
    x_data: Real `Tensor` of shape `[..., m]. The input values of the fitted
      function.
    y_data: Real `Tensor` of shape `[..., m]. The return values of the fitted
      function.
    initial_position: Real `Tensor` of shape `[..., n]`. The starting point, or
      points when using batching dimensions, of the search procedure. At these
      points the function value and the gradient norm should be finite.
    x_tolerance: Scalar `Tensor` of real dtype. If the absolute change in the
      position between one iteration and the next is smaller than this number,
      the algorithm is stopped.
    f_tolerance: Scalar `Tensor` of real dtype. If the relative change
      in the objective value between one iteration and the next is smaller than
      this value, the algorithm is stopped.
    g_tolerance: Scalar `Tensor` of real dtype. If the cosine angle between the
      position and any column of the jacobian is not more than the tolerance
      in absolute value, the algorithm is stopped.
    factor: Scalar `Tensor` of read dtype. It is used in determining the
      initial step bound. The bound is set to the product of the factor and
      the euclidean norm of the scaled position. In most cases the factor
      should lie in the interval (0.1, 100.0).
    max_iterations: Scalar positive int32 `Tensor`. The maximum number of
      iterations.
    parallel_iterations: Positive integer. The number of iterations allowed to
      run in parallel.
    stopping_condition: (Optional) A Python function that takes as input two
      Boolean tensors of shape `[...]`, and returns a Boolean scalar tensor. The
      input tensors are `converged` and `failed`, indicating the current status
      of each respective batch member; the return value states whether the
      algorithm should stop. The default is tfp.optimizer.converged_all which
      only stops when all batch members have either converged or failed. An
      alternative is tfp.optimizer.converged_any which stops as soon as one
      batch member has converged, or when all have failed.
    params: LevenbergMarquardtParams object with adjustable parameters of the
      algorithm. If not supplied, default parameters will be used.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'minimize' is used.
  Returns:
    optimizer_results: A namedtuple containing the following items:
      converged: boolean tensor of shape `[...]` indicating for each batch
        member whether the minimum was found within tolerance.
      failed:  boolean tensor of shape `[...]` indicating for each batch
        member whether a line search step failed to find a suitable step size
        satisfying Wolfe conditions. In the absence of any constraints on the
        number of objective evaluations permitted, this value will
        be the complement of `converged`. However, if there is
        a constraint and the search stopped due to available
        evaluations being exhausted, both `failed` and `converged`
        will be simultaneously False.
      num_objective_evaluations: The total number of objective
        evaluations performed.
      position: A tensor of shape `[..., n]` containing the last argument value
        found during the search from each starting point. If the search
        converged, then this value is the argmin of the objective function.
      objective_value: A tensor of shape `[...]` with the value of the
        objective function at the `position`. If the search converged, then
        this is the (local) minimum of the objective function.
      objective_gradient: A tensor of shape `[..., n]` containing the gradient
        of the objective function at the `position`. If the search converged
        the max-norm of this tensor should be below the tolerance.
  """
    with tf.compat.v1.name_scope(name, "minimize", [initial_position, x_tolerance]):
        initial_position = tf.convert_to_tensor(
            value=initial_position, name="initial_position"
        )
        dtype = initial_position.dtype
        x_data = tf.convert_to_tensor(value=x_data, dtype=dtype, name="x_data")
        y_data = tf.convert_to_tensor(value=y_data, dtype=dtype, name="y_data")
        x_tolerance = tf.convert_to_tensor(
            value=x_tolerance, dtype=dtype, name="x_tolerance"
        )
        f_tolerance = tf.convert_to_tensor(
            value=f_tolerance, dtype=dtype, name="f_tolerance"
        )
        g_tolerance = tf.convert_to_tensor(
            value=g_tolerance, dtype=dtype, name="g_tolerance"
        )
        max_iterations = tf.convert_to_tensor(
            value=max_iterations, name="max_iterations"
        )
        stopping_condition = stopping_condition or converged_all

        f0, j0 = value_and_gradients_function(initial_position, x_data) - y_data
        converged = tf.norm(f0, axis=-1) < f_tolerance

        # Initialize the scaling factor
        d0 = tf.convert_to_tensor([0.0])

        # Initialize the step size
        delta0 = 1000.0

        initial_state = _OptimizerState(
            converged=converged,
            failed=tf.zeros_like(converged),  # All false.
            num_iterations=tf.convert_to_tensor(value=0),
            num_objective_evaluations=tf.convert_to_tensor(value=1),
            position=initial_position,
            objective_value=f0,
            jacobian=j0,
            lm_parameter=0.0,
            scaling_factors=d0,
            step_size=delta0,
            reduction_ratio=0.0,
            actual_reduction=0.0,
            predicted_reduction=0.0,
            prev_iteration_state=True,
        )

        def _cond(state):
            """Continue if iterations remain and stopping condition is not met."""
            return (state.num_iterations < max_iterations) & tf.logical_not(
                stopping_condition(state.converged, state.failed)
            )

        def _body(state):
            """Main optimization loop."""
            # TODO: Main body
            new_state = _OptimizerState(
                converged=state.converged,
                failed=state.failed,
                num_iterations=state.num_iterations + 1,
                num_objective_evaluations=state.num_objective_evaluations + 1,
                position=state.position,
                objective_value=state.objective_value,
                jacobian=state.jacobian,
                lm_parameter=state.lm_parameter,
                scaling_factors=state.scaling_factors,
                step_size=state.step_size,
                reduction_ratio=state.reduction_ratio,
                actual_reduction=state.actual_reduction,
                predicted_reduction=state.predicted_reduction,
                prev_iteration_state=True,
            )
            return (new_state,)

        final_state = tf.while_loop(
            _cond, _body, (initial_state,), parallel_iterations=parallel_iterations
        )[0]

        return OptimizerResult(
            converged=final_state.converged,
            failed=final_state.failed,
            num_iterations=final_state.num_iterations,
            num_objective_evaluations=final_state.num_objective_evaluations,
            position=final_state.position,
            objective_value=final_state.objective_value,
            jacobian=final_state.jacobian,
        )

        return initial_state


def _value_and_jacobian(value_and_gradients_function, variables, x_data, y_data):
    """Get the value and jacobian matrix from the tensor of x and y data.
  """
    values = []
    gradients = []

    for idx in range(x_data.shape.as_list()[0]):
        v, g = value_and_gradients_function(variables, x_data[idx])
        values.append(v - y_data[idx])
        gradients.append(g)

    return tf.stack(values), tf.stack(gradients)
