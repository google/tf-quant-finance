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
"""Optimization methods."""


# ponytail: tfp optimizers rewired to jaxopt (tfp.substrates.jax is incompatible
# with jax 0.9.2). Provide tfp-compatible signatures + result namedtuples.
import jax
import jax.numpy as jnp
import jaxopt
from collections import namedtuple as _namedtuple

_OptResults = _namedtuple(
    "OptimizerResults",
    ["converged", "failed", "num_objective_evaluations", "position",
     "objective_value", "objective_gradient", "n_iterations", "status"])


def _run_quasi_newton(solver_cls, value_and_gradients_function,
                      initial_position, tolerance, max_iterations, **kw):
    del kw
    init = jnp.asarray(initial_position)
    # tfp supports batched initial_position [N, D] (N independent solves).
    # jaxopt's vmap triggers TracerBoolConversionError in line search; use
    # a Python loop instead (slower but correct).
    if init.ndim > 1:
        # Wrap function to return scalar for single input (i-th batch element).
        def make_single_fn(i):
            def single_fn(x):
                val, grad = value_and_gradients_function(x)
                # val may be scalar or batched. grad may be (D,) or (N, D).
                if hasattr(val, 'ndim') and val.ndim > 0:
                    return val[i], grad[i]
                else:
                    return val, grad
            return single_fn
        all_params = []
        all_states = []
        for i in range(init.shape[0]):
            solver = solver_cls(fun=make_single_fn(i),
                                value_and_grad=True, tol=tolerance,
                                maxiter=max_iterations, jit=False)
            p, s = solver.run(init[i])
            all_params.append(p)
            all_states.append(s)
        params = jnp.stack(all_params)
        err = jnp.stack([getattr(s, 'error', jnp.asarray(0.0)) for s in all_states])
        it = jnp.stack([getattr(s, 'iter_num', jnp.asarray(max_iterations)) for s in all_states])
        val = jnp.stack([getattr(s, 'value', jnp.asarray(0.0)) for s in all_states])
        grad = jnp.stack([getattr(s, 'grad', jnp.zeros_like(init[0])) for s in all_states])
    else:
        solver = solver_cls(fun=value_and_gradients_function,
                            value_and_grad=True, tol=tolerance,
                            maxiter=max_iterations, jit=False)
        params, state = solver.run(init)
        err = getattr(state, "error", jnp.asarray(0.0))
        it = getattr(state, "iter_num", jnp.asarray(max_iterations))
        val = getattr(state, "value", jnp.asarray(0.0))
        grad = getattr(state, "grad", jnp.zeros_like(params))
    return _OptResults(
        converged=jnp.asarray(err < tolerance),
        failed=jnp.asarray(False),
        num_objective_evaluations=jnp.asarray(it),
        position=params,
        objective_value=val,
        objective_gradient=grad,
        n_iterations=jnp.asarray(it),
        status=jnp.asarray(0))


def bfgs_minimize(value_and_gradients_function, initial_position,
                  tolerance=1e-8, max_iterations=50, **kwargs):
    return _run_quasi_newton(jaxopt.BFGS, value_and_gradients_function,
                             initial_position, tolerance, max_iterations, **kwargs)


def lbfgs_minimize(value_and_gradients_function, initial_position,
                   tolerance=1e-8, max_iterations=50, **kwargs):
    return _run_quasi_newton(jaxopt.LBFGS, value_and_gradients_function,
                             initial_position, tolerance, max_iterations, **kwargs)


def nelder_mead_minimize(function, initial_vertex=None, initial_position=None,
                         tolerance=1e-8, max_iterations=50, **kwargs):
    init = initial_position if initial_position is not None else initial_vertex
    solver = jaxopt.ScipyMinimize(method="Nelder-Mead", tol=tolerance,
                                  options={"maxiter": max_iterations}, jit=False)
    params, state = solver.run(init, fun=function)
    err = getattr(state, "error", jnp.asarray(0.0))
    it = getattr(state, "iter_num", jnp.asarray(max_iterations))
    return _OptResults(
        converged=jnp.asarray(getattr(state, "success", err < tolerance)),
        failed=jnp.asarray(False), num_objective_evaluations=jnp.asarray(it),
        position=params, objective_value=getattr(state, "fun", jnp.asarray(0.0)),
        objective_gradient=jnp.zeros_like(params), n_iterations=jnp.asarray(it),
        status=jnp.asarray(0))


def converged_all(losses, tolerance=1e-8, *args, **kw):
    """tfp.optimizer.converged_all: True when all values are within tolerance."""
    return jnp.all(jnp.abs(jnp.asarray(losses)) < tolerance)


def converged_any(losses, tolerance=1e-8, *args, **kw):
    return jnp.any(jnp.abs(jnp.asarray(losses)) < tolerance)


class _LineSearchNS:
    def sigmoid_cross_entropy_with_logits(self, *a, **k):
        return jnp.where(k.get("labels", a[1]) == 0,
            jnp.log1p(jnp.exp(-jnp.abs(a[0]))) + jnp.maximum(a[0], 0),
            jnp.log1p(jnp.exp(-jnp.abs(a[0]))) - a[0])
    def hager_zhang(self, *a, **k):
        raise NotImplementedError("linesearch.hager_zhang not wired (CG path)")


linesearch = _LineSearchNS()
differential_evolution_minimize = None
differential_evolution_one_step = None
nelder_mead_one_step = None

from tf_quant_finance.math.optimizer.conjugate_gradient import ConjugateGradientParams
from tf_quant_finance.math.optimizer.conjugate_gradient import minimize as conjugate_gradient_minimize

_allowed_symbols = [
    'bfgs_minimize',
    'differential_evolution_minimize',
    'differential_evolution_one_step',
    'conjugate_gradient_minimize',
    'converged_all',
    'converged_any',
    'lbfgs_minimize',
    'linesearch',
    'nelder_mead_minimize',
    'nelder_mead_one_step',
    'ConjugateGradientParams',
]

