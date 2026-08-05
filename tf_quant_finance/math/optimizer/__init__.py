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
     "objective_value", "objective_gradient", "num_iterations", "status"])


def _run_quasi_newton(solver_cls, value_and_gradients_function,
                      initial_position, tolerance, max_iterations, **kw):
    del kw
    init = jnp.asarray(initial_position)

    def _run_scipy(fun, x0):
        """Run scipy L-BFGS-B via jaxopt (robust line search).
        fun returns (value, grad); wrap to value-only for scipy."""
        def value_only(x):
            val, _ = fun(x)
            return val
        solver = jaxopt.ScipyMinimize(
            method='L-BFGS-B', jit=False, fun=value_only, tol=1e-10,
            maxiter=max_iterations)
        p, s = solver.run(x0)
        success = getattr(s, 'success', False)
        nit = getattr(s, 'nit', 0)
        val, grad = fun(p)
        return p, success, nit, val, grad

    # tfp supports batched initial_position [N, D] (N independent solves).
    if init.ndim > 1:
        def make_single_fn(i):
            def single_fn(x, *args):
                # x has shape (D,). Build full batch with x at position i,
                # others at their initial values (batch elements independent).
                full_x = init.at[i].set(x)
                val, grad = value_and_gradients_function(full_x)
                val = jnp.asarray(val)
                grad = jnp.asarray(grad)
                if val.ndim > 0:
                    return val[i], grad[i] if grad.ndim > 1 else grad
                else:
                    return val, grad
            return single_fn

        # Python loop over batch elements with scipy L-BFGS-B
        results = []
        for i in range(init.shape[0]):
            params_i, success_i, nit_i, val_i, grad_i = _run_scipy(
                make_single_fn(i), init[i])
            results.append((params_i, success_i, nit_i, val_i, grad_i))
        params = jnp.stack([r[0] for r in results])
        err = jnp.stack([jnp.asarray(0.0 if r[1] else 1.0) for r in results])
        it = jnp.stack([jnp.asarray(r[2]) for r in results])
        val = jnp.stack([jnp.asarray(r[3]) for r in results])
        grad = jnp.stack([r[4] for r in results])
        converged = jnp.asarray([bool(r[1]) for r in results])
    else:
        params, success, nit, val, grad = _run_scipy(
            value_and_gradients_function, init)
        err = jnp.asarray(0.0 if success else 1.0)
        it = jnp.asarray(nit)
        converged = jnp.asarray(bool(success))
    return _OptResults(
        converged=converged,
        failed=jnp.asarray(False),
        num_objective_evaluations=jnp.asarray(it),
        position=params,
        objective_value=val,
        objective_gradient=jnp.asarray(grad),
        num_iterations=jnp.asarray(it),
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
        objective_gradient=jnp.zeros_like(params), num_iterations=jnp.asarray(it),
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

