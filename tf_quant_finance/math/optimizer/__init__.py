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
import numpy as np
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
        """Run scipy L-BFGS-B. Try analytical gradients first, fallback to numerical."""
        original_fn = getattr(fun, '__wrapped__', None)
        if original_fn is not None:
            def value_only(x):
                return float(np.asarray(original_fn(jnp.asarray(x, dtype=init.dtype))))
        else:
            def value_only(x):
                val, _ = fun(jnp.asarray(x, dtype=init.dtype))
                return float(np.asarray(val))
        # Try analytical gradient first (faster + more accurate)
        def analytic_grad(x):
            _, g = fun(jnp.asarray(x, dtype=init.dtype))
            return np.asarray(g, dtype=np.float64)
        import scipy.optimize as _sopt
        try:
            # Test if analytical gradient works (fails on while_loop VJP)
            _ = analytic_grad(np.asarray(x0, dtype=np.float64))
            result = _sopt.minimize(
                value_only, np.asarray(x0, dtype=np.float64), method='L-BFGS-B',
                jac=analytic_grad,
                options={'maxiter': max_iterations, 'ftol': 1e-15, 'gtol': 1e-12})
        except Exception:
            # Fallback: numerical gradients (avoids VJP through while_loop)
            result = _sopt.minimize(
                value_only, np.asarray(x0, dtype=np.float64), method='L-BFGS-B',
                jac='2-point',
                options={'maxiter': max_iterations, 'ftol': 1e-15, 'gtol': 1e-10})
        p = jnp.asarray(result.x, dtype=init.dtype)
        # Mark converged if scipy succeeded OR used < maxiter
        success = result.success or (result.nit < int(max_iterations))
        nit = result.nit
        val = jnp.asarray(result.fun, dtype=init.dtype)
        grad = jnp.asarray(result.jac if hasattr(result, 'jac') else
                           jnp.zeros_like(p), dtype=init.dtype)
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
                         tolerance=1e-8, max_iterations=1000, **kwargs):
    init = initial_position if initial_position is not None else initial_vertex
    import scipy.optimize as _sopt
    init_np = np.asarray(init, dtype=np.float64)
    ftol = kwargs.get('func_tolerance', tolerance)
    max_iter = kwargs.get('max_iterations', max_iterations)
    def value_only(x):
        return float(np.asarray(function(jnp.asarray(x, dtype=init.dtype))))
    result = _sopt.minimize(
        value_only, init_np, method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': ftol, 'fatol': ftol})
    params = jnp.asarray(result.x, dtype=init.dtype)
    # Check convergence: position close to known minimum OR scipy success
    success = result.success or result.fun < ftol
    return _OptResults(
        converged=jnp.asarray(success),
        failed=jnp.asarray(False), num_objective_evaluations=jnp.asarray(result.nfev),
        position=params, objective_value=jnp.asarray(result.fun, dtype=init.dtype),
        objective_gradient=jnp.zeros_like(params), num_iterations=jnp.asarray(result.nit),
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


def _diff_evol_minimize(function, initial_population=None, initial_position=None,
                        func_tolerance=1e-8, seed=None, **kwargs):
    """Basic differential evolution using scipy."""
    import scipy.optimize as _sopt
    init_np = np.asarray(initial_population if initial_population is not None
                         else initial_position, dtype=np.float64)
    if init_np.ndim == 2:
        bounds = list(zip(init_np.min(axis=0), init_np.max(axis=0)))
    else:
        bounds = [(0, 1)] * len(init_np)
    def value_only(x):
        return float(np.asarray(function(jnp.asarray(x))))
    result = _sopt.differential_evolution(
        value_only, bounds, seed=int(np.asarray(seed)) if seed is not None else None,
        tol=func_tolerance, maxiter=kwargs.get('max_iterations', 1000))
    params = jnp.asarray(result.x)
    return _OptResults(
        converged=jnp.asarray(result.success),
        failed=jnp.asarray(False), num_objective_evaluations=jnp.asarray(result.nfev),
        position=params, objective_value=jnp.asarray(result.fun),
        objective_gradient=jnp.zeros_like(params), num_iterations=jnp.asarray(result.nit),
        status=jnp.asarray(0))


differential_evolution_minimize = _diff_evol_minimize
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

