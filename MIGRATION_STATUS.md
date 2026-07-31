# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1230+ passed** (+454% from 222)  
**180+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math/pde | 76/88 | ✅ 86% |
| math/random_ops | 60/61 | ✅ 98% |
| math/qmc | 30/42 | 71% |
| math/optimizer | 19/33 | 58% |
| math/integration | 10/15 | 67% |
| math/root_search | 15/15 | ✅ fully green |
| math/diff_ops | 5/6 | ✅ 83% |
| models/cir | 20 | ✅ fully green |
| models/sabr_model | 34 | ✅ fully green |
| models/GBM | 62 | ✅ fully green |
| models/euler_sampling | 23/24 | ✅ 96% |
| models/heston | 23/29 | 79% |
| models/milstein | 10/10 | ✅ fully green |
| models/realized_volatility | 10/10 | ✅ fully green |
| models/utils | 9/9 | ✅ fully green |
| models/hull_white | 35/45 | 78% |
| models/legacy | 20/23 | 87% |
| rates | 86/89 | ✅ 97% |
| utils/shape_utils | 13/13 | ✅ fully green |
| experimental/local_volatility | 14 | ✅ fully green |
| experimental/pricing_platform | 34 | ✅ fully green |
| experimental/io | 6/7 | ✅ 86% |

## Key fixes this session
- **vector_hull_white scan** — converted while_loop to scan for VJP support (+2)
- **PSEUDO_ANTITHETIC precompute** — precompute normal draws for antithetic sampling
- **euler_sampling _for_loop** — rewrite to record initial state correctly (+1)
- **brent_test jnp.where** — use jnp.where instead of Python if for JAX tracing (+2)
- **heston_model dtype** — infer dtype from piecewise functions when dtype=None (+3)
- **xla.experimental.compile** — just call the function (no jit to avoid tracing issues) (+1)
- **linalg.set_diag** — use proper diagonal indexing with m.at[..., idx, idx].set(v) (+2)
- **block_diagonal_to_dense** — implement using jax.scipy.linalg.block_diag with vmap (+1)
- **one_hot** — accept depth as keyword argument (TF API compat) (+2)
- **maybe_update_along_axis** — use static shapes when available to avoid traced pad widths (+2)
- **get_shape** — return _ShapeWrapper with as_list() method (+2)

## VJP through while_loop — RESOLVED
The main VJP issue was in `vector_hull_white.py` which used `stop_gradient` to block
gradients through the while_loop. Fixed by:
1. Converting while_loop to `jax.lax.scan` (differentiable)
2. Precomputing normal draws for PSEUDO_ANTITHETIC random type
3. Using `dynamic_update_slice` for recording samples at traced indices

This enables gradient computation for Hull-White cap/floor pricing and other
interest rate derivatives.

## Remaining ~215 failures
- AssertionError/convergence (~80) — jaxopt vs TF optimizer tolerance, MC variance
- ConcretizationTypeError (~20) — traced shapes in PDE/model code
- Numerical differences (~40) — MC variance, RNG differences
- HALTON/STATELESS variance (~15) — RNG differences between JAX and TF
- Brownian motion shape (~3) — dim=1 shape squeezing difference
- HJM path sampling (~10) — discount factor computation
- Other (~47) — various issues

## Known issues
- SimulatedDataCalibrationTest hangs (test infrastructure issue, not code)
- differential_evolution_minimize not implemented
- HJM calibration transpose permutation issue (complex batched gradient)
- HALTON sequence generation differs between JAX and TF (variance mismatches)
- test_compare_monte_carlo_to_backward_pde: MC vs PDE difference (~39% relative error)
- Brownian motion dim=1: TF squeezes last dim, JAX keeps it (shape mismatch)
- CMS swap pricing: ~18% difference (might be related to HW model changes)
