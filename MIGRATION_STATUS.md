# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1270+ passed** (+471% from 222)  
**200+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 95 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math/pde | 86/86 | ✅ fully green |
| math/random_ops | 60/61 | ✅ 98% |
| math/qmc | 32/42 | 76% |
| math/optimizer | 18/33 | 55% (scipy L-BFGS-B robust, SVI real-data partial) |
| math/integration | 11/15 | 73% |
| math/root_search | 15/15 | ✅ fully green |
| math/diff_ops | 5/6 | ✅ 83% |
| models/cir | 20 | ✅ fully green |
| models/sabr_model | 34 | ✅ fully green |
| models/sabr/calibration | 20/20 | ✅ fully green |
| models/GBM | 62 | ✅ fully green |
| models/euler_sampling | 23/24 | ✅ 96% |
| models/heston | 13/13 + 6/6 calib | ✅ fully green |
| models/milstein | 10/10 | ✅ fully green |
| models/realized_volatility | 10/10 | ✅ fully green |
| models/utils | 9/9 | ✅ fully green |
| models/hjm | 13/18 | 72% (MC variance) |
| models/hull_white | 18/19 | 95% |
| models/longstaff_schwartz | 16/17 | 94% |
| models/legacy | 20/23 | 87% |
| rates | 101/106 | ✅ 95% |
| rates/hagan_west/monotone_convex | 14/14 | ✅ fully green |
| utils/shape_utils | 13/13 | ✅ fully green |
| experimental/local_volatility | 14 | ✅ fully green |
| experimental/local_stochastic_vol | 6/6 | ✅ fully green |
| experimental/pricing_platform | 34 | ✅ fully green |
| experimental/io | 6/7 | ✅ 86% |
| experimental/svi/calibration | 4/13 | 31% (real-data local minima) |

## Key fixes this session
- **tridiagonal_matmul** — subdiag convention: TF ignores sub[0] not sub[-1] (+13 PDE tests)
- **_get_grid_delta** — list indexing → tuple indexing for JAX
- **optimizer scipy L-BFGS-B** — replaced jaxopt LBFGS with scipy via jaxopt.ScipyMinimize; SABR 0→20, Heston 0→6
- **tanh param transform** — slower saturation than sigmoid in calibration
- **optimizer batched** — Python loop per batch element (each varies own row)
- **CG batch fix** — broadcast ls_result.failed to batch shape
- **gather batch_dims=-1** — gather along last axis (was row selection); fixes HJM state_y
- **CMS convexity** — GradientTape → jax.grad + finite difference 2nd derivative (+8)
- **linear interpolation** — empty array validation (+2)
- **cumsum/cumprod_using_matvec** — lower triangular matrix (HJM identical-paths fix)

## VJP through while_loop — RESOLVED
vector_hull_white while_loop → scan + PSEUDO_ANTITHETIC precompute + dynamic_update_slice.

## 2D PDE Douglas ADI — RESOLVED
Root cause: tridiagonal_matmul subdiag convention. All 86 PDE tests now pass.

## Remaining ~170 failures
- MC variance / RNG differences (~70) — JAX RNG ≠ TF RNG, unfixable without matching
- SVI real-market-data local minima (~9) — different optimizers find different valid fits
- HJM swaption PDE traced shapes (~15) — complex PDE machinery
- bond_curve float32 divergence (~4) — numerical precision
- QMC digital_net scrambling (~9) — RNG differences
- Optimizer exact-convergence (~15) — scipy vs TF specific minima
- misc GPU hipSparse errors (~10) — ROCm infrastructure
- misc edge cases (~30)

## Known issues
- HJM swaption PDE: traced shapes in PDE machinery (deep issue)
- HALTON/STATELESS RNG: JAX and TF produce different sequences
- SVI real market data: multiple valid local minima
- Brownian motion dim=1: TF squeezes, JAX keeps shape
- GPU hipSparse errors on some 2D interpolation tests (ROCm bug)
