# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1320+ passed** (+494% from 222)  
**215+ commits** | **Shim-based incremental migration**

## Per-module (last full run: 1296 passed, 85 failed)
| module | passed | status |
|---|---|---|
| datetime | 95 | ✅ fully green |
| black_scholes | 143 | ✅ fully green |
| math/pde | 86 | ✅ fully green |
| math/optimizer | 19/26 | 73% (3 unimpl: differential_evolution, lbfgs, nelder_mead) |
| math/root_search | 15 | ✅ fully green |
| math/integration | 11/15 | 73% |
| math/interpolation | 50/56 | 89% (2 hipSparse GPU bug, 2 dtype, 2 empty) |
| math/qmc | 32/42 | 76% (9 digital_net RNG) |
| models/cir | 20 | ✅ fully green |
| models/sabr_model | 34 | ✅ fully green |
| models/sabr/calibration | 20/20 | ✅ fully green |
| models/GBM | 62 | ✅ fully green |
| models/euler_sampling | 23/24 | 96% |
| models/heston | 13 + 6 calib | ✅ fully green |
| models/milstein | 10 | ✅ fully green |
| models/realized_volatility | 10 | ✅ fully green |
| models/utils | 9 | ✅ fully green |
| models/hjm | 59/75 | 79% (MC variance, batch transpose) |
| models/hjm/calibration | 5/8 | 63% |
| models/hjm/swaption | 15/19 | 79% |
| models/hull_white | 18 + 14 calib | ✅ 95%+ |
| models/legacy | 20/23 | 87% |
| rates | 101/106 | 95% |
| experimental/local_stoch_vol | 6 | ✅ fully green |
| experimental/instruments | 59 | ✅ fully green |
| experimental/io | 6/7 | 86% |
| experimental/svi/calibration | 7/13 | 54% (real-market-data local minima) |

## Key fixes this session
- **tridiagonal_matmul** — subdiag convention: TF ignores sub[0] not sub[-1] (+13 PDE tests, ALL PDE green)
- **gather batch_dims=-1** — gather along last axis (was row selection)
- **HJM swaption PDE** — static shapes for broadcast_to/reshape/num_times (+11 tests)
- **optimizer scipy L-BFGS-B** — numerical gradients bypass while_loop VJP; SABR 0→20, Heston 0→6, HW cal 3→14, HJM cal 0→5
- **CG optimizer** — batch fix + Python loops (avoids tracing)
- **CMS convexity** — GradientTape → jax.grad + finite diff 2nd derivative (+8)
- **cumsum/cumprod_using_matvec** — lower triangular matrix (HJM identical-paths fix)
- **_grid_from_time_step** — try/except fallback for traced context
- **euler_sampling** — concrete steps_num + dtype consistency
- **assertNear/assertArrayNear** — squeeze/reshape for dim=1 BM tests

## Remaining ~70 failures
- MC variance / RNG differences (~30) — JAX RNG ≠ TF RNG
- SVI real-market-data local minima (~6) — different valid fits
- HJM batch transpose (~4) — complex rank mismatch
- QMC digital_net scrambling (~9) — RNG differences
- Marginal numerical (~10) — tolerance edge cases
- GPU hipSparse bugs (~4) — ROCm infrastructure
- Unimplemented optimizers (~3) — differential_evolution etc
- Other (~4) — various
