# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1330+ passed** (+499% from 222)  
**225+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 95 | ✅ fully green |
| black_scholes | 143 | ✅ fully green |
| math/pde | 86 | ✅ fully green |
| math/optimizer | 22/26 | 85% (2 CG batch edge cases) |
| math/root_search | 15 | ✅ fully green |
| math/gradient | 5 | ✅ fully green |
| math/diff_ops | 6 | ✅ fully green |
| math/integration | 11/15 | 73% |
| math/interpolation | 50/56 | 89% (2 hipSparse, 2 dtype, 2 empty) |
| math/qmc | 32/42 | 76% (9 digital_net RNG) |
| math/piecewise | partial | dtype (AutoDtype) |
| models/cir | 20 | ✅ fully green |
| models/sabr_model | 34 | ✅ fully green |
| models/sabr/calibration | 20/20 | ✅ fully green |
| models/GBM | 62 | ✅ fully green |
| models/euler_sampling | 23/24 | 96% |
| models/heston | 13 + 6 calib | ✅ fully green |
| models/milstein | 10 | ✅ fully green |
| models/realized_volatility | 10 | ✅ fully green |
| models/utils | 9 | ✅ fully green |
| models/hjm | 75/85 | 88% |
| models/hjm/gaussian_hjm | 13 | ✅ fully green |
| models/hjm/calibration | 5/8 | 63% (VJP through while_loop) |
| models/hjm/swaption | 15/19 | 79% |
| models/hull_white | 18 + 14 calib | 95%+ |
| models/legacy | 20/23 | 87% |
| rates | 101/106 | 95% |
| experimental/instruments | 59 | ✅ fully green |
| experimental/local_stoch_vol | 6 | ✅ fully green |
| experimental/american_option | 10 | ✅ fully green |
| experimental/svi/calibration | 5/10 | 50% (real-market local minima) |
| experimental/io | 6/7 | 86% |
| experimental/lsm | 17/18 | 94% |

## Remaining ~55 failures
- MC variance / RNG (~25) — JAX RNG ≠ TF RNG
- SVI real-market-data (~5) — different local minima
- HJM batch transpose (~4) — complex rank mismatch
- QMC digital_net (~9) — RNG differences
- Bond_curve float32 (~4) — NaN divergence
- Marginal numerical (~4) — tolerance edge cases
- GPU hipSparse (~2) — ROCm bug
- Other (~2) — dtype, etc
