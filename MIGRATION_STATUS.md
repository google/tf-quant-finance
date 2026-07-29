# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1110 passed** (+400% from 222)  
**90+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 304+ | qmc 29, pde 64, forwards green |
| models | 345+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards green ✅ |
| experimental | 105 | SVI 6, LSM 8 |

## Key VJP fix
- sqrt(variance) -> sqrt(maximum(variance, 1e-32)) avoids inf gradient at 0
- log(ratio) -> log(where(ratio>0, ratio, 1)) avoids NaN gradient
- cap_floor analytic gradient now works (simulation VJP needs scan refactor)

## Remaining ~341 failures
- 51 VJP through while_loop (simulation paths; needs scan conversion)
- 89 AssertionError (convergence/value differences)
- 116 broadcast shapes (optimizer batch)
- 14 concretization (traced shapes)
