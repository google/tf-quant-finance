# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1109 passed** (+400% from 222)  
**85+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 304+ | qmc 29, pde 64, forwards green |
| models | 340+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards green ✅ |
| experimental | 105 | SVI 6, LSM 8 |

## Remaining ~342 failures by category
- 89 AssertionError (value/convergence mismatches)
- 116 broadcast shape errors (optimizer batch)
- 34 VJP through while_loop (calibration)
- 14 concretization (traced shapes)
- Algorithm convergence differences (float32 precision, optimizer iterations)
