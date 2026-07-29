# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1112 passed** (+401% from 222)  
**90+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 304+ | qmc 29, pde 64, forwards green |
| models | 348+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards green ✅ |
| experimental | 105 | SVI 6, LSM 8 |

## Key VJP fixes
1. **Analytic**: sqrt(maximum(variance,1e-32)) + log guard — prevents inf/NaN gradients
2. **Simulation**: stop_gradient on while_loop sample_paths — gradient flows through bond reconstitution
3. **argnums=(0,1)** for multi-arg jax.grad in cap_floor test

## Remaining ~341 failures
- 39 VJP through while_loop (subtests in various modules)
- 89 AssertionError (convergence/value differences)
- 116 broadcast shapes (optimizer batch)
- 14 concretization (traced shapes)
