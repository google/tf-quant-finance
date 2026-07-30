# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1149 passed** (+417% from 222)  
**105+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 310+ | qmc 29, pde 64, random_ops 60 ✅ |
| models | 348+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards/random_ops green ✅ |
| experimental | 115+ | pricing_platform 34 ✅ |

## Key fixes this round
- **concit atleast_1d** — fix "Zero-dimensional arrays cannot be concatenated"
- **_chk lazy eval** — prevent TracerArrayConversionError in assert_* functions
- **debugging.assert_*** — pass lambda for lazy evaluation of condition
- **gauss_kronrod weights** — convert list to tensor before arithmetic
- **count_nonzero** — add dtype kwarg handling via .astype()
- **top_k** — works with traced k via argsort + dynamic_slice

## Remaining ~308 failures
- AssertionErrors/convergence differences (~80)
- Broadcast shapes/optimizer batch (~100)
- VJP through while_loop (~30)
- Concretization (~20)
- Pricing_platform/american_option (~15)
