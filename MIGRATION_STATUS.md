# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: ~1150 passed** (+418% from 222)  
**110+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 310+ | qmc 29, pde 64, random_ops 60 ✅ |
| models | 350+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards/random_ops green ✅ |
| experimental | 115+ | pricing_platform 34 ✅ |

## Key fixes this session
- **concat atleast_1d** — fix "Zero-dimensional arrays cannot be concatenated" (+7 tests)
- **_chk lazy eval** — prevent TracerArrayConversionError in assert_* (+~30 tests)
- **gauss_kronrod weights** — convert list to tensor before arithmetic
- **count_nonzero** — add dtype kwarg handling via .astype()
- **top_k** — SimpleNamespace with .values/.indices, works with traced k
- **cholesky** — wrap to accept name kwarg
- **optimizer batch** — wrap function for single inputs in LBFGS

## Remaining ~300 failures
- Optimizer batch broadcasts (~100) — CG/LBFGS batch shape mismatches
- Convergence differences (~80) — jaxopt vs TF optimizer tolerance
- VJP through while_loop (~30) — needs scan conversion or custom_vjp
- Concretization (~20) — traced shapes in PDE/model code
- Pricing_platform/american_option (~15) — adaptive integration traced bools
- Other (~55) — various shape/dtype issues
