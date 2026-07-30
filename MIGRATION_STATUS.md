# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1191 passed** (+435% from 222)  
**150+ commits** | **Shim-based incremental migration**

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
| math/root_search | 13/15 | 87% |
| models/cir | 20 | ✅ fully green |
| models/sabr_model | 34 | ✅ fully green |
| models/GBM | 62 | ✅ fully green |
| models/heston | 20/29 | 69% |
| models/legacy | 20/23 | 87% |
| rates | 86/89 | ✅ 97% |
| experimental/local_volatility | 14 | ✅ fully green |
| experimental/pricing_platform | 34 | ✅ fully green |

## Key fixes this session
- **concat atleast_1d** — fix "Zero-dimensional arrays cannot be concatenated" (+7)
- **_chk lazy eval** — prevent TracerArrayConversionError in assert_* (+~30)
- **gauss_kronrod weights** — convert list to tensor before arithmetic (+3)
- **count_nonzero** — add dtype kwarg handling via .astype() (+2)
- **top_k** — SimpleNamespace with .values/.indices, works with traced k (+2)
- **cholesky** — wrap to accept name kwarg (+1)
- **optimizer batch** — use vmap with jit=True for efficient batched optimization (+2)
- **stack values=** — accept values= keyword (TF API compat)
- **xla.experimental.compile** — jax.jit wrapper
- **milstein tuple+list** — list() wrap for shape concat
- **hjm/calibration** — convert target_values and init_corr to tensor
- **local_volatility** — convert dividend_yield to tensor
- **sigmoid_cross_entropy_with_logits** — implement manually for JAX
- **PDE ConcretizationTypeError** — use lax.pad for traced pad widths (+7)
- **PDE pad widths** — use jax.lax.pad with static pad config (+13)
- **top_k traced k** — catch ConcretizationTypeError (+1)

## Remaining ~254 failures
- AssertionError/convergence (~98) — jaxopt vs TF optimizer tolerance
- TracerArrayConversionError (~30) — numpy array conversion on traced arrays
- VJP through while_loop (~26) — needs scan conversion
- TracerIntegerConversionError (~12) — traced __index__
- Other (~88) — various issues

## Known issues
- SimulatedDataCalibrationTest hangs (test infrastructure issue, not code)
- differential_evolution_minimize not implemented
- HJM calibration transpose permutation issue (complex batched gradient)
- tf.gradients TF1-style not fully compatible with JAX (no computational graph)
