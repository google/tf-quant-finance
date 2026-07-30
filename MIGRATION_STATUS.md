# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: ~1160+ passed** (+423% from 222)  
**125+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 310+ | qmc 29, pde 64, random_ops 60 ✅ |
| models | 350+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards/random_ops green ✅ |
| experimental | 115+ | pricing_platform 34, SVI calibration ✅ |

## Key fixes this session
- **concat atleast_1d** — fix "Zero-dimensional arrays cannot be concatenated" (+7)
- **_chk lazy eval** — prevent TracerArrayConversionError in assert_* (+~30)
- **gauss_kronrod weights** — convert list to tensor before arithmetic (+3)
- **count_nonzero** — add dtype kwarg handling via .astype() (+2)
- **top_k** — SimpleNamespace with .values/.indices, works with traced k (+2)
- **cholesky** — wrap to accept name kwarg (+1)
- **optimizer batch** — wrap function for single inputs, expand to (1,D) for batched functions (+2)
- **stack values=** — accept values= keyword (TF API compat)
- **xla.experimental.compile** — jax.jit wrapper
- **milstein tuple+list** — list() wrap for shape concat
- **hjm/calibration** — convert target_values to tensor
- **local_volatility** — convert dividend_yield to tensor
- **sigmoid_cross_entropy_with_logits** — implement manually for JAX

## Remaining ~290 failures
- AssertionError/convergence (~98) — jaxopt vs TF optimizer tolerance
- ConcretizationTypeError/Shapes must be ND (~44) — traced shapes
- broadcast shapes (~28) — incompatible broadcasting (mostly fixed)
- VJP through while_loop (~26) — needs scan conversion
- TracerIntegerConversionError (~12) — traced __index__
- dot_general shape mismatch (~10) — mostly fixed
- Other (~72) — various issues

## Known issues
- SimulatedDataCalibrationTest hangs (test infrastructure issue, not code)
- differential_evolution_minimize not implemented
