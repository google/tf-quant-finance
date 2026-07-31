# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1220+ passed** (+450% from 222)  
**175+ commits** | **Shim-based incremental migration**

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
| math/root_search | 15/15 | ✅ fully green |
| math/diff_ops | 5/6 | ✅ 83% |
| models/cir | 20 | ✅ fully green |
| models/sabr_model | 34 | ✅ fully green |
| models/GBM | 62 | ✅ fully green |
| models/euler_sampling | 23/24 | ✅ 96% |
| models/heston | 23/29 | 79% |
| models/milstein | 10/10 | ✅ fully green |
| models/realized_volatility | 10/10 | ✅ fully green |
| models/utils | 9/9 | ✅ fully green |
| models/legacy | 20/23 | 87% |
| rates | 86/89 | ✅ 97% |
| experimental/local_volatility | 14 | ✅ fully green |
| experimental/pricing_platform | 34 | ✅ fully green |
| experimental/io | 6/7 | ✅ 86% |

## Key fixes this session
- **euler_sampling _for_loop** — rewrite to record initial state correctly (+1)
- **brent_test jnp.where** — use jnp.where instead of Python if for JAX tracing (+2)
- **heston_model dtype** — infer dtype from piecewise functions when dtype=None (+3)
- **heston_model _SQRT_2** — convert to match normals dtype to avoid float32/float64 mismatch
- **heston_model _get_parameters** — convert times to tensor to handle list input
- **xla.experimental.compile** — just call the function (no jit to avoid tracing issues) (+1)
- **linalg.set_diag** — use proper diagonal indexing with m.at[..., idx, idx].set(v) (+2)
- **block_diagonal_to_dense** — implement using jax.scipy.linalg.block_diag with vmap (+1)
- **one_hot** — accept depth as keyword argument (TF API compat) (+2)
- **maybe_update_along_axis** — use static shapes when available to avoid traced pad widths (+2)

## Previous session fixes
- **concat atleast_1d** — fix "Zero-dimensional arrays cannot be concatenated" (+7)
- **_chk lazy eval** — prevent TracerArrayConversionError in assert_* (+~30)
- **gauss_kronrod weights** — convert list to tensor before arithmetic (+3)
- **count_nonzero** — add dtype kwarg handling via .astype() (+2)
- **top_k** — SimpleNamespace with .values/.indices, works with traced k (+2)
- **cholesky** — wrap to accept name kwarg (+1)
- **optimizer batch** — use vmap with jit=True for efficient batched optimization (+2)
- **stack values=** — accept values= keyword (TF API compat)
- **milstein tuple+list** — list() wrap for shape concat
- **hjm/calibration** — convert target_values and init_corr to tensor
- **local_volatility** — convert dividend_yield to tensor
- **sigmoid_cross_entropy_with_logits** — implement manually for JAX
- **PDE ConcretizationTypeError** — use lax.pad for traced pad widths (+7)
- **PDE pad widths** — use jax.lax.pad with static pad config (+13)
- **top_k traced k** — catch ConcretizationTypeError (+1)
- **num_iterations** — add to OptimizerResults namedtuple (+15)
- **TensorProto** — SerializeToString/FromString for io tests (+6)
- **meshgrid** — flatten multi-dim inputs for JAX compat (+1)
- **tf.fill** — accept dims/value kwargs for TF compat (+2)
- **assertProtoEquals** — add to TestCase for proto_utils tests (+2)
- **tf.gradients** — return list (TF API compat) (+3)
- **euler_sampling traced indexing** — use dynamic_index_in_dim + squeeze (+17)

## Remaining ~225 failures
- AssertionError/convergence (~85) — jaxopt vs TF optimizer tolerance, MC variance
- ConcretizationTypeError (~26) — traced shapes
- VJP through while_loop (~26) — needs scan conversion
- broadcast shapes (~28) — incompatible broadcasting
- TracerIntegerConversionError (~12) — traced __index__
- HALTON/STATELESS variance (~15) — RNG differences between JAX and TF
- Brownian motion shape (~3) — dim=1 shape squeezing difference
- Other (~30) — various issues

## Known issues
- SimulatedDataCalibrationTest hangs (test infrastructure issue, not code)
- differential_evolution_minimize not implemented
- HJM calibration transpose permutation issue (complex batched gradient)
- tf.gradients TF1-style not fully compatible with JAX (no computational graph)
- HALTON sequence generation differs between JAX and TF (variance mismatches)
- test_compare_monte_carlo_to_backward_pde: MC vs PDE difference (~39% relative error)
- Brownian motion dim=1: TF squeezes last dim, JAX keeps it (shape mismatch)
