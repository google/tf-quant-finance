# TF Quant Finance → JAX Migration: Status Summary

**Branch:** `feat/jax-migration` | **Backend:** JAX 0.9.2 on gfx1151 ROCm GPU (CPU for tests)
**Full suite:** **740 passed** (started at 222, **+233%**) | 34+ commits this session

## Per-module status
| module | passed | notes |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% (1 TF-PRNG variance ref) |
| models | 245 | GBM 98%, sabr 67%; heston/cir/hjm/hull_white need work |
| math | 174 | gradient/jacobian/custom_loops green; interpolation/pde need tridiagonal |
| rates | 46 | curve building shape issues |
| experimental | partial | instruments/pricing_platform |

## Architecture (all done)
- **Shim** (`_tf.py`): tf-compatible namespace backed by JAX. Call sites unchanged; only `import tensorflow as tf` → `from tf_quant_finance import _tf as tf`.
- **while_loop + TensorArray** solved: TensorArray as JAX pytree (functional carry); while_loop adapts TF's unpack convention + accepts pytree-like carries.
- **GradientTape** → `jax.grad` (gradient/jacobian/custom_loops native; 6/6 test files converted).
- **random_ops** native (uniform.py, PRNG seed coercion); STATELESS deterministic + correct (GBM MC matches analytic to 1e-5).
- **euler_sampling** native (lax.scan); crr binomial tree (lax.scan fixed-width).
- **math/optimizer** rewired to jaxopt (BFGS/LBFGS/NelderMead).
- tfp eliminated from library code.

## Key bugs fixed (this session)
1. `band_part` row/col swapped → upper-tri instead of lower (GBM time-reversed)
2. `linalg.diag` extracted instead of creating batched diagonal matrices
3. `broadcast_static_shape` nested-tuple → splat (the "shape_utils landmine" root cause, +29 repo-wide)
4. `matmul(transpose_b=)`, `eye(batch_shape=)` — TF kwargs jnp lacks
5. `tensor[slices]` → `tensor[tuple(slices)]` (non-tuple indexing, +29)
6. `divide_no_nan` safe-denominator (JAX 0·inf=nan VJP)
7. `_safe_sqrt` custom_jvp (grad 0 at 0)
8. `@utils.dataclass` registered as JAX pytree
9. eager `debugging.assert_*`; `raise ValueError→InvalidArgumentError` (81 files)
10. `scatter_nd`, `cumsum(exclusive/reverse)`, `tf.range(delta=)`, `tf.size(out_type=)`, `test_util` flex-decorator, `tridiagonal_solve` matrix-format, many more.

## Hard blocker: `tridiagonal_solve` (~157 failures)
jax's `lax.linalg.tridiagonal_solve` has strict off-diagonal-length (m-1) and batch-dim
requirements conflicting with TF's padded/batched conventions. 5 approaches tried (trim,
adaptive trim, explicit linalg.solve, hybrid, matrix-format) — all regress or timeout.
Needs a correct + fast implementation.

## Other remaining (diffuse)
- per-model concretization (heston integration, hjm/LSM scan bodies)
- shape broadcasting in curve/PDE building
- TF-PRNG-specific numerical references (genuine MC variance, ~handful)
