# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on gfx1151 ROCm GPU  
**Full suite: 932 passed** (started 222, **+319%**) | 50+ commits

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 239 | gradient/jacobian/custom_loops/CG green |
| models | 309 | GBM 98%, sabr 94/124, hull_white 22, hjm 12 |
| rates | 53 | |
| experimental | 76 | |

## Key wins (cumulative)
1. **tridiagonal_solve** (was 157 failures) — rhs `[..., m, nrhs]` dim
2. **gather(batch_dims) negative axis** (+86 tests)  
3. **gather_nd** proper tuple indexing (+31 tests)
4. **band_part** row/col swap; **linalg.diag** create vs extract
5. **broadcast_static_shape** splat (shape_utils landmine)
6. **matmul(transpose_b)**, **eye(batch_shape)**, **tridiagonal_matmul**, **tf.pad(paddings=)**
7. **[slices]→[tuple(slices)]**, **divide_no_nan** safe-denom, **dataclass pytree**
8. **HJM concretization** — tf.pad→manual concat in scan body
9. CG backtracking line search; tf.unique; debugging.is_strictly_increasing
10. np imports; floor_div; cumprod; segment ops; global_variables_initializer
