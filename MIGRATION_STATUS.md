# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on CPU (GPU segfaults)
**Full suite: 948 passed** (started 222, **+327%**)

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 249 | gradient/jacobian/custom_loops/CG/root_search green |
| models | 309 | GBM 98%, sabr 94/124 |
| rates | 53 | |
| experimental | 82 | |

## Major wins (cumulative, 50+ commits)
1. **tridiagonal_solve** (was 157) — rhs `[..., m, nrhs]` dim required
2. **gather(batch_dims) negative axis** (+86) — "axis out of bounds" root cause
3. **gather_nd** proper tuple indexing (+31)
4. **band_part** row/col swap; **linalg.diag** create vs extract
5. **broadcast_static_shape** splat — the shape_utils landmine
6. **matmul(transpose_b)** , **eye(batch_shape)** , **tridiagonal_matmul** , **tf.pad(paddings=)** 
7. **[slices]→[tuple(slices)]**, **divide_no_nan** safe-denom, **dataclass pytree**
8. **HJM concretization** fixed — manual pad in scan body
9. CG backtracking line search; tf.unique; tf.gather_nd; is_strictly_increasing
10. evaluate() recursive dataclass handling; while_loop tuple return; global_variables_initializer
