# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on gfx1151 ROCm GPU (CPU for testing)  
**Full suite: 975 passed** (started 222, **+337%**) | 50+ commits

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 257 | gradient/jacobian/CG/root_search/brent/piecewise green |
| models | ** | CIR Fully green, GBM 98%, sabr ~94 |
| rates | 53 | |
| experimental | 82 | |

## Key wins (cumulative)
1. **tridiagonal_solve** (was 157) — rhs `[..., m, nrhs]` dim
2. **gather(batch_dims) negative axis** (+86) — axis out-of-bounds root cause
3. **gather_nd** tuple indexing (+31)
4. **band_part** row/col swap; **linalg.diag** create vs extract
5. **broadcast_static_shape** splat — shape_utils landmine
6. **matmul(transpose_b)**, **eye(batch_shape)**, **tridiagonal_matmul**, **tf.pad(paddings=)**
7. **[slices]→[tuple(slices)]**, **divide_no_nan** safe-denom, **dataclass pytree**
8. **HJM concretization** — manual pad in scan body
9. CG backtracking line search; tf.unique; gather_nd; is_strictly_increasing
10. evaluate() recursive dataclass; while_loop tuple return;
11. **CIR fully green** — gamma rate/scale fix; poisson dtype; _to_key fold_in; stateless_gamma beta
