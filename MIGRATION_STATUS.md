# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on gfx1151 ROCm GPU (CPU for tests)
**Full suite:** **755 passed** (started at 222, **+240%**) | 40+ commits

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ green |
| black_scholes | 143/144 | ✅ 99% |
| models | 245 | GBM 98%, sabr 67% |
| math | 174 | gradient/jacobian/custom_loops green |
| rates | 46 | |

## Key wins
- **tridiagonal_solve SOLVED** (was ~157 failures): jax 0.9.2 requires rhs `[...,m,nrhs]`
- **band_part** row/col swap; **linalg.diag** create vs extract; **broadcast_static_shape** splat (shape_utils landmine)
- **matmul(transpose_b)**, **eye(batch_shape)**, **tridiagonal_matmul**, **tf.pad(paddings=)**
- **[slices]→[tuple(slices)]**; **divide_no_nan** safe-denom; **dataclass pytree**; eager **debugging.assert_***

## Remaining
- PDE stepper axis errors (123), shape broadcasting (168 diffuse), concretization (50), per-model deep dives
