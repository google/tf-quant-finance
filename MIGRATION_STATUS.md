# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on gfx1151 ROCm GPU
**Full suite:** **844 passed** (started at 222, **+280%**) | 39 commits

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 216 | gradient/jacobian/custom_loops/interpolation green |
| models | 245 | GBM 98%, sabr 67% |
| rates | 53 | |
| experimental | 75 | |

## Key wins this session
1. **tridiagonal_solve SOLVED** (157→0 failures): jax 0.9.2 requires rhs `[...,m,nrhs]`
2. **gather(batch_dims) negative axis** (+86): axis-1-axis - batch_dims was invalid for negative axes
3. **band_part** row/col swap; **linalg.diag** create vs extract
4. **broadcast_static_shape** splat (shape_utils landmine root cause)
5. **matmul(transpose_b)**, **eye(batch_shape)**, **tridiagonal_matmul**, **tf.pad(paddings=)**
6. **[slices]→[tuple(slices)]**, **divide_no_nan** safe-denom, **dataclass pytree**
7. eager **debugging.assert_***, **tf.Variable** shim, **halton tile** fix

## Remaining (diffuse)
- PDE "Shapes must be ND" (62) — stepper builds shapes from traced/float dims
- Shape broadcasting (~168 diffuse)
- Per-model concretization (heston integration, hjm/LSM scan bodies)
- HJM unpack (40) + concretization (26)
