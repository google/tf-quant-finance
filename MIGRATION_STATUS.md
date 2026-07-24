# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on gfx1151 ROCm GPU  
**Full suite: 855 passed** (started 222, **+285%**) | 41 commits

## Per-module
| module | passed | Δ |
|---|---|---|
| datetime | 94 ✅ | GREEN |
| black_scholes | 143/144 | 99% |
| math | 216 | +27 (gather axis fix) |
| models | 256 | +11 (HJM concretization) |
| rates | 53 | +7 (gather axis fix) |

## What's been solved (systematic issues)
1. **tridiagonal_solve** (was 157 failures) — rhs `[..., m, nrhs]` dim required
2. **gather(batch_dims) negative axis** (+86 tests) — axis=-1-batch_dims→-ND invalid
3. **HJM concretization** (26→0) — tf.pad→manual concat in scan body
4. **band_part**, **linalg.diag**, **broadcast_static_shape** splat, **matmul** transpose_b/eye batch_shape
5. **[slices]→[tuple(slices)]**, **divide_no_nan**, **dataclass pytree**, **debugging.assert***
6. **tf.Variable**, **assign_add**, **halton tile**, **assert_less_equal**

## What remains (536 failed, diffuse)
- PDE (34): douglas_adi value errors + multidim stepper shapes
- Math: optimizer (22), random_ops (29), qmc (17), root_search (12)
- Models: HJM (remaining 55), hull_white (49), heston (18), cir (15), longstaff (12), sabr (41 → 83 passing)
- Rates: curve shape (48)
- Experimental: instruments/pricing_platform (106)

Next targets: experimental (106) and remaining math sub-modules.
