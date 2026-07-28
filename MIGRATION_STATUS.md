# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** on gfx1151 ROCm GPU (CPU for testing)  
**Full suite: 1056 passed** (started 222, **+375%**) | 60+ commits

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 262 | pde 64, optimizer, gradient, CG, root_search |
| models | 312+ | CIR fully green, GBM 98%, sabr, heston 20, hull_white 36 |
| rates | 82 | bond_curve 13, swap_curve green |
| experimental | 82+ | LSM |

## Key wins this session (per-formula investigation)
1. **CIR fully green** — gamma rate/scale (divide not multiply); poisson dtype; _to_key fold_in; stateless_gamma beta
2. **sparse.to_dense shim** — scatter-based for multi-dim
3. **while_loop dataclass** — unpack @utils.dataclass fields, reconstruct on return
4. **halton dtype keys** — _dtype_key() normalizes jnp.float64/np.dtype lookups
5. **broadcast_common_batch_shape** — static shape list (jit-safe, no traced concat)
6. **heston TensorArray** — pass element_shape (fixes carry structure mismatch)
7. **LSM basis_fn** — static shape (not tf.shape) for slice sizes
8. **segment_sum num_segments** — forward kwarg + static call sites
9. **nextafter jnp** — np.nextafter fails on tracers
10. **swap_curve_fit** — stack instrument_weights (list*array -> stack)

## Cumulative major wins (50+ commits prior)
- tridiagonal_solve, gather(batch_dims), gather_nd, band_part, linalg.diag
- broadcast_static_shape, matmul(transpose), eye(batch_shape), tf.pad
- [slices]→[tuple(slices)], divide_no_nan, dataclass pytree
- HJM concretization, CG line search, evaluate() recursive
