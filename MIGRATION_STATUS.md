# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1076 passed** (+384% from 222)  
**60+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 262 | pde/optimizer/jacobian/CG/root_search green |
| models | 340+ | CIR/sabr_model fully green ✅ |
| rates | 82 | bond_curve/swap_curve |
| experimental | 82+ | |

## Key wins (all commits)
1. CIR fully green — gamma rate/scale; poisson dtype; fold_in; stateless_gamma beta
2. Sabr_model fully green — is_strictly_increasing bool return
3. tridiagonal_solve — rhs [..., m, nrhs] dim; band_part row/col swap
4. gather(batch_dims) negative axis; gather_nd tuple indexing
5. broadcast_static_shape splat; matmul(transpose_b); eye(batch_shape)
6. while_loop — @utils.dataclass unpack/reconstruct
7. TFP shim — optimizer/distributions/stats for test compat
8. as_numpy_dtype on dtype aliases; segment_sum num_segments
9. LSM static shapes; PDE broadcast_common_batch_shape
10. swap_curve_fit stack; halton _dtype_key; nextafter jnp
