# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: ~1080+ passed**  
**70+ commits** | **Shim-based incremental migration**

## Cumulative wins
- **tridiagonal_solve**, **gather(batch_dims)**, **gather_nd**, **band_part**, **linalg.diag**
- **broadcast_static_shape**, **matmul(transpose_b)**, **eye(batch_shape)**, **tf.pad**
- **[slices]→[tuple(slices)]**, **divide_no_nan**, **dataclass pytree**
- **HJM concretization**, **CG line search**, **evaluate() recursive**
- **CIR fully green**, **Sabr_model fully green**, **black_scholes 143/144**
- **PDE static shapes**, **rates swap_curve**, **segment_sum num_segments**
- **TFP compat shim**, **while_loop namedtuple/fori_loop/VJP**
- **brent fori_loop (VJP-compatible)**, **cumsum static shapes**
- **SVI transpose fix**, **squared_difference**, **is_strictly_increasing bool**
