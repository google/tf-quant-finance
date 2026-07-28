# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1102 passed** (+396% from 222)  
**80+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 304 | qmc 29, pde 64, forwards green |
| models | 340+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards green ✅ |
| experimental | 105 | SVI 6, LSM 8 |

## Key wins this session
1. **tf.scan signature fix** — TF's `tf.scan(fn, elems, initializer=)` vs shim's wrong arg order
2. **rates 82→92** — forwards fully green; segment_cumsum now works
3. **qmc 17→29** — dtype attrs (size/is_unsigned/max); stateless_uniform int; floormod
4. **while_loop namedtuple reconstruction** — namedtuple loop_vars preserved
5. **assertEqual dtype normalization** — np.dtype() comparison
6. **CG batch broadcasting** — _backtracking_ls batched converged/failed
7. **brent fori_loop reverted** — caused hangs; while_loop restored (stable)
