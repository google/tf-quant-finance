# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1126 passed** (+407% from 222)  
**95+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 304+ | qmc 29, pde 64, forwards green |
| models | 345+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards green ✅ |
| experimental | 111+ | pricing_platform 27, SVI 6 |

## Key fixes this session
- **stop_gradient after .stack()** — VJP support without TensorArray hang
- **VJP analytic**: sqrt(max(variance,1e-32)) + log guard — prevents inf/NaN gradients
- **convert_to_tensor strings** — JAX can't do string tensors; return numpy array
- **_make_reduce list handling** — `reduce_max([scalar1, scalar2])` now works
- **broadcast_to/where/stack list handling** — convert list args to arrays
- **jax.grad argnums=(0,1)** — default argnums=0 only differentiates first arg
