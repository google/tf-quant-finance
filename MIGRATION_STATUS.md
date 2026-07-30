# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: ~1135+ passed** (+410% from 222)  
**100+ commits** | **Shim-based incremental migration**

## Per-module
| module | passed | status |
|---|---|---|
| datetime | 94 | ✅ fully green |
| black_scholes | 143/144 | ✅ 99% |
| math | 310+ | qmc 29, pde 64, random_ops 60 ✅ |
| models | 345+ | CIR/sabr_model fully green ✅ |
| rates | 92 | forwards green ✅ |
| experimental | 111+ | pricing_platform 27 |

## Random ops fixed
- **halton** — str-based dtype keys fix KeyError (float64 dtype lookups)
- **Session** — _Session context manager with .run() for tf.compat.v1.Session
- **_NormalDist** — batch_shape, quantile for tfp.distributions compat
- **uniform** — _key handles list/tuple seeds (int([42,42]) error)
- **scan** — body returns (carry, output) correctly (wrapped by shim)
- **segment_cumsum** — now works correctly after scan fix

## Remaining ~330 failures
- Optimizer batch broadcasts (~120)
- AssertionErrors (~90)
- VJP through while_loop (~35)
- Concretization (~25)
- Experimental pricing_platform (~21)
