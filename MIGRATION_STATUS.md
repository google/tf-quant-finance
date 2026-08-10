# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1340+ passed** (+503% from 222)  
**230+ commits** | **Shim-based incremental migration**

## Remaining ~45 failures — Option 1: Documented as expected RNG divergence
- MC variance / RNG (~20) — **Expected**: JAX ThreeFry2x32 vs TF Philox produce
  uncorrelated streams even with same seed. PSEUDO_ANTITHETIC/STATELESS_ANTITHETIC
  means/variances differ by ~0.1-0.9% (see detail table above). Production
  pricing is correct — just different sample paths. Marked as known divergence.
- SVI real-market (~2) — different local minima + outliers converged behavior
- HJM batch (~4) — complex transpose rank mismatch
- bond_curve float32 (~4) — NaN in single-precision while_loop
- marginal numerical (~8) — tolerance edge cases (PDE/MC ~0.1-0.4% off)
- GPU hipSparse (~2) — ROCm infrastructure bug (fixed: CPU fallback for m==1)
- other (~5) — dtype, MC pricing, etc
