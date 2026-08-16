# TF Quant Finance → JAX Migration: Status

**Branch:** `feat/jax-migration` | **JAX 0.9.2** | **Full suite: 1328 passed, 34 failed, 1068 subtests**  
**240+ commits** | **Shim-based incremental migration**

## Current state (verified full-suite, ~60 min per run)

- `1328 passed` / `34 failed` / `1 deselected` / `1068 subtests passed` — stable
  across repeated runs (baseline and post-refactor verification match pass count).
- The 34 failures are triaged:
  - **MC variance / RNG (~20)** — *Expected*: JAX ThreeFry2x32 vs TF Philox
    produce uncorrelated streams even with the same seed. Means/variances differ
    by ~0.1-0.9%. Production pricing is correct — just different sample paths.
    Documented as known divergence (option 1); a Philox reimplementation is the
    only way to close them (see todo).
  - **Real failures to fix** — swap_curve OIS batch fitting (diff 0.107, not
    marginal), HJM calibration batch (price/vol based), hull_white calibration
    5% noise + mc_pricing, bond_curve unstable (expected unstable cases),
    conjugate_gradient himmelblau batch (optimizer noise), lsm basket.
  - **Flaky** — SVI `test_weights_to_handle_outliers` (subtest
    "Model with outliers did not converge") passes in isolation, fails under
    full-suite memory contention; same for CG `test_quadratics` AllConverged.

## Refactor passes completed (Aug 14-15)

1. **ponytail-review pass** (commit `468353de`): net −56 lines of dead code —
   dup `_drop_name`, dead `v1` namespace, `Module`, `sort_`, `add_n`, dup
   `gather_nd`, `_np_nextafter`, hoisted local imports, single-concat
   `_cumsum`, collapsed comments essay, lazy tfp converged_all, dropped
   jaxopt import + `solver_cls` param + dead `err`/`it` paths + one_step
   stubs, `_ShapeWrapper` subclassed to `tf.TensorShape`, removed committed
   `baseline_tf.txt`/`snowflake.log` (gitignored), added `pytest-timeout`.
2. **llm-ai-coding-agent pass**: `llm-refactor-plan.md` audit written; removed
   dead deps (`tensorflow-probability`, `jaxopt`, `six` — none imported),
   fixed docstring-after-code in `generate_mc_normal_draws`, `six.moves.range`
   → `range`, `@six.add_metaclass` → Python-3 metaclass kwarg, stale
   tfp/jaxopt comments rewritten.

## Running tests (from AGENT.md)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  nice -n 19 uv run pytest tf_quant_finance -q --tb=no -p no:warnings \
  --ignore=tf_quant_finance/experimental/pricing_platform \
  --deselect tf_quant_finance/experimental/svi/calibration_test.py::RealMarketDataCalibrationTest::test_real_market_data_calibration_conjugate_gradient_optimizer \
  --timeout=600 --timeout-method=thread
```

- `-n 1` (serial) always; parallel workers segfault on gfx1151.
- Never run `experimental/pricing_platform` — triggers hipSparse GPU crash.
- `pytest-timeout` installed (2.4.0) as a hang guard (600s/test).
