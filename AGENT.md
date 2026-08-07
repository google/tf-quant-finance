# Agent Configuration — tf-quant-finance JAX Migration

## Running Tests

**CRITICAL:** JAX on gfx1151 (Strix Halo) pre-allocates GPU memory and causes OOM kills.
Always set these environment variables before running pytest:

```bash
# Low memory (safe, ~14GB): MEM_FRACTION=0.25
# Medium memory (~27GB):    MEM_FRACTION=0.50
# High memory (~54GB):      MEM_FRACTION=0.75
# Max memory (~70GB):       MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

### Recommended test commands

```bash
# Full suite (serial, ~8-10 min, ~54GB GPU mem at 0.75)
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  nice -n 19 uv run pytest tf_quant_finance/ -n 1 --tb=no -q -p no:warnings

# Single module (fast)
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  nice -n 19 uv run pytest tf_quant_finance/models/hjm/ -n 1 --tb=no -q -p no:warnings

# Single test
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  nice -n 19 uv run pytest "tf_quant_finance/models/hjm/quasi_gaussian_hjm_test.py::HJMModelTest::test_correctness_1d" -x -q
```

### Rules
- **Always use `-n 1`** (serial). Parallel workers (`-n 4+`) cause segfaults on gfx1151.
- **Always use `nice -n 19`** to avoid starving the system.
- **Always set `XLA_PYTHON_CLIENT_PREALLOCATE=false`** to avoid pre-allocating all GPU memory.
- **Never run `tf_quant_finance/experimental/pricing_platform/`** — it triggers a hipSparse GPU crash (ROCm bug, not our code).
- **The full models/ suite can take >15 min** — set timeout to 1200+ or run per-module.

## Project Context

- **Branch:** `feat/jax-migration`
- **Approach:** Shim-based incremental migration (`from tf_quant_finance import _tf as tf`)
- **JAX:** 0.9.2 with ROCm 7.14 nightly wheels
- **x64 enabled** via `conftest.py` (quant finance needs float64)
- **Status:** See `MIGRATION_STATUS.md` for per-module pass rates and known issues

## Key Files
- `tf_quant_finance/_tf.py` — the TF→JAX shim (~1600+ lines)
- `conftest.py` — enables x64 + initializes ROCm
- `MIGRATION_STATUS.md` — progress tracker
