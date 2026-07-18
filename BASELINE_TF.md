# TF CPU Baseline (pre-migration reference)

Captured before any JAX conversion, so we can prove parity module-by-module later.

- **Env:** `.venv-tf` (uv), CPython 3.12.13, `tensorflow==2.21.0` (CPU),
  `tensorflow-probability==0.25.0`, `tf-keras==2.21.0`, numpy 2.5.1.
  (No TF-ROCm wheel exists for gfx1151 — verified; CPU only here.)
- **Command:** `TF_CPP_MIN_LOG_LEVEL=3 .venv-tf/bin/python -m pytest tf_quant_finance -q --continue-on-collection-errors -n auto`
- **Raw log:** `baseline_tf.txt`

## Result
```
1208 passed, 69 failed, 89 skipped, 357 errors, 4276 subtests passed, in 206.20s
```

## Interpretation
- **Core library = green.** `math`, `models`, `black_scholes`, `rates`, `datetime`
  (non-experimental), `types`, `utils` largely pass.
- **`experimental/` = bulk of the 357 errors.** Cause: `pricing_platform` imports
  Bazel-generated protobuf modules (`*_pb2.py`) that are **not checked into the
  repo** and were never generated (no Bazel build run). These tests cannot even be
  collected. Pre-existing repo rot, not caused by us. Same for the downstream
  `tff.experimental` namespace errors.
- **A few core failures drift vs TF 2.21** (repo targeted TF ~2.3–2.x):
  - `models/hull_white/...::test_xla_simulation` (XLA compile path)
  - `models/hjm/quasi_gaussian_hjm_test.py`, `models/legacy/brownian_motion_test.py`
  - `math/integration/integration_test.py`, `math/pde/steppers/multidim_parabolic_equation_stepper_test.py`
  - `rates/hagan_west/bond_curve_test.py`, `utils/tf_functions_test.py`, `datetime/holiday_calendar_test.py::test_both_impls`
  These are the pre-existing baseline; the migration does **not** need to fix TF-side
  failures — only to reproduce (or improve) the per-module pass set under JAX.

## What "green per module" means during migration
For each module ported to JAX (Phase 3), its test file must pass at least the set of
cases that passed in this TF baseline. The 357 collection errors in `experimental/`
are excluded from the gate until their protos are generated or the modules are
rewritten (Phase 3, last).
