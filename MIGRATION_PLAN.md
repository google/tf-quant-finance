# TF Quant Finance → JAX Migration Plan

> Convert the archived `tf-quant-finance` library from TensorFlow to JAX.
> **Scope:** full repo (all 358 files incl. `experimental/`). **End state:**
> JAX-only, zero `import tensorflow`, all 121 test files green.
> **Approach:** shim-based incremental migration.
> **Tooling:** [`uv`](https://github.com/astral-sh/uv) for all Python env/package
> management; `pytest` via `uv run`.

## Machine & tooling baseline (checked)

- `uv 0.11.29` installed at `/var/home/yangye/.local/bin/uv`.
- Python interpreters available to uv: **3.12**, **3.13**, **3.14**
  (system default `python` = 3.14.6). uv can fetch/pin any of them.
- **Neither TensorFlow nor JAX is installed** anywhere on this machine
> (pip / uv / conda / pipx all empty) → clean slate, nothing to uninstall.
- JAX support window (per docs.jax.dev): Python **3.14 supported** through
  July 2029, 3.13 through July 2028. We pin **3.12** for the broadest
  ecosystem compatibility (numpy/scipy/finance deps); bump later if needed.

---

## 1. Decisions (locked)

| Decision | Choice |
|---|---|
| Scope | **Everything**, including `experimental/` (pricing_platform, LSV, instruments, svi, …) |
| Test gate | **All 121 test files must pass** at the end; suite stays green per module as we go |
| Strategy | **Shim-based incremental** → port module-by-module → delete shim → JAX-only |
| Precision | **float64 mandatory** (quant finance; JAX disables x64 by default) |

## 2. Current-state inventory

- **358** Python files, **~91k** lines, **121** `_test.py` files, **58** Bazel `BUILD` files.
- TF surface area (non-test): the bulk is mechanical.

| Area | Files | Difficulty |
|---|---|---|
| `types/`, `utils/` | 9 | trivial |
| `datetime/` | 20 | low (op-heavy, little math) |
| `black_scholes/` | 21 | low–medium |
| `math/` | 84 | medium (random_ops, pde, optimizer, integration are the hard spots) |
| `models/` | 84 | **high** (while_loop, functional PRNG, gradients) |
| `rates/` | 24 | medium |
| `experimental/` | 108 | mixed; `pricing_platform` (54) is large but mostly type/curve plumbing; `dev/` is one WIP notebook (drop) |

**TF API hotspots (non-test):**
- Mechanical (1:1 → `jnp`/`jax.numpy`): `convert_to_tensor` (926), `math.exp/log/sqrt/...` (919), `constant` (647), `expand_dims` (451), `where` (243), `concat` (206), `reshape` (158), `cast` (145), `zeros_like/ones_like/zeros/ones` (~310), `transpose` (126), `stack/squeeze/range/gather/broadcast_to` (~360).
- `name_scope` (174, 112 files) → **delete entirely** (pure removal).
- dtypes `tf.float32/64`, `tf.int32/64`, `tf.bool` (~810 refs, 65 files) → `jnp.*`.

**Genuinely hard conversions (few, concentrated):**
- `tf.while_loop` (51, models) → `jax.lax.while_loop` / `lax.scan` (pure body, static shapes).
- `tf.function` + `TensorSpec`/`input_signature` (66 + 55) → `jax.jit` or drop.
- `tf.GradientTape` (39, **4 files**: `math/gradient.py`, `math/jacobian.py`, `math/custom_loops.py`, `experimental/instruments/cms_swap.py`) → `jax.grad` / `jax.jacfwd` / `jax.jacrev`.
- `tf.random` + stateless (51, `math/random_ops/*`) → **functional** `jax.random.PRNGKey` (semantic change: keys threaded explicitly).
- `tf.linalg.tridiagonal_solve`/`band_part`/`expm`/`LinearOperator*` → `jax.lax.linalg` / `jax.scipy.linalg` / custom.
- `tf.math.segment_sum` (14) → `jax.ops.segment_sum`; `divide_no_nan` (47) → `jnp.where`.
- `tf.errors.InvalidArgumentError` (44) → `ValueError`; `tf.train.*`/`tf.io.TFRecord*` (one test-data path) → plain file/numpy IO.
- `tf.compat.v1/v2` (122) → resolve to v2 API, then map.

---

## 3. Strategy: shim-based incremental migration

Add a single internal backend module that aliases TF calls to JAX equivalents. Convert files to `import` from the shim first (keeps them runnable under TF semantics while we swap the body), then port each module's body to native JAX, then delete the shim.

```
tf_quant_finance/_backend/__init__.py   # NEW — the shim
```

The shim exposes a `tf`-compatible surface (`convert_to_tensor`, `constant`, `math`, `linalg`, `float32`, …) backed by `jax.numpy`/`jax.lax`. **Why:** lets us run the existing 121-test suite after each module port instead of an all-or-nothing rewrite. It is scaffolding with a known ceiling — deleted in Phase 6.

> `ponytail:` the shim is a temporary bridge; ceiling = it must not leak into the final API. Tracked for deletion in Phase 6.

---

## 4. TF → JAX reference mapping

### 4a. Mechanical (codemod targets)
| TensorFlow | JAX |
|---|---|
| `tf.convert_to_tensor(x, dtype=d, name=...)` | `jnp.asarray(x, dtype=d)` |
| `tf.constant(x)` | `jnp.asarray(x)` / `jnp.array(x)` |
| `tf.float32 / .float64 / .int32 / .int64 / .bool` | `jnp.float32 / .float64 / .int32 / .int64 / .bool_` |
| `tf.expand_dims / squeeze / reshape / concat / stack / transpose / broadcast_to` | `jnp.expand_dims / .squeeze / ...` (identical) |
| `tf.where(cond, x, y)` | `jnp.where(cond, x, y)` |
| `tf.range / linspace / zeros / ones / zeros_like / ones_like` | `jnp.*` |
| `tf.cast(x, d)` | `x.astype(d)` / `jnp.asarray(x, d)` |
| `tf.newaxis` | `jnp.newaxis` |
| `tf.math.exp/log/sqrt/abs/sin/cos/pow/square/erf/ceil` | `jnp.exp/log/sqrt/...` |
| `tf.math.reduce_sum/mean/max/min/any/all` | `jnp.sum/mean/...` (drop `reduce_` prefix) |
| `tf.math.cumsum` | `jnp.cumsum` |
| `tf.math.maximum / minimum` | `jnp.maximum / minimum` |
| `tf.math.logical_and/or` | `jnp.logical_and / or` |
| `tf.math.is_nan` | `jnp.isnan` |
| `tf.math.real` | `jnp.real` |
| `tf.math.divide_no_nan(x,y)` | `_div_no_nan(x,y)` helper = `jnp.where(y!=0, x/y, 0)` (1 shim fn) |
| `tf.math.segment_sum` | `jax.ops.segment_sum` |
| `tf.linalg.matmul / matvec / cholesky / inv / pinv / eigh / einsum / norm / expm` | `jnp.matmul` / `jnp.linalg.*` / `jax.scipy.linalg.expm` |
| `tf.linalg.tridiagonal_solve` | `jax.lax.linalg.tridiagonal_solve` |
| `tf.linalg.band_part` | custom `_band_part` helper (jnp mask) |
| `tf.linalg.LinearOperator*` (2 uses) | rewrite to plain `jnp` ops |
| `tf.name_scope(...)` | **delete** (and the `with` line) |
| `tf.debugging.assert_*` / `tf.assert_equal` | `assert` on `.item()` for static, or `jnp`-based assert; mostly delete |
| `tf.errors.InvalidArgumentError` | `ValueError` |
| `tf.shape(x)` | `jnp.array(x.shape)` (static) or `x.shape` |
| `tf.TensorSpec` / `input_signature` | drop (not used by `jax.jit`) |
| `tf.types.experimental.TensorLike` (types.py) | `Union[jnp.ndarray, np.ndarray, float, int]` |

### 4b. Semantic (manual, per-pattern)
| TensorFlow | JAX | Notes |
|---|---|---|
| `tf.while_loop(cond, body, loop_vars, ...)` | `jax.lax.while_loop` or `lax.scan` | body must be pure; **all carry shapes static**; no Python side effects. `maximum_iterations`→carry a counter; `parallel_iterations` ignored. |
| `tf.function` | `@jax.jit` or **remove** | Most `tf.function` here just batched math → drop, jit at call site. |
| `tf.GradientTape().gradient(y, x)` | `jax.grad(fn)(x)` | Must refactor to a **pure function** of `x`. |
| reverse-mode Jacobian | `jax.jacrev` | `math/jacobian.py` |
| forward-mode Jacobian | `jax.jacfwd` | |
| `tf.random.stateless_uniform(seed, shape)` | `jax.random.uniform(key, shape)` | seed `[2]` int → `jax.random.PRNGKey`; **thread/split keys** explicitly. |
| `tf.random.uniform` (stateful) | `jax.random.uniform(key, ...)` | every caller now needs a `key` arg. |
| multivariate normal / sobol / halton | `jax.random.multivariate_normal` + ported sobol/halton | halton/sobol are deterministic sequences → mostly `jnp` ops, low effort. |

---

## 5. Phased plan (dependency-ordered, bottom-up)

Modules ported in dependency order so dependents inherit working, tested code.
Verified internal dep graph: `models/*` import `math.random`, `math.gradient`,
`math.optimizer`, `math.pde`, `math.piecewise`; `black_scholes` imports `math`;
`rates` imports `math` + `datetime`; `experimental/*` imports everything.

### Phase 0 — Foundation (day 0)
- [ ] **uv project init:** `uv init --no-readme --python 3.12` → creates `pyproject.toml` + `.python-version` (pins 3.12) + `uv.lock`.
- [ ] **uv add runtime deps:** `uv add jax jaxlib numpy scipy`.
- [ ] **uv add dev/group deps:** `uv add --group dev pytest pytest-xdist absl-py parameterized`.
- [ ] **Keep TF temporarily** for diff validation only, isolated in an extra
> dependency group so it never ships: `uv add --group legacy tensorflow`.
> (Not installed today; uv installs it on demand into the legacy group.)
- [ ] `.gitignore`: add `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`.
- [ ] `tf_quant_finance/__init__.py`: at top, `jax.config.update("jax_enable_x64", True)` **before any jax import** (global, mandatory).
- [ ] Create `tf_quant_finance/_backend/` shim (§3) covering §4a entries.
- [ ] Pick codemod tool: `ast-grep` or a `libcst` script for the mechanical rewrites in §4a. Keep the rule list versioned in `tools/tf2jax_codemod.py`.
- [ ] Decide test runner now: `tf.test.TestCase` → `absltest`+`parameterized`+`unittest` (Phase 4), but until then tests still import `tf.test`. Land a `conftest.py` enabling x64 + `jax.config.update("jax_platform_name", ...)`.
- [ ] Baseline: capture current TF test pass list (`uv run pytest --collect-only` → record) so we know what "green" means per module. Run everything through `uv run pytest` from here on.

### Phase 1 — Mechanical bulk codemod (repo-wide, one pass)
Apply §4a across all non-test files via codemod, commit in small grouped chunks:
- [ ] dtype literals (`tf.float32`→`jnp.float32`, …).
- [ ] `name_scope` deletion (174 sites) — pure removal, biggest easy win.
- [ ] 1:1 op swaps (`convert_to_tensor`, `math.*`, `linalg.*`, shape ops, `where`, casts).
- [ ] `tf.errors.*` → `ValueError`; drop `debugging.assert_*`.
- [ ] `types.py` `TensorLike` → Union alias.
- [ ] After this pass, files still import `tensorflow as tf` for the residual hard patterns; that's expected. Run shim-backed tests; fix fallout.

### Phase 2 — Semantic pattern rewrites (concentrated files)
- [ ] **Random/PRNG** (`math/random_ops/`: `stateless.py`, `uniform.py`, `multivariate_normal.py`, `halton/`, `sobol/`): introduce a `PRNGKey`-in/-out convention; update every `tff.math.random` caller in `models/` (33 sites). This is the widest semantic blast radius — do it once, document the key-threading rule, and update all callers together.
- [ ] **while_loop** (`models/euler_sampling.py`, `cir/cir_model.py`, `generic_ito_process.py`, `math/custom_loops.py`, `pde/`, `optimizer/`): convert to `lax.while_loop`/`lax.scan`; enforce static carry shapes. Watch `watch_params` path in euler_sampling (grad through loop).
- [ ] **Gradients** (`math/gradient.py`, `math/jacobian.py`, `math/custom_loops.py`, `experimental/instruments/cms_swap.py`): `tf.GradientTape`→`jax.grad`/`jacrev`/`jacfwd`; refactor targets into pure functions.
- [ ] **jit** (`tf.function`×66): remove most; add `@jax.jit` only where measurably beneficial (default: drop).
- [ ] **linalg edge** (`band_part`, `LinearOperator*`, `tensor_diag`, `tridiagonal_matmul`): custom `jnp` helpers.
- [ ] **misc TF IO**: `tf.train.*`/`tf.io.TFRecord*` (single test-data loader) → numpy/`np.save` or plain file read. Drop `experimental/dev/*.ipynb`.

### Phase 3 — Module-by-module body port + per-module green (bottom-up)
Port remaining residual `tf.` refs to native `jnp`/`jax`, remove shim imports, **run that module's tests to green before moving up**:
- [ ] `types/`, `utils/`
- [ ] `datetime/` (20) — op-heavy, few math deps
- [ ] `math/`: `integration`, `interpolation`, `qmc`, `root_search`, `random_ops`, `optimizer`, `pde`, `linalg helpers`, `gradient`/`jacobian` (already done in P2, finalize)
- [ ] `models/`: `geometric_brownian_motion`, `ito_process`/`generic_ito_process`, `euler_sampling`, `heston`, `cir`, `sabr`, `hull_white`, `hjm`, `longstaff_schwartz`, `legacy`
- [ ] `black_scholes/` (21)
- [ ] `rates/`: `hagan_west`, `constant_fwd`, `nelson_seigel_svensson`, `analytics`
- [ ] `experimental/`: `instruments`, `finite_difference`, `lsm_algorithm`, `local_volatility`, `local_stochastic_volatility`, `svi`, `american_option_pricing`, `pricing_platform` (54 — last; mostly types/curves/daycount)

### Phase 4 — Test framework migration
- [ ] `tf.test.TestCase` (244 refs, 121 files) → `absltest.TestCase` + `parameterized`; `self.evaluate(x)`→`np.asarray(x)`; `self.cached_session()`→delete.
- [ ] Random seeds in tests → explicit `jax.random.PRNGKey(seed)`; assert on `np.asarray(out)`.
- [ ] `pytest` as the single runner; `conftest.py` enforces x64.
- [ ] **Gate: all 121 test files pass.**

### Phase 5 — Packaging & build (uv-native)
- [ ] **Delete `setup.py`**; `pyproject.toml` (from Phase 0) is the single source
> of truth, managed by uv. Runtime deps already = `jax jaxlib numpy scipy`;
> dev deps in the `dev` group. Confirm `[project]` name/version/entry points.
- [ ] Remove the `legacy` TF dependency group: `uv remove --group legacy tensorflow`.
- [ ] Build dist artifacts via uv when needed: `uv build` (replaces `build_pip_pkg.sh`).
- [ ] `tf_quant_finance/__init__.py`: remove `_ensure_tf_install`, `remove_undocumented`, `_REQUIRED_TENSORFLOW_VERSION`; keep x64 config at top.
- [ ] 58 Bazel `BUILD` files: **retire Bazel** in favor of uv/pytest (Bazel's
> only role was TF hermetic builds, which we no longer need). Delete `WORKSPACE`,
> `ci_build/`, `build_pip_pkg.sh`, `third_party/`, root `BUILD`. If any team still
> needs Bazel, that's a separate task — keep this port pip/uv-only.
- [ ] `README.md`: rewrite for JAX + uv (`uv sync`, `uv run pytest`), drop "ARCHIVED"
> note, document float64 + PRNG-key API change.

### Phase 6 — Shim removal & final cleanup
- [ ] Delete `tf_quant_finance/_backend/` (the shim).
- [ ] `grep -rn "import tensorflow\|import tf\| as tf\b"` → **0 hits** outside comments.
- [ ] Final full test run; performance spot-check vs. recorded TF baselines (Phase 0).

---

## 6. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **float32 default** silently loses precision | Wrong prices | Global x64 enable in `__init__` + `conftest.py`; assert in tests |
| **PRNG key threading** forgotten at a call site | Correlated/wrong samples, silent | One sweep over all `tff.math.random` callers (33); type the key arg |
| **while_loop carry shapes** dynamic | `lax.while_loop` shape errors | Enforce static shapes; fall back to `lax.scan` for counted loops |
| **Gradient through while_loop / random** | `jax.grad` of stochastic fn | Use stop-gradient / reparam where TF used `watch_params`; test vs analytic greeks |
| `LinearOperator*` / `band_part` gaps | Blocked files | Small `jnp` helpers in a `math/_linalg_ext.py` |
| **pricing_platform** (54 files) rot | Broken on port | Port last; if a sub-module is unrecoverable, isolate behind a clear `# TODO` rather than block core |
| Bazel `BUILD` rewrite churn | Build breakage | Migrate to pip/poetry if Bazel adds little; keep tests runnable via pytest first |
| Codemod false positives (e.g. `tf.math.real` vs `jnp.real`, kwarg `name=`) | Subtle bugs | Codemod only safe 1:1 rules; review diff per file; keep shim so mistakes surface as test failures not crashes |

## 7. Definition of Done
1. `grep -rn "tensorflow" tf_quant_finance` → **0** (outside this plan / comments).
2. All **121** test files pass: `uv run pytest`.
3. `import tf_quant_finance` works after `uv sync` with only JAX-family deps.
4. float64 enabled; sample greeks match analytic values within tolerance.
5. `setup.py` deleted; `pyproject.toml` is the source of truth; README documents
   the JAX + uv + PRNG-key API.

## 8. Suggested execution order (quick wins → hard)
1. **Phase 0** foundation + shim + x64 (unblocks everything).
2. **Phase 1** codemod (dtype + `name_scope` deletion + 1:1 ops) — clears ~70% of churn in one pass, suite still green via shim.
3. **Phase 2** PRNG sweep + while_loop + gradients — the 3 semantic pillars; do PRNG first (widest blast radius).
4. **Phase 3** bottom-up module finalization, tests green per module.
5. **Phase 4** test framework.
6. **Phase 5** packaging/Bazel.
7. **Phase 6** delete shim, verify DoD.
