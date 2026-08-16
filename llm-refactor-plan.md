# LLM-AI Refactor Plan — tf-quant-finance JAX Migration

Audit against the LLM AI Coding Agent principles: flat explicit architecture,
regenerable files, no dead flexibility, docs-before-code, minimal deps.
The migration (shim-based `tf` → JAX) is functionally complete (1328+ passed);
this plan is about making the code predictable for the next LLM maintainer.

## High

1. **Docstrings after code** — `models/utils.py:generate_mc_normal_draws` has 4
   lines of int-casting before the docstring. Move the docstring to be the first
   statement of the function body. (Also scan for other functions with
   statements preceding the docstring: `grep -l '^[^ ]*def .*:\n  [^" ]'`.)
   *Why:* a future LLM reads docs first; code-before-doc breaks regeneration.

2. **Remove dead dependencies** — `pyproject.toml` still lists
   `tensorflow-probability>=0.25.0` and `jaxopt>=0.8.5`, but neither is
   imported anywhere in the repo:
   - tfp: all 8 users import the shim namespace
     `from tf_quant_finance._tf import tfp` — the real `tensorflow_probability`
     package is never loaded. (The `_tf.py:1133` comment about a sys.modules
     alias is stale — no such alias exists.)
   - jaxopt: only stale comments in `math/optimizer/*`.
   Run `uv remove tensorflow-probability jaxopt`; verify import still works.
   *Why:* two huge wheels (tfp pulls scipy/abs1 stacking); every dependency is
   a question mark for the next LLM.

3. **`_tf.py` public contract** — 1600-line flat shim that auto-exports every
   `jnp.*` name. Add a one-block header inventory: (a) TF-API compat names,
   (b) JAX-native names that mirror automatically, (c) known semantic gaps
   (RNG streams, GradientTape, string tensors). The `ponytail:` comments are
   already good — consolidate them into one readable contract block.
   *Why:* regenerability — a future LLM must know what's TF-shape vs JAX-native
   without reading all 1600 lines.

## Medium

4. **`_TfpOptimizer` static methods → module functions** — `_tf.py` defines
   `bfgs_minimize`/`lbfgs_minimize`/`converged_all` as staticmethods of a
   namespace class that has no state. Per the OOP guidance (no class for
   stateless logic), these belong as module-level functions in
   `math/optimizer/__init__.py` (already exist there). The `tfp` namespace in
   `_tf.py` should alias them, not re-wrap.

5. **`test_util` import churn** — 120 test files do
   `from tf_quant_finance._tf import test_util`. A re-export from
   `tf_quant_finance/__init__.py` or a single `tf_quant_finance.testing`
   module would centralize this. Low risk, high consistency win.

6. **`six.moves.range` → `range`** — `math/pde/grids.py`, `sobol_impl.py`,
   `halton_test.py` import `from six.moves import range` for Py2 compat that
   no longer exists. Replace with plain `range` (Python 3.12). Keep `six`
   only for the `@six.add_metaclass` in `ito_process.py` (or replace with
   `class X(metaclass=abc.ABCMeta)`).

## Lower

7. **`conftest.py` double try/except** — two defensive blocks; fine, but could
   be one `_try_init_jax_rocm()` call. Cosmetic.

8. **Stale comments referencing tfp.substrates.jax / jaxopt** —
   `conjugate_gradient.py:35`, `optimizer/__init__.py:17-18` describe a plan
   that no longer applies (scipy replaced both). Rewrite to state what IS
   there (scipy L-BFGS-B + backtracking CG).

9. **`_tf.py` local `import` hoisting** — already done in the ponytail pass;
   a few method-local `import tempfile`/`contextlib` remain in `TestCase`.
   Hoist to module top for consistency.

## Explicitly NOT changing

- The shim approach itself (works; 263 files import `_tf as tf`).
- `shape_utils.py` TF-style API (142-test landmine; already subclassed to
  `TensorShape`).
- Test tolerances / expected RNG divergence (documented in MIGRATION_STATUS.md).
- The `while_loop` counter path (too hot; consolidation risk outweighs gain).
