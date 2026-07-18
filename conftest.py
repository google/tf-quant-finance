"""Pytest bootstrap shared by both the TF baseline (.venv-tf) and JAX (.venv).

- Activates the pip-installed ROCm runtime (gfx1151) before JAX imports it, so the
  GPU backend registers without a hand-set LD_LIBRARY_PATH.
- Enables float64 globally (quant finance needs it; JAX defaults to float32).

Both blocks are defensive: if the package isn't installed (e.g. the TF baseline env
has no jax/rocm_sdk), they no-op so TF tests are unaffected.
"""
import os


def _try_init_jax_rocm():
    try:
        import rocm_sdk
        rocm_sdk.initialize_process()
    except Exception:
        pass  # not a JAX env, or rocm_sdk absent — fine
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass  # not a JAX env


_try_init_jax_rocm()
