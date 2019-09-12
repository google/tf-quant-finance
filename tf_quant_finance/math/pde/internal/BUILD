# Utility functions for pde_time_marching_schemes.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "internal",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pde_time_marching_schemes_internal",
        # tensorflow dep,
    ],
)

py_library(
    name = "pde_time_marching_schemes_internal",
    srcs = ["pde_time_marching_schemes_internal.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tf_quant_finance/math/pde/time_marching_schemes:pde_time_marching_scheme",
        # tensorflow dep,
    ],
)

py_test(
    name = "pde_time_marching_schemes_internal_test",
    size = "small",
    srcs = ["pde_time_marching_schemes_internal_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)
