# PDE solver methods.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "pde",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":grid_stepper",
        ":pde_kernels",
        "//tf_quant_finance/math/pde/grids",
        "//tf_quant_finance/math/pde/time_marching_schemes",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "grid_stepper",
    srcs = ["grid_stepper.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tf_quant_finance/math/pde/grids",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "grid_stepper_test",
    size = "small",
    srcs = ["grid_stepper_test.py"],
    deps = [
        ":grid_stepper",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "pde_kernels",
    srcs = [
        "pde_kernels.py",
    ],
    deps = [
        "//tf_quant_finance/math/pde/time_marching_schemes",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "pde_kernels_test",
    size = "medium",
    srcs = ["pde_kernels_test.py"],
    shard_count = 3,
    deps = [
        "//tf_quant_finance/math",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
