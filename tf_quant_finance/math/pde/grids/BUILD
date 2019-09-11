# Grids for the PDE solver.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "grids",
    srcs = ["__init__.py"],
    deps = [
        ":grids_impl",
    ],
)

py_library(
    name = "grids_impl",
    srcs = ["grids_impl.py"],
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "grids_test",
    size = "medium",
    srcs = ["grids_test.py"],
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)
