# Finite difference methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "finite_difference",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":methods",
    ],
)

py_library(
    name = "methods",
    srcs = ["methods.py"],
    srcs_version = "PY2AND3",
    deps = [
        # autograd dep,
        # pandas dep,
        # sobol_seq dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
