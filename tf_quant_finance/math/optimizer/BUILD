# Optimization methods.

licenses(["notice"])

# [internal] load python3.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "optimizer",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":conjugate_gradient",
    ],
)

py_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    deps = [
        "//tf_quant_finance",
    ],
)

py_library(
    name = "conjugate_gradient",
    srcs = ["conjugate_gradient.py"],
    deps = [
        # attr dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "conjugate_gradient_test",
    size = "medium",
    timeout = "long",
    srcs = ["conjugate_gradient_test.py"],
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)
