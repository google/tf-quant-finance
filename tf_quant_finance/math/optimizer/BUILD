# Optimization methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "optimizer",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":conjugate_gradient",
    ],
)

py_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "conjugate_gradient",
    srcs = ["conjugate_gradient.py"],
    deps = [
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
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
    python_version = "PY3",
    shard_count = 10,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
