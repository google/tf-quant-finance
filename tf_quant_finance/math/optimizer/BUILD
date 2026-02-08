# Optimization methods.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "optimizer",
    srcs = ["__init__.py"],
    strict_deps = False,
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
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "conjugate_gradient",
    srcs = ["conjugate_gradient.py"],
    strict_deps = False,
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
