# Ops related to random or quasi random number generation.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "random_ops",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":multivariate_normal",
        ":stateless",
        ":uniform",
        "//tf_quant_finance/math/random_ops/halton",
        "//tf_quant_finance/math/random_ops/sobol",
    ],
)

py_library(
    name = "multivariate_normal",
    srcs = ["multivariate_normal.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math/random_ops/halton",
        "//tf_quant_finance/math/random_ops/sobol",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "uniform",
    srcs = ["uniform.py"],
    strict_deps = False,
    deps = [
        ":multivariate_normal",
        "//tf_quant_finance/math/random_ops/halton",
        "//tf_quant_finance/math/random_ops/sobol",
        # tensorflow dep,
    ],
)

py_library(
    name = "stateless",
    srcs = [
        "stateless.py",
    ],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "multivariate_normal_test",
    size = "small",
    timeout = "moderate",
    srcs = ["multivariate_normal_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "stateless_test",
    size = "small",
    srcs = ["stateless_test.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "uniform_test",
    size = "small",
    srcs = ["uniform_test.py"],
    shard_count = 3,
    strict_deps = False,
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
