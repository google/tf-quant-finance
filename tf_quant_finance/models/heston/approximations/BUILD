# Approximations for option prices in the Heston Model

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "approximations",
    srcs = ["__init__.py"],
    deps = [
        ":european_option",
    ],
)

py_library(
    name = "european_option",
    srcs = ["european_option.py"],
    deps = [
        "//tf_quant_finance/math/integration",
        "//tf_quant_finance/types",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "european_option_test",
    size = "small",
    srcs = ["european_option_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
