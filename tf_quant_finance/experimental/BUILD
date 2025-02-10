# Experimental modules.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    deps = [
        ":io",
        "//tf_quant_finance/experimental/american_option_pricing",
        "//tf_quant_finance/experimental/finite_difference",
        "//tf_quant_finance/experimental/instruments",
        "//tf_quant_finance/experimental/local_stochastic_volatility",
        "//tf_quant_finance/experimental/local_volatility",
        "//tf_quant_finance/experimental/lsm_algorithm",
        "//tf_quant_finance/experimental/pricing_platform",
        "//tf_quant_finance/experimental/svi",
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
    data = [
        "//tf_quant_finance/experimental/finite_difference:docs",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "io_test",
    size = "small",
    srcs = ["io_test.py"],
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
