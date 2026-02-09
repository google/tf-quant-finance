# Approximations for option prices in the Heston Model

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "approximations",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":asian_prices",
        ":calibration",
        ":european_option",
    ],
)

py_library(
    name = "european_option",
    srcs = ["european_option.py"],
    strict_deps = False,
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
    shard_count = 4,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    strict_deps = False,
    deps = [
        ":european_option",
        "//tf_quant_finance/math:gradient",
        "//tf_quant_finance/math/optimizer",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
    ],
)

py_test(
    name = "calibration_test",
    timeout = "long",
    srcs = ["calibration_test.py"],
    shard_count = 12,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "asian_prices",
    srcs = ["asian_prices.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "asian_prices_test",
    srcs = ["asian_prices_test.py"],
    strict_deps = False,
    deps = [
        ":asian_prices",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
