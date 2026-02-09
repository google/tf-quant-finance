# Approximations for the option prices in the SABR Model.

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
        ":calibration",
        ":european_options",
        ":implied_volatility",
    ],
)

py_library(
    name = "european_options",
    srcs = ["european_options.py"],
    strict_deps = False,
    deps = [
        ":implied_volatility",
        # tensorflow dep,
    ],
)

py_test(
    name = "european_options_test",
    srcs = ["european_options_test.py"],
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
    name = "implied_volatility",
    srcs = ["implied_volatility.py"],
    strict_deps = False,
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "implied_volatility_test",
    srcs = ["implied_volatility_test.py"],
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
        ":european_options",
        ":implied_volatility",
        "//tf_quant_finance/black_scholes",
        "//tf_quant_finance/math:gradient",
        "//tf_quant_finance/math/optimizer",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
    ],
)

py_test(
    name = "calibration_test",
    timeout = "moderate",
    srcs = ["calibration_test.py"],
    shard_count = 20,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
