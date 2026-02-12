# Local stochastic volatility model.

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "local_stochastic_volatility",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":local_stochastic_volatility_model",
        ":lsv_variance_model",
    ],
)

py_library(
    name = "lsv_variance_model",
    srcs = ["lsv_variance_model.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/models:generic_ito_process",
        # tensorflow dep,
    ],
)

py_library(
    name = "local_stochastic_volatility_model",
    srcs = ["local_stochastic_volatility_model.py"],
    strict_deps = False,
    deps = [
        ":lsv_variance_model",
        "//tf_quant_finance/datetime",
        "//tf_quant_finance/experimental/local_volatility",
        "//tf_quant_finance/experimental/pricing_platform/framework/market_data",
        "//tf_quant_finance/math",
        "//tf_quant_finance/models:generic_ito_process",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "local_stochastic_volatility_model_test",
    size = "medium",
    timeout = "long",
    srcs = ["local_stochastic_volatility_model_test.py"],
    shard_count = 12,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
    ],
)
