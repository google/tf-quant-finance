# Local stochastic volatility model.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "local_stochastic_volatility",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":local_stochastic_volatility_model",
        ":lsv_variance_model",
    ],
)

py_library(
    name = "lsv_variance_model",
    srcs = ["lsv_variance_model.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/models:generic_ito_process",
        # tensorflow dep,
    ],
)

py_library(
    name = "local_stochastic_volatility_model",
    srcs = ["local_stochastic_volatility_model.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    shard_count = 12,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
    ],
)
