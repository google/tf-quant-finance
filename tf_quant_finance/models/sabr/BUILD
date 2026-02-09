# SABR model

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "sabr",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":sabr_model",
        "//tf_quant_finance/models/sabr/approximations",
    ],
)

py_library(
    name = "sabr_model",
    srcs = ["sabr_model.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/black_scholes:implied_vol_newton_root",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/math/root_search:newton",
        "//tf_quant_finance/models:euler_sampling",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "sabr_model_test",
    size = "medium",
    timeout = "long",
    srcs = ["sabr_model_test.py"],
    shard_count = 12,
    strict_deps = False,
    deps = [
        ":sabr_model",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
