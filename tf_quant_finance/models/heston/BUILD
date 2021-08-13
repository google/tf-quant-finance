# Heston model

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "heston",
    srcs = ["__init__.py"],
    deps = [
        ":heston_model",
        "//tf_quant_finance/models/heston/approximations",
    ],
)

py_library(
    name = "heston_model",
    srcs = ["heston_model.py"],
    deps = [
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "heston_model_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["heston_model_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
