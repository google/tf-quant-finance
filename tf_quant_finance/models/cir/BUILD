# CIR model

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "cir",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":cir_model",
    ],
)

py_library(
    name = "cir_model",
    srcs = ["cir_model.py"],
    deps = [
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "cir_test",
    size = "medium",
    timeout = "long",
    srcs = ["cir_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":cir_model",
        "//tf_quant_finance",
        # test util,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
