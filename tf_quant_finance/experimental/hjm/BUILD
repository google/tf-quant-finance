# HJM model and its flavors.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "hjm",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":quasi_gaussian_hjm",
    ],
)

py_library(
    name = "quasi_gaussian_hjm",
    srcs = ["quasi_gaussian_hjm.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "quasi_gaussian_hjm_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["quasi_gaussian_hjm_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        ":quasi_gaussian_hjm",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
