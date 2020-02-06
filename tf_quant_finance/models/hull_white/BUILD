# Hull White model
package(default_visibility = ["//tf_quant_finance:__subpackages__"])

licenses(["notice"])

py_library(
    name = "hull_white",
    srcs = ["__init__.py"],
    deps = [
        ":one_factor",
        ":vector_hull_white",
    ],
)

py_library(
    name = "vector_hull_white",
    srcs = ["vector_hull_white.py"],
    deps = [
        "//tf_quant_finance/math:piecewise",
        "//tf_quant_finance/math/random_ops",
        "//tf_quant_finance/models:generic_ito_process",
        "//tf_quant_finance/models:utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "one_factor",
    srcs = ["one_factor.py"],
    deps = [
        ":vector_hull_white",
    ],
)

py_test(
    name = "hull_white_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["hull_white_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)
