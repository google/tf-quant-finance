# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "american_option_pricing",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":andersen_lake",
        ":common",
        ":exercise_boundary",
    ],
)

py_library(
    name = "exercise_boundary",
    srcs = ["exercise_boundary.py"],
    strict_deps = False,
    deps = [
        ":common",
        "//tf_quant_finance/math/integration",
        "//tf_quant_finance/math/interpolation/cubic",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "andersen_lake",
    srcs = ["andersen_lake.py"],
    strict_deps = False,
    deps = [
        ":common",
        ":exercise_boundary",
        "//tf_quant_finance/black_scholes",
        "//tf_quant_finance/math/integration",
        "//tf_quant_finance/types",
        # tensorflow dep,
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/types",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "exercise_boundary_test",
    size = "small",
    timeout = "moderate",
    srcs = ["exercise_boundary_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        ":common",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "andersen_lake_test",
    size = "small",
    timeout = "moderate",
    srcs = ["andersen_lake_test.py"],
    shard_count = 10,
    strict_deps = False,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
