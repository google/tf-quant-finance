licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "american_option_pricing",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":exercise_boundary",
    ],
)

py_library(
    name = "exercise_boundary",
    srcs = ["exercise_boundary.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/integration",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "exercise_boundary_test",
    size = "small",
    srcs = ["exercise_boundary_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
