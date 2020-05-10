# Tools to build Diffusion Models

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":brownian_motion",
        ":ito_process",
    ],
)

py_library(
    name = "ito_process",
    srcs = ["ito_process.py"],
    deps = [
        "//tf_quant_finance/math/random_ops",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "brownian_motion_utils",
    srcs = ["brownian_motion_utils.py"],
    deps = [
        "//tf_quant_finance/math:gradient",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "brownian_motion_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["brownian_motion_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":brownian_motion_utils",  # This dependency is deliberate. The bm utils will not be exported to top level TFF.
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "brownian_motion",
    srcs = ["brownian_motion.py"],
    deps = [
        ":brownian_motion_utils",
        ":ito_process",
        "//tf_quant_finance/math/random_ops",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "brownian_motion_test",
    size = "small",
    timeout = "moderate",
    srcs = ["brownian_motion_test.py"],
    python_version = "PY3",
    deps = [
        ":brownian_motion",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
