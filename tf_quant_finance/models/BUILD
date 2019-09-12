# Tools to build Diffusion Models

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ito_process",
    ],
)

py_library(
    name = "ito_process",
    srcs = ["ito_process.py"],
    deps = [
        "//tf_quant_finance/math/random",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "ito_process_test",
    size = "small",
    timeout = "moderate",
    srcs = ["ito_process_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)
