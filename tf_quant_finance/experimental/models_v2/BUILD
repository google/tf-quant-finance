# Staging for the new models interface

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "models_v2",
    srcs = ["__init__.py"],
    deps = [
        ":euler_sampling",
        ":ito_process",
    ],
)

py_library(
    name = "ito_process",
    srcs = ["ito_process.py"],
    deps = [],
)

py_library(
    name = "generic_ito_process",
    srcs = ["generic_ito_process.py"],
    deps = [
        ":euler_sampling",
        ":ito_process",
        # tensorflow dep,
    ],
)

py_test(
    name = "generic_ito_process_test",
    size = "small",
    timeout = "moderate",
    srcs = ["generic_ito_process_test.py"],
    python_version = "PY3",
    deps = [
        ":generic_ito_process",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "euler_sampling",
    srcs = ["euler_sampling.py"],
    deps = [
        "//tf_quant_finance/math/random_ops",
        # tensorflow dep,
    ],
)

py_test(
    name = "euler_sampling_test",
    size = "small",
    timeout = "moderate",
    srcs = ["euler_sampling_test.py"],
    python_version = "PY3",
    deps = [
        ":euler_sampling",
        "//tf_quant_finance/math/random_ops",
        # numpy dep,
        # tensorflow dep,
    ],
)
