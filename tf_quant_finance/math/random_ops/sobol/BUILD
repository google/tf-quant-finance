# Targets for Sobol sequence generation.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "sobol",
    srcs = ["__init__.py"],
    deps = [
        ":sobol_impl",
        # tensorflow dep,
    ],
)

py_library(
    name = "sobol_impl",
    srcs = ["sobol_impl.py"],
    data = ["//third_party/sobol_data:new-joe-kuo-6.21201"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/types",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sobol_test",
    size = "medium",
    srcs = ["sobol_test.py"],
    python_version = "PY3",
    deps = [
        # test util,
        "//tf_quant_finance/math",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
