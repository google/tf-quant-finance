# Targets for Sobol sequence generation.

licenses(["notice"])

# [internal] load python3.bzl
# [internal] load pytype.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "sobol",
    srcs = ["__init__.py"],
    deps = [
        ":sobol_impl",
        # tensorflow dep,
    ],
)

# pytype
py_library(
    name = "sobol_impl",
    srcs = ["sobol_impl.py"],
    data = ["//third_party/sobol_data:new-joe-kuo-6.21201"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sobol_test",
    size = "medium",
    srcs = ["sobol_test.py"],
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
