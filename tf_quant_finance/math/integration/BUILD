# Numerical integration methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "integration",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":gauss_constants",
        ":gauss_legendre",
        ":integrate",
        ":simpson",
    ],
)

py_library(
    name = "integrate",
    srcs = [
        "integrate.py",
    ],
    deps = [
        ":gauss_constants",
        ":gauss_legendre",
        ":simpson",
        # tensorflow dep,
    ],
)

py_library(
    name = "simpson",
    srcs = [
        "simpson.py",
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "gauss_legendre",
    srcs = [
        "gauss_legendre.py",
    ],
    deps = [
        ":gauss_constants",
        # tensorflow dep,
    ],
)

py_library(
    name = "gauss_constants",
    srcs = [
        "gauss_constants.py",
    ],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "integration_test",
    size = "medium",
    srcs = ["integration_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
