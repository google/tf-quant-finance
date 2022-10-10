# Numerical integration methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "integration",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":adaptive_update",
        ":gauss_constants",
        ":gauss_kronrod",
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
    name = "gauss_kronrod",
    srcs = [
        "gauss_kronrod.py",
    ],
    deps = [
        ":adaptive_update",
        ":gauss_constants",
        # tensorflow dep,
    ],
)

py_library(
    name = "adaptive_update",
    srcs = [
        "adaptive_update.py",
    ],
    deps = [
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
