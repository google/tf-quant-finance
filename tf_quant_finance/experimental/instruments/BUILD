licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "instruments",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":forward_rate_agreement",
        ":rate_curve",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "forward_rate_agreement",
    srcs = [
        "forward_rate_agreement.py",
    ],
    srcs_version = "PY3",
    deps = [
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "rate_curve",
    srcs = [
        "rate_curve.py",
    ],
    srcs_version = "PY3",
    deps = [
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "forward_rate_agreement_test",
    size = "small",
    timeout = "moderate",
    srcs = ["forward_rate_agreement_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)
