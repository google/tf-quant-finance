licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "instruments",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        "rates_common",
        ":eurodollar_futures",
        ":forward_rate_agreement",
        ":rate_curve",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "rates_common",
    srcs = [
        "rates_common.py",
    ],
    srcs_version = "PY3",
    deps = [
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
    name = "eurodollar_futures",
    srcs = [
        "eurodollar_futures.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":rates_common",
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

py_test(
    name = "eurodollar_futures_test",
    size = "small",
    timeout = "moderate",
    srcs = ["eurodollar_futures_test.py"],
    python_version = "PY3",
    deps = [
        ":instruments",
        "//tf_quant_finance",
        "//tf_quant_finance/experimental/dates",
        # test util,
        # tensorflow dep,
    ],
)
