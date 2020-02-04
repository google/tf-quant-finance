licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "dates",
    srcs = [
        "__init__.py",
        "constants.py",
        "date_tensor.py",
        "date_utils.py",
        "holiday_calendar.py",
        "holiday_calendar_v2.py",
        "periods.py",
        "schedules.py",
        "tensor_wrapper.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":holiday_utils",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "date_tensor_test",
    size = "small",
    timeout = "moderate",
    srcs = ["date_tensor_test.py"],
    python_version = "PY3",
    deps = [
        ":dates",
        ":test_data",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "date_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["date_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":dates",
        ":test_data",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "holiday_calendar_test",
    size = "small",
    timeout = "moderate",
    srcs = ["holiday_calendar_test.py"],
    python_version = "PY3",
    deps = [
        ":dates",
        ":test_data",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "holiday_calendar_v2_test",
    size = "small",
    timeout = "moderate",
    srcs = ["holiday_calendar_v2_test.py"],
    python_version = "PY3",
    deps = [
        ":dates",
        ":test_data",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "schedules_test",
    size = "small",
    timeout = "moderate",
    srcs = ["schedules_test.py"],
    python_version = "PY3",
    deps = [
        ":dates",
        ":test_data",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "test_data",
    srcs = [
        "test_data.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":dates",
    ],
)

py_library(
    name = "holiday_utils",
    srcs = ["holiday_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tf_quant_finance/math/interpolation/linear",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)
