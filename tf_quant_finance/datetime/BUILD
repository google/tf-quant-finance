# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "datetime",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":constants",
        ":date_tensor",
        ":date_utils",
        ":daycounts",
        ":holiday_calendar",
        ":holiday_calendar_factory",
        ":schedules",
        "//tf_quant_finance/datetime/periods",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "date_utils",
    srcs = ["date_utils.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "tensor_wrapper",
    srcs = ["tensor_wrapper.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "date_tensor",
    srcs = ["date_tensor.py"],
    deps = [
        ":constants",
        ":date_utils",
        ":tensor_wrapper",
        "//tf_quant_finance/datetime/periods",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "bounded_holiday_calendar",
    srcs = ["bounded_holiday_calendar.py"],
    deps = [
        ":constants",
        ":date_tensor",
        "//tf_quant_finance/datetime/periods",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "holiday_utils",
    srcs = ["holiday_utils.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "unbounded_holiday_calendar",
    srcs = ["unbounded_holiday_calendar.py"],
    deps = [
        ":constants",
        ":date_tensor",
        ":date_utils",
        ":holiday_utils",
        "//tf_quant_finance/datetime/periods",
        # tensorflow dep,
    ],
)

py_library(
    name = "holiday_calendar",
    srcs = ["holiday_calendar.py"],
    deps = [
        ":constants",
        # tensorflow dep,
    ],
)

py_library(
    name = "holiday_calendar_factory",
    srcs = ["holiday_calendar_factory.py"],
    deps = [
        ":bounded_holiday_calendar",
        ":unbounded_holiday_calendar",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "schedules",
    srcs = ["schedules.py"],
    deps = [
        ":constants",
        ":date_tensor",
        # tensorflow dep,
    ],
)

py_test(
    name = "date_tensor_test",
    size = "small",
    timeout = "moderate",
    srcs = ["date_tensor_test.py"],
    deps = [
        ":test_data",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "date_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["date_utils_test.py"],
    deps = [
        ":test_data",
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "holiday_calendar_test",
    size = "small",
    timeout = "moderate",
    srcs = ["holiday_calendar_test.py"],
    shard_count = 3,
    deps = [
        ":bounded_holiday_calendar",
        ":test_data",
        ":unbounded_holiday_calendar",
        "//tf_quant_finance",
        # test util,
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
    shard_count = 4,
    deps = [
        ":test_data",
        "//tf_quant_finance",
        # test util,
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
    deps = [
        ":datetime",
    ],
)

py_library(
    name = "daycounts",
    srcs = ["daycounts.py"],
    deps = [
        ":date_tensor",
        ":date_utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "daycounts_test",
    size = "small",
    srcs = ["daycounts_test.py"],
    shard_count = 4,
    deps = [
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
