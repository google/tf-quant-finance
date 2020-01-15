licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "dates",
    srcs = [
        "__init__.py",
        "constants.py",
        "date_tensor.py",
        "date_utils.py",
        "periods.py",
        "tensor_wrapper.py",
    ],
    srcs_version = "PY3",
    deps = [
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

py_library(
    name = "test_data",
    srcs = [
        "test_data.py",
    ],
    srcs_version = "PY3",
    deps = [],
)
