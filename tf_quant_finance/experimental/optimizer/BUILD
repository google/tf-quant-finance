# Optimization methods.

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "optimizer",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":levenberg_marquardt",
    ],
)

py_library(
    name = "levenberg_marquardt",
    srcs = ["levenberg_marquardt.py"],
    deps = [
        # attr dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "levenberg_marquardt_test",
    size = "medium",
    timeout = "long",
    srcs = ["levenberg_marquardt_test.py"],
    python_version = "PY3",
    shard_count = 10,
    deps = [
        "//tf_quant_finance",
        # test util,
        # numpy dep,
        # tensorflow dep,
    ],
)

filegroup(
    name = "docs",
    srcs = [
        "README.md",
    ],
)
