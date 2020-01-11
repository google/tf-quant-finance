
package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "approximations",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":american_option",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "american_option",
    srcs = ["american_option.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        # numpy dep,
  ],
)

py_test(
    name = "american_option_test",
    srcs = ["american_option_test.py"],
    python_version = "PY3",
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # tensorflow dep,
    ],
)