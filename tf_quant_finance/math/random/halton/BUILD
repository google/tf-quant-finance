# Halton sequences

licenses(["notice"])

# [internal] load python3.bzl
# [internal] load pytype.bzl

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "halton",
    srcs = ["__init__.py"],
    deps = [
        ":halton_impl",
        # tensorflow dep,
    ],
)

# pytype
py_library(
    name = "halton_impl",
    srcs = ["halton_impl.py"],
    deps = [
        "//tf_quant_finance/math/random:stateless",
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "halton_test",
    size = "medium",
    timeout = "long",
    srcs = ["halton_test.py"],
    shard_count = 2,
    deps = [
        ":halton",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)

py_test(
    name = "halton_long_test",
    size = "medium",
    timeout = "eternal",
    srcs = ["halton_long_test.py"],
    shard_count = 5,
    deps = [
        ":halton",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
