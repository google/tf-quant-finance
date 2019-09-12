# Halton sequences

licenses(["notice"])

package(default_visibility = ["//tf_quant_finance:__subpackages__"])

py_library(
    name = "halton",
    srcs = ["__init__.py"],
    deps = [
        ":halton_impl",
        # tensorflow dep,
    ],
)

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
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tf_quant_finance",
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
    python_version = "PY3",
    shard_count = 5,
    deps = [
        "//tf_quant_finance",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
