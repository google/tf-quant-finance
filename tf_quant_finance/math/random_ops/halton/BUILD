# Halton sequences

# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "halton",
    srcs = ["__init__.py"],
    strict_deps = False,
    deps = [
        ":halton_impl",
        # tensorflow dep,
    ],
)

py_library(
    name = "halton_impl",
    srcs = ["halton_impl.py"],
    strict_deps = False,
    deps = [
        "//tf_quant_finance/math/random_ops:stateless",
        "//tf_quant_finance/types",
        "//tf_quant_finance/utils",
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
        # test util,
        "//tf_quant_finance/math",
        # numpy dep,
        # six dep,
        # tensorflow dep,
        # tensorflow_probability dep,
    ],
)
