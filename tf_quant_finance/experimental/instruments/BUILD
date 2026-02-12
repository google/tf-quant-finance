# Placeholder: load py_library
# Placeholder: load py_test

package(
    default_visibility = ["//tf_quant_finance:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "instruments",
    srcs = [
        "__init__.py",
    ],
    strict_deps = False,
    deps = [
        ":bond",
        ":cap_floor",
        ":cashflow_stream",
        ":cms_swap",
        ":eurodollar_futures",
        ":floating_rate_note",
        ":forward_rate_agreement",
        ":interest_rate_swap",
        ":overnight_index_linked_futures",
        ":rate_curve",
        ":rates_common",
        ":swaption",
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
    strict_deps = False,
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
    strict_deps = False,
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
    strict_deps = False,
    deps = [
        ":rates_common",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "overnight_index_linked_futures",
    srcs = [
        "overnight_index_linked_futures.py",
    ],
    strict_deps = False,
    deps = [
        ":rates_common",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "cashflow_stream",
    srcs = [
        "cashflow_stream.py",
    ],
    strict_deps = False,
    deps = [
        ":rates_common",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "interest_rate_swap",
    srcs = [
        "interest_rate_swap.py",
    ],
    strict_deps = False,
    deps = [
        ":cashflow_stream",
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
    strict_deps = False,
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
    strict_deps = False,
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
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_test(
    name = "overnight_index_linked_futures_test",
    size = "small",
    timeout = "moderate",
    srcs = ["overnight_index_linked_futures_test.py"],
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_test(
    name = "interest_rate_swap_test",
    size = "small",
    timeout = "moderate",
    srcs = ["interest_rate_swap_test.py"],
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_test(
    name = "cashflow_stream_test",
    size = "small",
    timeout = "moderate",
    srcs = ["cashflow_stream_test.py"],
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "bond",
    srcs = [
        "bond.py",
    ],
    strict_deps = False,
    deps = [
        ":cashflow_stream",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "bond_test",
    size = "small",
    timeout = "moderate",
    srcs = ["bond_test.py"],
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "swaption",
    srcs = [
        "swaption.py",
    ],
    strict_deps = False,
    deps = [
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "swaption_test",
    size = "small",
    timeout = "moderate",
    srcs = ["swaption_test.py"],
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "cap_floor",
    srcs = [
        "cap_floor.py",
    ],
    strict_deps = False,
    deps = [
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "cap_floor_test",
    size = "small",
    timeout = "moderate",
    srcs = ["cap_floor_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "floating_rate_note",
    srcs = [
        "floating_rate_note.py",
    ],
    strict_deps = False,
    deps = [
        ":cashflow_stream",
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "floating_rate_note_test",
    size = "small",
    timeout = "moderate",
    srcs = ["floating_rate_note_test.py"],
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_library(
    name = "cms_swap",
    srcs = [
        "cms_swap.py",
    ],
    strict_deps = False,
    deps = [
        # attr dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "cms_swap_test",
    size = "small",
    timeout = "long",
    srcs = ["cms_swap_test.py"],
    shard_count = 10,
    strict_deps = False,
    deps = [
        ":instruments",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)

py_test(
    name = "rate_curve_test",
    size = "small",
    timeout = "moderate",
    srcs = ["rate_curve_test.py"],
    strict_deps = False,
    deps = [
        ":rate_curve",
        "//tf_quant_finance",
        # test util,
        # tensorflow dep,
    ],
)
