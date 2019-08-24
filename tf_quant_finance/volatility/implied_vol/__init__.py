"""Implied volatility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_quant_finance.volatility.implied_vol.newton_root import implied_vol
from tf_quant_finance.volatility.implied_vol.polya_approx import implied_vol as polya_approximation

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    'implied_vol',
    'polya_approximation',
]

remove_undocumented(__name__, _allowed_symbols)
