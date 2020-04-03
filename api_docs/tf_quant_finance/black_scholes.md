<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.black_scholes" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_quant_finance.black_scholes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/black_scholes/__init__.py">View source</a>



TensorFlow Quantitative Finance volatility surfaces and vanilla options.



## Modules

[`approximations`](../tf_quant_finance/black_scholes/approximations.md) module: Approximations to the black scholes formula.

## Classes

[`class ImpliedVolMethod`](../tf_quant_finance/black_scholes/ImpliedVolMethod.md): Implied volatility methods.

## Functions

[`binary_price(...)`](../tf_quant_finance/black_scholes/binary_price.md): Computes the Black Scholes price for a batch of binary call or put options.

[`brownian_bridge_double(...)`](../tf_quant_finance/black_scholes/brownian_bridge_double.md): Computes probability of not touching the barriers for a 1D Brownian Bridge.

[`brownian_bridge_single(...)`](../tf_quant_finance/black_scholes/brownian_bridge_single.md): Computes proba of not touching the barrier for a 1D Brownian Bridge.

[`implied_vol(...)`](../tf_quant_finance/black_scholes/implied_vol.md): Finds the implied volatilities of options under the Black Scholes model.

[`implied_vol_approx(...)`](../tf_quant_finance/black_scholes/implied_vol_approx.md): Approximates the implied vol using the Stefanica-Radiocic algorithm.

[`implied_vol_newton(...)`](../tf_quant_finance/black_scholes/implied_vol_newton.md): Computes implied volatilities from given call or put option prices.

[`option_price(...)`](../tf_quant_finance/black_scholes/option_price.md): Computes the Black Scholes price for a batch of call or put options.

[`option_price_binomial(...)`](../tf_quant_finance/black_scholes/option_price_binomial.md): Computes the BS price for a batch of European or American options.

