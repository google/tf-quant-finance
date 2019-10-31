# Black Scholes Tools

This module contains implementation of well-known results from the Black Scholes
model. This includes analytically known pricing expressions (or approximations)
for various derivatives as well as methods to compute implied volatility from
prices.

## Pricing

Currently, the following pricing formula are available:
  * Vanilla calls and puts.
  * Binary options.


## Utilities

A very common task in finance is to convert market prices of options into
implied volatilities. While there is no exact closed-form expression for this,
this module provides methods to compute approximations. The available methods
are:

  * A fast but approximate method to infer implied volatility based on the Polya
    approximation of the Normal CDF. This was proposed by Radiocic and Stefanica
    in Ref [1].
  * A more precise method based on Newton root finder. This method uses the
    Radiocic & Stefanica algorithm to initialize the root finder.


## References

  [1]: Dan Stefanica and Rados Radoicic. An explicit implied volatility formula.
    International Journal of Theoretical and Applied Finance,
    Vol. 20, no. 7, 2017.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494
