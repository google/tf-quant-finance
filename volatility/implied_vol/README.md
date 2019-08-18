# Methods to compute implied volatility from prices.

This directory contains methods to infer implied volatility given market prices.
Black-Scholes value for a vanilla call option is:

$$\begin{equation}
C = S_0 N(d_1) - e^{-rT} K N(d_2)  \label{eqn1}
\end{equation}$$

with:

$$\begin{eqnarray}
d_1 &=& \frac{1}{\sigma \sqrt{T}} \left[ \ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2 }{2}\right) T\right] \\
d_2 &=& d_1 - \sigma \sqrt{T}
\end{eqnarray}$$

where:

  * $$S_0$$ is the price of the underlying asset at time 0.
  * $$K$$ is the strike price of the option.
  * $$T$$ is the expiry time of the option.
  * $$r$$ is the risk free interest that applies to time $$T$$.
  * $$\sigma$$ is the volatility that applies to time $$T$$.

While the Black Scholes model makes a number of assumptions which are not borne
out in the market (e.g. log normal asset prices), it is still an extremely
useful reference. Given an observed market price $$C_*$$, the *implied*
volatility is the value $$\sigma_*$$ such that the Black Scholes price with this
volatility is equal to the observed market price. Assuming no arbitrage
conditions hold for the observed price, it is always possible to find such a
value by inverting the formula in Eq. [\ref{eqn1}] for $$\sigma$$ given all
other variables.

This module contains functions to solve this problem.

