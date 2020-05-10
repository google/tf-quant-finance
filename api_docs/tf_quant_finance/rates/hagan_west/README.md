# Hagan West Algorithm


This module contains an implementation of the Hagan West procedure for
bootstrapping and interpolating interest rate curves. The algorithm applies
quadratic interpolation to the forward curve and the rate curve is computed
by integrating the interpolated forward curve.

The current implementation only supports building curves from bonds. The swap
curve building will be added shortly.

#TODO(b/140370128): Support swap curve building.
#TODO(b/140370679): Add amelioration and positivity constraints.


## References

[1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve
  Construction. Applied Mathematical Finance. Vol 13, No. 2, pp 89-129.
  June 2006.
https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction

[2]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.
Wilmott Magazine, pp. 70-81. May 2008.
https://www.researchgate.net/profile/Patrick_Hagan3/publication/228463045_Methods_for_constructing_a_yield_curve/links/54db8cda0cf23fe133ad4d01.pdf
