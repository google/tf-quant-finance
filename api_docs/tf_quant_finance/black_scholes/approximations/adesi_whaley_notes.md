# Adesi Whaley approximation for American Options

## General Setup

Let the price of an American option (either a call or a put) be denoted $$A(S,
T)$$ where $$S$$ is the spot price of the underlier and $$T$$ is the time to
expiry. Let the corresponding European option price be $$a(S, T)$$. The early
exercise premium $$\rho(S, T)$$ is defined as:

$$\begin{equation}
\rho(S, T) = A(S, T) - a(S, T) \label{premium}
\end{equation}$$

Recall the expressions for the European call and put option prices:

$$
\newcommand{\N}{\mathcal{N}}
\begin{equation}
a(S,T) = \left\{\begin{array}{cc}
Se^{-dT} \N(d_1) - K e^{-rT} \N(d_2) \equiv c(S,T), && (Call) \\
Ke^{-rT} \N(-d_2) - S e^{-dT} \N(-d_1) \equiv p(S,T), && (Put) \\
\end{array} \right.
\end{equation}$$

with:

$$\begin{eqnarray}
d_1 &=& \frac{1}{\sigma \sqrt{T}} \left(\ln \frac{S}{K} + \left[r-d+\frac{\sigma^2}{2}\right]T\right) \\
d_2 &=& \frac{1}{\sigma \sqrt{T}} \left(\ln \frac{S}{K} + \left[r-d-\frac{\sigma^2}{2}\right]T\right) \\
\end{eqnarray}$$

As both the European and the American option prices satisfy the Black Scholes
equation, by linearity, so does the premium. In terms of the time to expiry
$$\tau =T - t$$ the equation is:

$$\begin{equation}
\frac{\alpha}{r}\frac{\partial \rho}{\partial \tau} =
  S^2 \frac{\partial^2 \rho}{\partial S^2}  +
  \beta S \frac{\partial \rho}{\partial S} - \alpha \rho
\end{equation}$$

where,

$$\begin{equation}
\alpha = \frac{2 r}{\sigma^2}, \ \beta = \frac{2(r - d)}{\sigma^2}
\end{equation}$$

$$r$$ is the risk free rate, $$d$$ is the (continuous) dividend rate and
$$\sigma$$ is the constant volatility. Following [1], we perform a change of
variables as follows:

$$\begin{equation}
u = 1 - e^{-r\tau}, \ \rho(S, \tau) = u F(S, u)
\end{equation}$$

Note that, at $$\tau = 0$$ (i.e. at expiry), $$u = 0$$ and hence the functional
form above for the premium vanishes at expiry (assuming that $$F$$ is finite
there). In the new variables, the time derivative can be expressed as:

$$\begin{equation}
\frac{\partial \rho}{\partial \tau} =
  r(1-u) \left( F + u \frac{\partial F}{\partial u}\right)
\end{equation}$$

The full equation can be simplified to:

$$\begin{equation}
S^2 \partial_S^2 F + \beta S \partial_S F - \frac{\alpha}{u} F
  - \alpha (1-u)\partial_u F = 0
\end{equation}$$

The approximation of Baron-Adesi and Whaley consists of setting the last term in
the equation above to zero. This is reasonable approximation because in the
limit of very short expiry, $$u \rightarrow 0$$, we expect $$\partial_u F = 0$$.
Similarly in the limit of very long expiry (i.e. $$u \rightarrow 1$$), the
coefficient of that term $$(1-u)$$ approaches 0. Dropping this term, the
equation simplifies to:

$$\begin{equation}
S^2 \frac{d^2F}{dS^2} + \beta S \frac{dF}{dS} - \frac{\alpha}{u} F = 0
\end{equation}$$

This equation is homogenous in $$S$$ so the solutions are of the form, $$k S^p$$
Plugging this form in the equation, we find a quadratic equation for the power
coefficient $$p$$:

$$\begin{equation}
p^2 + (\beta - 1) p - \frac{\alpha}{u} = 0
\end{equation}$$

with solutions:

$$\begin{equation}
p_{\pm} = \frac{1 - \beta \pm \sqrt{(\beta-1)^2 + 4 \alpha / u}}{2} \label{powers}
\end{equation}$$

Note that the root $$p_+$$ is positive while $$p_-$$ is negative. The rest of
the discussion needs to be specialized to calls and puts. We treat the call
options in detail and quote the corresponding results for the put options.

## Call Options

The root $$p_{+}$$ is the only positive root. The solution for the early
exercise premium $$ukS^p$$ diverges as $$S\rightarrow 0 $$ with the negative
root. For call options, we know that both American and European calls are worth
zero at $$S=0$$ so we must choose the positive root:

$$\begin{equation}
F(S, u) = k S^{p_+}
\end{equation}$$

To fix the constant $$k$$, we impose the free boundary condition at the exercise
boundary where the option must be worth its intrinsic value.
Assume there exists a value of the spot $$S_{*}$$ such that for $$S >
S_{*}$$, early exercise is optimal. At this point, the value of the American
option must be equal to its intrinsic value $$S_{*} - K$$. This leads to the
condition:

$$\begin{equation}
S_* - K = c(S_*, T) + k (1-e^{-rT}) S_{*}^{p_{+}} \label{valuematch}
\end{equation}$$

which implies:

$$\begin{equation}
k = \frac{S_* - K - c(S_*, T)}{(1-e^{-rT}) S_{*}^{p_{+}}} \label{constant}
\end{equation}$$

While the previous condition ensures that the solution is continuous in $$S$$,
we also need the first derivative of [$$\ref{valuematch}$$] to be continuous
(in financial terms, we want delta to not jump suddenly across the boundary).
This leads to a second equation which can be used to find the exercise boundary
$$S_*$$.


$$\begin{equation}
1 = e^{-dT} \N(d_1) + \frac{(S_* - K - c(S_*, T))}{ S_{*}} p_{+} \label{temp1}
\end{equation}$$

It is convenient to define the right hand side of the above equation as a
function:

$$\begin{equation}
H(S) = e^{-dT} \N(d_1(S)) + \frac{(S - K - c(S, T))}{S} p_{+} \label{hdef}
\end{equation}$$

Equation [$$\ref{temp1}$$] is expressed as:

$$\begin{equation}
H(S_*) = 1 \label{tosolve}
\end{equation}$$

The exercise boundary is found by solving Eq. ($$\ref{tosolve}$$). A few
limiting cases are important. Firstly, if the dividends are zero, it is easy to
see that there is no solution to the equation (i.e. early exercise is never
optimal). This follows from the well known
bound on the European call option price: $$c(S, T) > S - K e^{-rT} >= S - K$$.
Hence,
the second term in Eq. ($$\ref{hdef}$$) is always negative. The first term is
the call option delta monotonically approaches 1 as $$S \rightarrow \infty$$.
Hence, the sum of the two terms (one always less than one and the other
negative) cannot equal 1 for any finite value of $$S$$. A similar argument (but
a little more tricky) shows that this is also true if the dividends are
negative.

For a positive dividend, the equation always has a solution. To show this, it is
sufficient to show that the asymptotic value of the function $$H(S)$$ is greater
than 1, that $$H(S)$$ is negative for
sufficiently small $$S$$ and that $$H(S)$$ is monotonically increasing. That
$$H(S)$$ is negative for small $$S$$ is seen by considering the limit of
$$S\rightarrow 0$$. As $$S\rightarrow 0$$, $$\N(d_1), c(S,T) \rightarrow 0$$
and $$H(S) \rightarrow -\frac{Kp_+}{S} < 0$$ (because $$p_+ >0$$).
The gradient of $$H(S)$$ with respect to $$S$$ is straightforward to compute
and the result is:

$$\begin{equation}
\frac{dH(S)}{dS} = \frac{e^{-dT} n(d_1) }{S\sigma \sqrt{T}} + \frac{Kp_+(1-e^{-rT} \N(d_2))}{S^2}
\end{equation}$$

which is clearly positive. We are finally left with having to prove that
$$H(S)$$ is $$>1$$ as $$S \rightarrow \infty$$.

To prove this we first need to prove an intermediate result that $$p_+ > 1$$.

**Claim 1:** $$p_+ > 1$$ if either ($$r \geq 0$$, $$d > 0$$) or ($$r > 0$$, $$d
\geq 0$$) and $$T > 0$$. If $$r = d = 0$$ then $$p_+=1$$.

**Proof:** The last part of the claim in obvious from the expression for
$$p_+$$. We prove the first case. Suppose $$r \geq 0$$, $$d > 0$$, then it
follows:

$$\begin{equation}
d > 0 \implies \beta < \alpha
\end{equation}$$

We also have (for any value of $$r$$):

$$\begin{equation}
u = 1 - e^{-rT} \leq 1
\end{equation}$$

and with $$r\geq 0$$:

$$\begin{equation}
u \geq 0
\end{equation}$$

Combining $$\beta < \alpha$$ with $$u < 1$$ we get:

$$\begin{eqnarray}
\beta < &\alpha& < \frac{\alpha}{u} \\
4\beta &<& \frac{4\alpha}{u} \\
2\beta &<& \frac{4\alpha}{u}-2\beta \\
1 + \beta^2 + 2\beta &<& 1 + \beta^2 -2\beta + \frac{4\alpha}{u} \\
(1+\beta)^2 &<& (1-\beta)^2 + \frac{4\alpha}{u}
\end{eqnarray}$$

Plugging this last inequality in the numerator of the expression for $$p_+$$:

$$\begin{eqnarray}
(1 - \beta) + \sqrt{(1 - \beta)^2 +\frac{4 \alpha}{u}} &>& (1-\beta) + \sqrt{(1 + \beta)^2} \\
\implies (1 - \beta) + \sqrt{(1 - \beta)^2 + \frac{4 \alpha}{u}} &>& 1 - \beta + |1 + \beta| \geq 2
\end{eqnarray}$$

Hence, we have the claim that $$p_+ > 1$$ if $$d > 0$$. Now consider $$d=0$$
with $$r > 0$$. Then $$0 < \alpha = \beta$$ and $$0 < u < 1$$:

$$\begin{equation}
2p_+ = (1-\alpha) + \sqrt{(1 - \alpha)^2 + \frac{4 \alpha}{u}} > 2
\end{equation}$$

This proves the claim. $$\square$$

TODO: Prove that the above is true even if $$r<0$$.

Now we can prove the main claim.

**Claim 2:** The limit of $$H(S)$$ in Eq [$$\ref{tosolve}$$] as $$S\rightarrow
\infty$$ is greater than 1 if $$d > 0$$ and $$T > 0$$

**Proof:** From the definition of $$H(S)$$ (plugging in the definitions of the
delta and the European call option price):

$$
\begin{equation}
H(S) = e^{-dT} \N(d_1) + \frac{(S - K - S e^{-dT} \N(d_1) + K e^{-rT} \N(d_2))}{ S} p_{+}
\end{equation}$$

Expanding the second term:

$$
\begin{equation}
=e^{-dT} \N(d_1) + p_+(1 - e^{-dT} \N(d_1)) -p_+ (1 - e^{-rT} \N(d_2))\frac{K}{S} \label{expanded}
\end{equation}
$$

In limit of $$S \rightarrow \infty$$, $$d_{1, 2} \rightarrow \infty$$ so we get:

$$
\begin{equation}
\lim\limits_{S \to \infty} H(S) = e^{-dT} + (1 - e^{-dT}) p_+
\end{equation}$$

The claimed result follows from the previous proposition that $$p_+>1$$ as
follows (which holds true whenever $$d > 0$$):

$$ \begin{eqnarray} p_+ &>& 1 \\
(1 - e^{-dT}) p_+ &>& 1 - e^{-dT} \\ e^{-dT} + (1 - e^{-dT}) p_+ &>& 1 \\
\end{eqnarray}$$

Note that for $$d = 0$$, the result fails because while $$p_+ > 1$$,
$$(1-e^{-dT})$$ is zero. $$\square$$

Next, we derive a useful lower bound on the exercise boundary.
Consider the subleading term of the $$H(S)$$.

**Claim 3:** $$H(S)$$ is bounded above by the following function:

$$\begin{equation}
h(S) = e^{-dT} + p_+ (1 - e^{-dT}) - p_+(1 - e^{-rT})\frac{K}{S}
\end{equation}$$

**Proof:** Let $$D(S) = h(S) - H(S)$$. We need to prove that $$D(S) \geq 0$$ for
all values of $$S>K$$. Using $$\N(-x) = 1 - N(x)$$:

$$\begin{equation}
D(S) = e^{-dT} \N(-d_1) - p_+ e^{-dT} \N(-d_1) + p_+ e^{-rT}\N(-d_2)\frac{K}{S}
\end{equation}$$

This expression can be rearranged into a nicer form:

$$\begin{eqnarray}
D(S) &=& e^{-dT} \N(-d_1) + \frac{p_+}{S} \left[ Ke^{-rT}\N(-d_2) - S e^{-dT} \N(-d_1)\right] \\
&=& e^{-dT} \N(-d_1) + \frac{p(S, T)}{S}p_+
\end{eqnarray}$$

where $$p(S, T)$$ is the price of a European put option. Given that $$p_+ > 0$$,
it follows from the positivity of the put option price that $$D(S) > 0$$.
$$\square$$

From the functional form of $$h(S)$$, it is also clear that $$h(S)$$ is
monotonically increasing.

**Claim 4:** For all $$S_1, S_2 > K$$, such that $$S_1 > S_2$$, $$h(S_1) <
h(S_2)$$ if $$r > 0$$. $$\square$$

It is also worth noting that, by construction, $$D(S) \rightarrow 0$$ as
$$S\rightarrow \infty$$. Hence the upper bound $$g(S)$$ becomes tight
asymptotically. Finally, we have the following result:

**Claim 5:** If $$r > 0$$, the exercise boundary price $$S_*$$ is bounded below
by:

$$\begin{equation}
S_b = \frac{Kp_+(1-e^{-rT})}{( p_+ -1)(1 - e^{-dT})}
\end{equation}$$

**Proof:** Note that $$S_b$$ is the solution to the equation $$h(S) = 1$$. From
Claim (3) we know that $$h(S) > H(S)$$. Furthermore, both $$H(S)$$ and $$h(S)$$
are monotonically increasing. We prove the result by contradiction. Suppose for
some parameter choice $$S_* < S_b$$. By definition of $$S_*$$ and $$S_b$$,
$$h(S_b) = H(S_*) = 1$$. From monotonicity of $$h(S)$$ and the assumption that
$$S_b > S_*$$, we must have $$1=h(S_b) > h(S_*)$$. Hence, $$H(S_*) > h(S_*)$$
which contradicts the result in Claim (3) that $$h(S)$$ is an upper bound for
$$H(S)$$ for all values of $$S$$ if $$r>0$$. $$\square$$

Note that this bound is not particularly useful in the case that $$r \leq 0$$.
However, for the "regular" case of $$r>0$$ it does provide a very useful
starting point for finding the solution of Eq $$[\ref{tosolve}]$$. 

## Put Options

We can mirror the previous section for the put options. The main equations (in
obvious notation) are:

$$\begin{eqnarray}
\rho(S, T) &=& P(S, T) - p(S, T) \\
K - S_* &=& p(S_*, T) + k (1-e^{-rT}) S_{*}^{p_{-}} \\
k &=& \frac{K - S_* - p(S_*, T)}{(1-e^{-rT}) S_{*}^{p_{-}}} \\
-1 &=& -e^{-dT} \N(-d_1) + \frac{(K - S_* - p(S_*, T))}{ S_{*}} p_{-} \equiv G(S_*) \label{tosolveput}\\
\end{eqnarray}$$

In order, these correspond to Eqns $$[\ref{premium}, \ref{valuematch},
\ref{constant}, \ref{tosolve}]$$ from the previous section.

This time we are interested in the limit of $$S \rightarrow 0$$. We need the
following results:

**Claim 6:** $$p_- < 0$$ for any value of $$r$$ and $$d$$.

**Claim 7:** The following limits hold:

$$\begin{equation}
\lim\limits_{S\rightarrow 0} G(S) = \left\{
\begin{array}{cc}
-\infty && r > 0 \\
\infty && r < 0 \\
-e^{-dt} && r = 0
\end{array}
\right.
\end{equation}$$

and

$$\begin{equation}
\lim\limits_{S\rightarrow \infty} G(S) = -p_- > 0
\end{equation}$$

From these two results, we conclude that $$G(S) = -1$$ always has a solution
when $$r > 0$$. If $$r=0$$, then it doesn't have a solution unless $$d\leq 0$$.
If $$r < 0$$, we don't have a definite conclusion.

Finally, we have the following approximation for a starting point:

**Claim 8:** $$G(S)$$ is bounded below by:

$$\begin{equation}
g(S) = -e^{-dT} - p_- (1 - e^{-dT}) + p_- (1 - e^{-rT})\frac{K}{S}
\end{equation}$$

**Proof:** Expanding $$G(S)$$ (and plugging in the known formula for the
European put option price):

$$\begin{equation}
G(S) = -e^{-dT} \N(-d_1) - p_- (1 - e^{-dT}\N(-d_1)) + p_- (1 - e^{-rT}\N(-d_2))\frac{K}{S}
\end{equation}$$

As before, define the difference function $$D(S) = G(S) - g(S)$$ and simplify:

$$\begin{equation}
D(S) = e^{-dT} \N(d_1) - \frac{p_-}{S}\left[ S e^{-dT}\N(d_1) - K e^{-rT}\N(d_2)\right]
\end{equation}$$

As $$p_- < 0$$ and term in the bracket is the call option price, we conclude
that $$D(S) > 0$$. $$\square$$

**Claim 9:** The exercise price for an American put option is bounded above by:

$$\begin{equation}
S_b = \frac{K p_- (1 - e^{-rT})}{(p_--1)(1 - e^{-dT})}
\end{equation}$$

## Adesi-Whaley Starting Point

Adesi and Whaley also provide an approximation to the boundary which while
coming without any guarantees on bounds, may still be useful as a starting point
for a root search. We quote their result (for details, see Ref [1]):

$$\begin{equation}
S_{\pm} = Ke^{h_{\pm}} + S^{\infty}_{\pm}(1-e^{h_{\pm}})
\end{equation}$$

with the definitions:

$$\begin{eqnarray}
p_{\pm}^{\infty} &=& \frac{1 - \beta \pm \sqrt{(\beta-1)^2 + 4 \alpha}}{2} \\
S^{\infty}_{\pm} &=& \frac{K p_{\pm}^{\infty}}{p_{\pm}^{\infty} - 1} \\
h_{\pm} &=& ((r-d) T \pm 2 \sigma \sqrt{T}) \frac{K}{K - S^{\infty}_{\pm}}
\end{eqnarray}$$

$$S_+$$ and $$S_-$$ are the suggested starting points for calls and puts
respectively.


## Root Finding

As we will be batching put and call options together, it is useful to combine
the expressions for calls and puts into one equation.
The equations we need to solve are $$H(S) = 1$$ and $$G(S) = -1$$ with
definitons in Eqns [$$\ref{tosolve}$$] and [$$\ref{tosolveput}$$]. Define:

$$\begin{eqnarray}
K_{+}(S) &=& S(H(S) - 1) \\
K_{-}(S) &=& S(G(S) + 1)
\end{eqnarray}$$

Some algebraic manipulation shows that $$K_{\pm}(S)$$ has the following symmetric
form:

$$\begin{equation}
K_{\pm}(S)=S(p_{\pm}-1)(1-e^{-dT}\N(\pm d_1)) - p_{\pm} (1-e^{-rT} \N(\pm d_2))K
\end{equation}$$

And we need to find the root of $$K_{\pm}$$. This is done iteratively. Let
$$S_i$$ be the root estimate at the i'th iteration. Then
$$S_{i+1} = S_{i}+\delta S_{i}$$ and:

$$\begin{equation}
\delta S_{i} = -\frac{K_{\pm}(S_i)}{K'_{\pm}(S_i)}
\end{equation}$$

where $$K'_{\pm}$$ is the derivative with respect to $$S$$. A little bit of
algebra along with the identity ($$n(x)$$ is the standard normal PDF):

$$\begin{equation}
\frac{n(d_1)}{n(d_2)} = \frac{K}{S} e^{-(r-d)T}
\end{equation}$$

yields the following expression for the derivative:

$$\begin{equation}
K'_{\pm}(S) = (p_{\pm}-1)[1-e^{-dT} \N(\pm d_1)] \pm \frac{e^{-dT}n(d_1)}{\sigma\sqrt{T}}
\end{equation}$$

Given the bounds on $$p_{\pm}$$ from the previous sections, it is clear that the
derivative is always positive for $$K_+$$ and always negative for $$K_-$$. Hence
the two functions are monotonic. This implies that given a current value of the
spot $$S_0$$, for a call, if $$K_+(S) > 0$$ then we know early exercise is
optimal and we can avoid the root finding (i.e. we don't need the value of
$$S_*$$). Similarly, for a put if $$K_-(S) > 0$$, we are in early exercise
region and we can avoid the root finding.

## References

[1] Baron-Adesi, Whaley, Efficient Analytic Approximation of American Option
Values, The Journal of Finance, Vol XLII, No. 2, June 1987
https://deriscope.com/docs/Barone_Adesi_Whaley_1987.pdf
