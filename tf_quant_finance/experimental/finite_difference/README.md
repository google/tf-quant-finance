# Approximating Derivatives

## Problem statement

Given a function $$f\colon\mathbb{R}^n \to \mathbb{R}$$. We know $$k$$ points
$$x_1, \dots, x_k \in \mathbb{R}^n$$, and the respective values $$f(x_i)$$ for
each. We want to compute approximations for $$\nabla f$$
and $$\nabla^2 f$$ evaluated at those same $$k$$ points.

We can write the Taylor series expansion for each point around each
other. Namely:

$$f(x_i) = f(x_j) + (x_i - x_j)^{\top} \nabla f (x_j)
  + \frac{1}{2}(x_i - x_j)^{\top}\{\nabla^2 f(x_j)\}(x_i - x_j) + \cdots$$

No. of equations: $$k^2 - k = k(k - 1)$$.

No. of unknowns:
$$k \left[n + \frac{n(n + 1)}{2}\right] = \frac{1}{2} kn(n + 3)$$.

### Notes

* If $$k = n + 1$$ then "the best" (?) we can do is approximate $$f$$ with the
  hyperplane $$h$$ going throguh the known $$k$$ points, i.e.
  $$h(x) = a \cdot x + b$$ with $$a$$ and $$b$$ chosen such that
  $$h(x_i) = f(x_i)$$.
  For example, if $$n = 2$$ and we have three points the best we can do is
  approximate $$f$$ with the plane going through those three points
  (if not colinear).

  From the form of the solution
  $$\nabla h = a$$ is constant and $$\nabla^2 h = 0$$.
  And the coefficients can be obtained by solving the system of equations
  $$\{f(x_i) = a\cdot x_i + b\}$$, namely:

$$\begin{bmatrix}f(x_1) \\ \vdots \\ f(x_{k-1}) \\ f(x_k)\end{bmatrix}
=\begin{bmatrix}
x_{1,1}   & \dots & x_{1,n}   & 1 \\
\vdots    &       & \vdots    & \vdots \\
x_{k-1,1} & \dots & x_{k-1,n} & 1 \\
x_{k,1}   & \dots & x_{k,n}   & 1
\end{bmatrix}
\begin{bmatrix}
a_1 \\ \vdots \\ a_n \\ b
\end{bmatrix}$$

* If $$k \leq n$$ we can only do worst than that.

  * For example if $$n = k = 2$$, a "reasonable" thing to do would be to
    project and solve along each dimension independently; i.e.
    $$\nabla h = \left(\frac{f_2 - f_1}{x_2 - x_1},
                  \frac{f_2 - f_1}{y_2 - y_1}\right)$$.

  * Is this a "reasonable" thing we can always do (project along each dimension
    and solve independently) even if we have more points?
    $$\to$$ Nope. Doesn't seem to work well.

  * For $$n = k = 2$$, wlog assume
    $$x_1 = (0,0), x_2 = (a, b), f(x_1) = 0, f(x_2) = z$$. Another
    reasonable thing to do is to approximate $$f$$ with a plane $$h$$ whose
    slope along the direction $$x_2$$ is $$z / |x_2|$$, and the slope along
    directions orthogonal to $$x_2$$ is $$0$$ (because we know nothing about
    that direction). It's not too hard from this to get:

    $$h(x, y) = z \frac{xa + yb}{a^2 + b^2} \qquad
    \nabla h = \left(\frac{za}{a^2 + b^2}, \frac{zb}{a^2 + b^2}\right)$$

  * Generalizing to higher dimensions: given $$X = \{x_1, \dots, x_k\}$$, we can
    try to find a hyperplane whose slope along (... something ... something
    ... "directions" in $$X$$),
    and the slope along directions orthogonal to $$X$$, i.e. directions $$v$$
    for which $$X^T v = 0$$, is $$0$$.


* If $$k = 1$$ the best (most reasonable) we can do is approximate with the
  constant $$h(x) = f(x_1)$$, where $$\nabla h = \nabla^2 h = 0$$.

* If $$k = n + 1 + \frac{n(n+1)}{2} = \frac{1}{2}(n^2 + 3n + 2)$$ can find
  solution with error $$O(w^3)$$ for first and second derivatives.

* If $$n + 1 < k < \frac{1}{2}(n^2 + 3n + 2)$$ then ...

## Appendix

### Finite central difference

$$\begin{align*}
\left(\frac{\partial f}{\partial x}\right) &\approx
  \frac{f_{i+1} - f_{i-1}}{2w} \\
\left(\frac{\partial^2 f}{\partial x^2}\right)_i &\approx
  \frac{f_{i+1} - 2f_i + f_{i-1}}{w^2} \\
\left(\frac{\partial^2 f}{\partial x \partial y}\right)_i &\approx
  \frac{f_{i+1,j+1} - f_{i+1,j-1} - f_{i-1,j+1} + f_{i-1,j-1}}{4w^2}
\end{align*}$$

### 1-D Taylor expansion

$$f(x) = f(a) + (x - a) f'(a) + \frac{(x - a)^2}{2!} f''(a) + \cdots
  = \sum_{k=0}^{\infty} \frac{(x - a)^k}{k!} f^{(k)}(a)$$
