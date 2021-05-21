<!-- 
After updating, convert this document into pde_solvers.pdf as follows:

pandoc --toc --highlight-style=tango --variable urlcolor=blue -o pde_solvers.pdf pde_solvers.md
-->

---
title: PDE Solvers in Tensorflow Finance
geometry: margin=3cm
---

\renewcommand{\b}{\mathbf}
\renewcommand{\d}{\partial}

# Introduction

The `math.pde` package provides ops to compute numerical solutions of
linear parabolic second order partial differential equations.

We currently support equations of the following form:

\begin{equation}
    \frac{\d V}{\d t} +
    \sum\limits_{i,j=1}^N a_{ij}(\b x, t) \frac{\d^2}{\d x_i \d x_j}
    \left[A_{ij}(\b x, t) V\right] +
    \sum\limits_{i=1}^N b_i(\b x, t) \frac{\d}{\d x_i}
    \left[B_{i}(\b x, t) V\right] +
     c(\b x, t) V = 0.
\end{equation}

Given $V(\b x, t_0)$, the solver approximates $V(\b x, t_1)$. The solver can
go both forward ($t_1 > t_0$) and backward ($t_1 < t_0$) in time.

This includes as particular cases the backward Kolmogorov equation: 
$A_{ij} \equiv 1, B_{ij} \equiv 1, t_1 < t_0$, and the forward Kolmogorov
(Fokker-Plank) equation: $a_{ij} \equiv 1, b_{ij} \equiv 1, t_1 > t_0.$ 

The spatial grid (i.e. the grid of $\b x$ vectors) can be arbitrary in
one-dimensional problems ($N = 1$). In multiple dimensions the grid should be
rectangular and uniform in each dimension (the spacing in each dimension can be
different, however).

We support [Robin](https://en.wikipedia.org/wiki/Robin_boundary_condition)
boundary conditions on each edge of the spatial grid:

\begin{equation}
\alpha(\b x_b, t) V(\b x_b, t) + \beta(\b x_b, t)\frac{\d V}{\d \b n}(\b x_b, t)
= \gamma(\b x_b, t), 
\label{boundcond}
\end{equation}

where $\b x_b$ is a point on the boundary, and $\d V/\d \b n$ is the
derivative with respect to the outer normal to the boundary. In particular,
Dirichlet ($\alpha \equiv 1, \beta \equiv 0$) and Neumann
($\alpha \equiv 0, \beta \equiv 1$) boundary conditions are supported.

Below we describe in detail how to use the solver, and then the algorithms it
uses internally.

# Usage guide

## Simplest case: 1D PDE with constant coefficients

Let's start with a one-dimensional PDE with constant coefficients[^const_coeff_eq]:

\begin{equation}
 \frac{\d V}{\d t} - D \frac{\d^2{V}}{\d x^2} + \mu \frac{\d V}{\d x} - r V = 0.
\end{equation}

[^const_coeff_eq]: This is a diffusion-convection-reaction equation with
 constant diffusion coefficient $D$, drift $\mu$, and reaction rate $r$.

Let the domain be $x \in [-1, 1]$, the distribution at the initial time be
Gaussian, and the boundary conditions - Dirichlet with zero value on both
boundaries:
\begin{equation}
\begin{split}
&V(x, t_0) = e^{-x^2/2\sigma^2} (2\pi\sigma)^{-1/2},\\
&V(-1, t) = 0,\\
&V(1, t) = 0.
\end{split}
\end{equation}

We're seeking $V(x, t_1)$ with $t_1 > t_0$. 

Let's prepare the necessary ingredients. First, the spatial grid:

```python
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
pde = tff.math.pde

grid = pde.grids.uniform_grid(minimums=[-1],
                              maximums=[1],
                              sizes=[300],
                              dtype=tf.float32)
```

This grid is uniform with 300 points between (and including) $x=-1$ and
$x=1$. The `grids` module provides other types of grids, for example a
log-spaced grid. The `grid` object is a list of `Tensors`. Each element in the
list represents a spatial dimension. In our example `len(grid) = 1` and
`grid[0].shape = (300,)`.

We can also easily make a custom grid out of a numpy array:

```python
grid_np = np.array(...)
grid = [tf.constant(grid_np, dtype=tf.float32)]
```

The next ingredient is the PDE coefficients:

```python
d = 1
mu = 2
r = 3

def second_order_coeff_fn(t, grid):
  return [[-d]]

def first_order_coeff_fn(t, grid):
  return [mu]

def zeroth_order_coeff_fn(t, grid):
  return -r
```

Note the square brackets - these are required for conformance with
multidimensional case.

Next, the values at the initial time $t_0$:

```python
variance = 0.2
xs = grid[0]
initial_value_grid = (tf.math.exp(-xs**2 / (2 * variance)) / 
    tf.math.sqrt(2 * np.pi * variance)
```

And finally, define the final time, initial time, and number of time steps:

```python
t_0 = 0
t_1 = 1
num_time_steps = 100
```
Just as the spatial grid, the temporal grid can be customized: we can specify
number of time steps, a size of a step, or even a callable that accepts the
current time and returns the size of the next time step.

We now have all the ingredients to solve the PDE:

```python
result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(
        start_time=t_0,
        end_time=t_1,
        num_steps=num_time_steps,
        coord_grid=grid,
        values_grid=initial_value_grid,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn))
```
The resulting approximation of $V(x, t_1)$ is contained in
`result_value_grid`. The solver also yields the final spatial grid (because the
solver may modify the grid we've provided), the end time (because it may not be
exactly the one we specified, e.g. in case of custom time step callable), and
the number of performed steps.

To visualize the result, we can simply convert it to a numpy array (assuming
we use Tensorflow 2.x or are in the eager mode of Tensorflow 1.x), and apply the
usual tools:

```python
import matplotlib.pyplot as plt

xs = final_grid[0].numpy()
vs = result_value_grid.numpy()
plt.plot(xs, vs)
```
*TODO: insert pictures.*


## Non-constant PDE coefficients: Black-Scholes equation.

Let's now turn to an example of a PDE with non-constant coefficient - the Black-
Scholes equation:

\begin{equation}
 \frac{\d V}{\d t} + \frac12 \sigma^2 S^2 \frac{\d^2{V}}{\d S^2} +
 rS \frac{\d V}{\d S} - r V = 0.
\end{equation}

All we need to change in the previous example is the PDE coefficient callables:

```python
sigma = 1
r = 3

def second_order_coeff_fn(t, grid):
  s = grid[0]
  return [[sigma**2 * s**2 / 2]]

def first_order_coeff_fn(t, grid):
  s = grid[0]
  return [r * s]

def zeroth_order_coeff_fn(t, grid):
  return -r
```

As seen, the coordinates are extracted from the `grid` passed into the
callables, and then can undergo arbitrary Tensorflow transormations. The
returned tensors must be either scalars or have a shape implied by the `grid`
(this is easy to do in 1D, and a bit more tricky in multidimensional case, more
details are below).

The Black-Scholes equation is evolved backwards in time. Therefore use 
`fd_solvers.solve_backward` instead of `fd_solvers.solve_forward`, and make sure
that `start_time` is greater than `end_time`.

That's it, we can now numerically solve the Black-Scholes equation.

## Coefficients under the derivatives. Fokker-Planck equation.

As the next example let's consider the Fokker-Planck equation arising in the
Black-Scholes model for the probability distribution of the stock prices:

\begin{equation}
 \frac{\d p}{\d t} - \frac12 \sigma^2 \frac{\d^2}{\d S^2} \left[S^2 p\right] +
 \mu \frac{\d}{\d S}\left[S p\right] = 0.
\end{equation}

To specify coefficients under the derivatives, use
`inner_second_order_coeff_fn` and `inner_first_order_coeff_fn` arguments. The
specification is exactly the same as for `second_order_coeff_fn` and
`first_order_coeff_fn`. For our example we write:

```python
sigma = 1
mu = 2

def inner_second_order_coeff_fn(t, grid):
  s = grid[0]
  return [[-sigma**2 * s**2 / 2]]

def inner_first_order_coeff_fn(t, grid):
  s = grid[0]
  return [mu * s]

result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(...,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn))
```

We may specify both "outer" and "inner" coefficients, so the following is
equivalent (albeit possibly less performant):

```python
sigma = 1
mu = 2

def second_order_coeff_fn(t, grid):
  return [[-sigma**2 / 2]]

def first_order_coeff_fn(t, grid):
  return [mu]

def inner_second_order_coeff_fn(t, grid):
  s = grid[0]
  return [[s**2]]

def inner_first_order_coeff_fn(t, grid):
  s = grid[0]
  return [s]

result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(...,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        inner_second_order_coeff_fn=inner_second_order_coeff_fn,
        inner_first_order_coeff_fn=inner_first_order_coeff_fn))
```

 
## Batching

The solver can work with multiple PDEs in parallel. For example, let's solve
Black-Scholes equation for European options with a batch of strike values.
European options imply the final condition $V(S, t_f) = (S - K)_+$, where `K`
is the strike. So with one strike value we write:

```python
strike = 0.3
s = grid[0]
final_value_grid = tf.nn.relu(s - strike)
```

With a batch of `b` strike values, we need to stack the final value grids, so
that `final_value_grid[i]` is the grid for `i`th strike value. The simplest way
to do this is by using `tf.meshgrid`:

```python
strikes = tf.constant([0.1, 0.3, 0.5])
s = grid[0]
strikes, s = tf.meshgrid(strikes, s, indexing='ij')
final_value_grid = tf.nn.relu(s - strikes)
```

`tf.meshgrid` broadcasts the two tensors into a rectangular grid, and then we
can combine them with any algebraic operations. In this example `s` has shape
`(300,)`, `strikes` has shape `(3,)`. After applying `tf.meshgrid`, both
`strikes` and `s` have shape `(3, 300)`, and so does `final_value_grid`.

A more efficient way to obtain the shape `(3, 300)` is to use
[broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
The shapes `(3, )` and `(300, )` are not compatible, but `(3, 1)` and `(300, )`
are, and when combined lead to the desired shape `(3, 300)`. So we need to add
an extra dimension to `strikes`. Here are a few ways of doing that:

```python
strikes = tf.constant([0.1, 0.3, 0.5])
strikes = tf.reshape(strikes, (-1, 1))  # "-1" will automatically resolve to 3.
s = grid[0]
final_value_grid = tf.nn.relu(s - strikes)
```

```python
strikes = tf.constant([0.1, 0.3, 0.5])
# "axis=-1" means add a dimension to the end.
strikes = tf.expand_dims(strikes, axis=-1)
s = grid[0]
final_value_grid = tf.nn.relu(s - strikes)
```

```python
# Give strikes the desired shape right away.
strikes = tf.constant([[0.1], [0.3], [0.5]])
s = grid[0]
final_value_grid = tf.nn.relu(s - strikes)
```

There can be arbitrary number of batching dimensions, which is convenient when
there are multiple parameters:

```python
param1 = tf.constant([...])
param1 = tf.reshape(param1, (-1, 1, 1))
param2 = tf.constant([...])
param2 = tf.reshape(param2, (-1, 1))
s = grid[0]
final_value_grid = ...  # combine param1, param2 and s
```

Always make sure that the batch dimensions go before the grid dimensions in the
resulting Tensor.

After constructing `final_value_grid`, we pass it to `fd_solvers` as usual, and
the `result_value_grid` will contain the batch of solutions.

We may also have different models for each element of the batch, for example:

```python
strikes = tf.constant([[0.1], [0.3], [0.5]])
sigmas = tf.constant([[1.0], [1.5], [2]])
rs = tf.constant([[0.0], [1.0], [2.0]])

s = grid[0]
final_value_grid = tf.nn.relu(s - strikes)

def second_order_coeff_fn(t, grid):
  s = grid[0]
  return [[sigmas**2 * s**2 / 2]]

def first_order_coeff_fn(t, grid):
  s = grid[0]
  return [rs * s]

def zeroth_order_coeff_fn(t, grid):
  return -rs
```

This way we construct three PDEs: `i`-th equation has strike `strikes[i]`
and model parameters `sigmas[i]`, `rs[i]`.

In the simplest case, the batch shapes of `final_value_grid` and PDE coefficient
tensors match exactly. The general requirement is as
follows. The shape of value grid is composed of `batch_shape` and `grid_shape`.
Both are determined from the shape of `final_value_grid`. The dimensonality
`dim` of the PDE, i.e. the the rank of `grid_shape`, is inferred from the `grid`
passed into the solver: `dim = len(grid)` (in all examples so far, `dim = 1`).
`grid_shape` may evolve
with time, so should be determined from the `grid` argument passed into
`second_order_coeff_fn`, `first_order_coeff_fn` and `zeroth_order_coeff_fn`.
Recall that `grid` is a List of 1D Tensors; `grid_shape` is a concatenation of
shapes of these tensors. The requirement is that 
`second_order_coeff_fn(...)[i][j]`,
`first_order_coeff_fn(...)[i]` and `zeroth_order_coeff_fn(...)` must be tensors
whose shape is broadcastable to the shape `batch_shape + grid_shape`.

In the last example we return tensors of shapes `(3, 300)`, `(3, 300)`, and
`(3, 1)` from `second_order_coeff_fn`, `first_order_coeff_fn`, and
`zeroth_order_coeff_fn`, respectively, which satisfies the requirement.

The boundary conditions (see below) can be also batched in a similar way.
The coordinate grid and the temporal grid cannot be batched.

## Boundary conditions

The solver supports two types of boundary conditions. The first type is Robin
boundary conditions:

\begin{equation}
\alpha(\b x_b, t) V(\b x_b, t) + \beta(\b x_b, t)\frac{\d V}{\d \b n}(\b x_b, t)
= \gamma(\b x_b, t), 
\end{equation}

where $\b x_b$ is a point on the boundary, $\d V/\d n$ is the derivative
with respect to the outer normal to the boundary. The functions
$\alpha, \beta, \gamma$ can be arbitrary.

The other type is "default" boundary conditions: the differential equation is
satisfied on the boundary, except that the second-order terms involving
derivatives in the direction of that boundary vanish:

\begin{equation}
    \frac{\d V}{\d t} +
    \sum\limits_{i,j=1, i,j \neq k}^N a_{ij}(\b x_b, t)
        \frac{\d^2}{\d x_i \d x_j}
    \left[A_{ij}(\b x_b, t) V(\b x_b, t)\right] +
    \sum\limits_{i=1}^N b_i(\b x_b, t) \frac{\d}{\d x_i}
    \left[B_{i}(\b x_b, t) V(\b x_b, t)\right] +
     c(\b x_b, t) V = 0,
\end{equation}

where $k$ is the direction orthogonal to the boundary (note that in 1D this
simplifies to just "second order term vanished").

The boundary conditions are specified by callables returning
$\alpha, \beta, \gamma$ as Tensors or scalars, or returning `None` in case of
default boundary conditions. For example, a boundary condition

\begin{equation}
V + 2 \frac{\d V}{\d \b n} = 3 
\end{equation}

is defined as follows:

```python 
def boundary_cond(t, grid):
  return 1, 2, 3
```

Dirichlet and Neumann boundary conditions can be specified by setting
$\alpha = 1, \beta = 0$ and $\alpha = 0, \beta = 1$ explicitly, or by using
utilities found in `boundary_conditions.py`:

```python
@boundary_conditions.dirichlet
def dirichlet_boundary(t, grid):
  return 1  # V = 1

@boundary_conditions.neumann
def neumann_boundary(t, grid):
  return 2  # dV/dn = 2
```

Note that the derivative is taken with respect to the outer normal to the
boundary, not along the coordinate axis. So, for example, the condition
$(\d V/\d x)_{x=x_{min}} = 2$ translates to
$(\d V/\d \b n)_{x=x_{min}} = -2$.

The callables are passed into solver as a list of pairs, where each pair
represents the lower and the upper boundary of a particular spatial dimension.
In 1D this looks as follows:

```python
grid = grids.uniform_grid(minimums=[-1], maximums=[1], ...)

def lower_boundary(t, grid):
  return ...  # boundary condition at x = -1

def upper_boundary(t, grid):
  return ...  # boundary condition at x = 1

result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(...,
        boundary_conditions=[(lower_boundary, upper_boundary)]))
```

## Multidimensional PDEs; Hull-Heston-White equation.

As a first example of a multi-dimensional PDE, let's consider a 2D PDE with
constant coefficients[^const_coeff_eq_2d]:

\begin{equation}
\frac{\d V}{\d t} -
D_{xx} \frac{\d^2{V}}{\d x^2} - 2D_{xy} \frac{\d^2{V}}{\d x \d y} -
D_{yy} \frac{\d^2{V}}{\d y^2} +
\mu_x \frac{\d V}{\d x} + \mu_y \frac{\d V}{\d y} - r V = 0
\end{equation}

[^const_coeff_eq_2d]: This is a 2D diffusion-convection-reaction equation with
anisotropic diffusion (i.e. $D$ is now a 2x2 matrix).

First, we create a rectangular grid:

```python
x_min, x_max = -1.0, 1.0
y_min, y_max = -2.0, 2.0
x_size, y_size = 200, 300

grid = pde.grids.uniform_grid(minimums=[y_min, x_min],
                              maximums=[y_max, x_max],
                              sizes=[y_size, x_size],
                              dtype=tf.float32)
```

The `grid` object is a list of two Tensors of shapes (300,) and (200,).

Currently, only uniform grids are supported in the multidimensional case.
However, the steps of the grid can be different in each dimension.

The PDE coefficient callables look as follows:

```python
d_xx, d_xy, d_yy = 1, 0.2, 0.5
mu_x, mu_y = 1, 0.5
r = 3

def second_order_coeff_fn(t, grid):
  return [[-d_yy, -d_xy], [-d_xy, -d_xx]]

def first_order_coeff_fn(t, grid):
  return [mu_y, mu_x]

def zeroth_order_coeff_fn(t, grid):
  return -r
```

The matrix returned by `second_order_coeff_fn` can be a list of lists like
above, a Numpy array or a Tensorflow tensor - the requirement is that
`second_order_coeff_fn(...)[i][j]` is defined for `i <= j < dim` and represents
the corresponding PDE coefficient. The matrix is assumed symmetrical, and
elements with `i > j` are never accessed. Therefore, they can be anything,
so the following is also acceptable[^nones]:

```python
def second_order_coeff_fn(t, grid):
  return [[-d_yy, -d_xy], [None, -d_yy]]
```

[^nones]: `None` can also be used to indicate that certain PDE terms are absent.
  For example, when $\mu_x = 0$, we can return `[mu_y, None]` from
  `first_order_coeff_fn`. If both $\mu_x, \mu_y = 0$, we can return
  `[None, None]` or simply pass `first_order_coeff_fn=None` into the solver.

Let's take the following initial values:

\begin{equation}
V(x, y, t_0) =  (2\pi\sigma)^{-1} e^{-\frac{x^2 + y^2}{2\sigma^2}},
\end{equation}

which translates into


```python
sigma = 0.1
ys, xs = grid
ys = tf.reshape(ys, (-1, 1))
initial_value_grid = (tf.math.exp(-(xs**2 + ys**2) / (2 * sigma))
    / (2 * np.pi * sigma))
```

Finally, call `fd_solvers.solve_forward` or `fd_solvers.solve_backward` as
usual:

```python
t_0 = 0
t_1 = 1
num_time_steps = 100

result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(
        start_time=t_0,
        end_time=t_1,
        num_steps=num_time_steps,
        coord_grid=grid,
        values_grid=initial_value_grid,
        second_order_coeff_fn=second_order_coeff_fn,
        first_order_coeff_fn=first_order_coeff_fn,
        zeroth_order_coeff_fn=zeroth_order_coeff_fn)))
```

One way to visualize the result is by creating a heatmap:

```python
plt.imshow(result_value_grid.numpy(), 
           extent=[x_min, x_max, y_min, y_max],
           cmap='hot')
plt.show()
```

The boundary of the domain is rectangular, and each side can has its
boundary condition (by default - Dirichlet with zero value). For example, let's
"heat up" the right boundary and set zero flux across y-boundaries:

```python
@pde.boundary_conditions.dirichlet
def boundary_x_max(t, grid):
  return 1

@pde.boundary_conditions.dirichlet
def boundary_x_min(t, grid):
  return 0

@pde.boundary_conditions.neumann
def boundary_y_max(t, grid):
  return 0

@pde.boundary_conditions.neumann
def boundary_y_min(t, grid):
  return 0

result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(...,
         boundary_conditions=[(boundary_y_min, boundary_y_max),
                              (boundary_x_min, boundary_x_max)]))
```

The boundary conditions can be inhomogeneous. E.g. a "heat source" near
$x = x_{max}, y = 0$ may look like this:

```python
sigma = 0.2

@pde.boundary_conditions.dirichlet
def boundary_x_max(t, grid):
  ys = grid[0]
  return tf.exp(-ys**2 / (2 * sigma**2))
```

The general requirement for the shape of the boundary condition tensors is that
their shape must be broadcastable to
`batch_shape + grid_shape'`, where `grid_shape'` is the `grid_shape`
excluding the axis orthogonal to the boundary. In the simplest case of 1D
problem with no batching, the boundary callables must return either a scalar or
a rank-0 tensor. With batching, shapes that are broadcastable to the shape of
the batch are also acceptable. Thus, items in a batch can have different
boundary conditions. As another example, in a 3D problem the value
grid shape is `batch_shape + (z_size, y_size, x_size)`, so the boundary tensors
on the planes $y=y_{min}$ and $y=y_{max}$ should be broadcastable to
`batch_shape + (z_size, x_size)`.

As a more elaborate example, let's translate into code the Heston-Hull-White
PDE, as defined in [1]:

\begin{equation}
\begin{split}
&\frac{\d\varphi}{\d t} + \frac12 s^2 v \frac{\d^2\varphi}{\d s^2} +
\frac12 \sigma_1^2 v \frac{\d^2\varphi}{\d v^2} +
\frac12 \sigma_2^2 v \frac{\d^2\varphi}{\d r^2} + \\
&\rho_{12} \sigma_1 s v \frac{\d^2\varphi}{\d s \d v} +
\rho_{13} \sigma_2 s \sqrt{v} \frac{\d^2\varphi}{\d s \d r} +
\rho_{23} \sigma_1 \sigma_2 \sqrt{v} \frac{\d^2\varphi}{\d v \d r} + \\
&rs \frac{\d\varphi}{ds} + \kappa(\eta - v) \frac{\d\varphi}{dv} +
a(bt-r)\frac{\d\varphi}{dr} - ru = 0
\end{split}
\end{equation}

with boundary conditions

\begin{equation}
\begin{split}
&\varphi(0, v, r, t) = 0, \\
&\frac{\d\varphi}{\d s}(s_{max}, v, r, t) = 1, \\
&\left(\frac{\d\varphi}{\d t} + rs \frac{\d\varphi}{ds} +
 \kappa\eta\frac{\d\varphi}{dv} + a(bt-r)\frac{\d\varphi}{dr} - ru\right)
 \bigg\rvert_{v=0} = 0, \\
&\varphi(s, v_{max}, r, t) = s, \\
&\frac{\d\varphi}{\d r}(s, v, r_{min}, t) = 0, \\
&\frac{\d\varphi}{\d r}(s, v, r_{max}, t) = 0. \\
\end{split}
\end{equation}

The boundary condition at $v = 0$ is obtained by plugging $v = 0$ into the PDE,
resulting in a "default" boundary condition.
```python
def second_order_coeff_fn(t, grid):
  s, v, r = grid
  s = tf.reshape(s, (-1, 1, 1))
  v = tf.reshape(v, (-1, 1))
  coeff_ss = s**2 * v / 2
  coeff_vv = sigma1**2 * v / 2
  coeff_rr = sigma2**2 * v / 2
  coeff_sv = rho12 * sigma1 * s * v / 2
  coeff_sr = rho13 * sigma2 * s * tf.sqrt(v) / 2
  coeff_vr = rho23 * sigma1 * sigma2 * tf.sqrt(v) / 2
  return [[coeff_ss, coeff_sv, coeff_sr],
          [None, coeff_vv, coeff_vr],
          [None, None, coeff_rr]]

def first_order_coeff_fn(t, grid):
  s, v, r = grid
  s = tf.reshape(s, (-1, 1, 1))
  v = tf.reshape(v, (-1, 1))
  coeff_s = r * s
  coeff_v = kappa * (eta - v)
  coeff_r = a * (b * t - r)
  return [coeff_s, coeff_v, coeff_r]

def zeroth_order_coeff_fn(t, grid):
  return -r

@pde.boundary_conditions.dirichlet
def boundary_s_min(t, grid):
  return 0

@pde.boundary_conditions.neumann
def boundary_s_max(t, grid):
  return 1

def boundary_v_min(t, grid):
  return None

@pde.boundary_conditions.dirichlet
def boundary_v_max(t, grid):
  s = grid[0]
  # Shape must be broadcastable to the shape of the boundary, which
  # is (s_size, r_size). By using expand_dims we obtain an acceptable shape
  # (s_size, 1).
  return tf.expand_dims(s, -1)

@pde.boundary_conditions.neumann
def boundary_r_min(t, grid):
  return 0

@pde.boundary_conditions.neumann
def boundary_r_max(t, grid):
  return 0
```

## Customizing the time marching scheme

The solver allows specifying a time marching scheme. Time marching schemes
define how a single time step is performed (after spatial discretization of a
PDE), and they differ in numerical stability, accuracy, and performance.

To use, for example, the explicit scheme, which is less accurate and stable but
much faster than the default (Crank-Nicolson) scheme, we write:

```python
result_value_grid, final_grid, end_time, steps_performed = (
    pde.fd_solvers.solve_forward(...
        one_step_fn=pde.steppers.explicit.explicit_step()))
```

Currently the following schemes are supported for 1D:

1. Explicit,
2. Implicit,
3. Crank-Nicolson,
4. Weighted explicit-implicit,
5. Extrapolation scheme, [3]
6. Crank-Nicolson with oscillation damping (see discussion below).

For multidimensional problems we currently have:

1. Douglas ADI [1, 4].

By default, the Crank-Nicolson with oscillation damping is used for 1D problems
and Douglas ADI with $\theta = 0.5$ - for multidimensional problems.

See below a detailed discussions of these schemes.

# Implementation

The bird's eye view of a finite-difference PDE solving algorithm is

```
for time_step in time_steps:       (1)
   do_spatial_discretization()     (2)
   apply_time_marching_scheme()    (3)
```

The spatial discretization converts a PDE for $V(\b x, t)$ into a linear
system of equations for $\b V(t)$, which is a vector consisting of
$V(\b x_i, t),$ where $\b x_i$ are grid points. The resulting system of
equations has the form

\begin{equation}
\frac{d\b V}{dt} = L(t)\b V + \b b(t),\label{space_discretized}
\end{equation}

where $\b b$ is a vector, and $L$ is a sparse matrix describing the
influence of adjacent grid points on each other. $L(t)$ and $\b b(t)$
are defined by both PDE coefficients and the boundary conditions, evaluated at
time $t$.

The job of a time marching scheme is to approximately solve
Eq.\eqref{space_discretized} for $\b V(t + \delta t)$ given $\b V(t)$
($\delta t$ may be positive or negative).

All three lines in the above pseudocode represent the routines that are mostly
independent from each other. Therefore, the solver is modularized accordingly:

- `fd_solvers.py` implement looping over time steps, i.e. line (1) in the 
pseudocode. The routines accept (and provide reasonable defaults for)
`one_step_fn` argument, which is a callable performing both (2) and (3).

- `parabolic_equation_stepper.py` and `multidim_parabolic_equation_stepper.py`
perform space discretization for one-dimensional and multidimensional parabolic
PDEs, respectively. They accept a `time_marching_scheme` callable for performing
(3).

- `crank_nicolson.py`, `explicit.py`, `douglas_adi.py` and other modules
implement specific time marching schemes. Each of these modules has a
`<scheme_name>_scheme` function for the scheme itself, and a
`<scheme_name>_step` function. The latter is a convenience function which
combines the given scheme and the appropriate space discretizer; it can be
passed directly to `fd_solvers`.

This setup is flexible, allowing for easy customization of every component.
For example, it's straightforward to implement a new time marching scheme and
plug it in.

Below are the implementation notes for space discretizers and the time marching
schemes.

## Spatial discretization

The spatial derivatives are approximated to the second order of accuracy, and
taking into account that the grid may be non-uniform. Consider first the 1D case,
and denote $x_0$ the given point in the grid, and $\Delta_+, \Delta_-$ the
distances to adjacent grid points.
The Taylor's expansion of a function $f(x)$ with smooth second derivative
gives
\begin{eqnarray}
f(x_0 + \Delta_+) &=& f(x_0) +  f'(x_0) \Delta_{+}
    + \frac{1}{2}f''(x_0) (\Delta_{+})^2 + \mathcal{O}((\Delta_{+})^3),
    \label{taylorplus}\\
f(x_0 - \Delta_-) &=& f(x_0) - f'(x_0) \Delta_{-}
    + \frac{1}{2}f''(x_0) (\Delta_{-})^2 + \mathcal{O}((\Delta_{-})^3).
    \label{taylorminus}
\end{eqnarray}

Solving this system of equations for $f'(x_0)$ and $f''(x_0)$ yields
\begin{eqnarray}
&f'(x_0) \approx C_+ f(x_0 + \Delta_+) + C_- f(x_0 - \Delta_-) - 
  (C_+ + C_-) f(x_0),
\label{deriv_approx_central} \\
&C_\pm = \pm\frac{\Delta_\mp}{(\Delta_+ + \Delta_-)\Delta_\pm}.
\end{eqnarray}

and
\begin{eqnarray}
&f''(x_0) \approx D_+ f(x_0 + \Delta_+) + D_- f(x_0 - \Delta_-) - 
  (D_+ + D_-) f(x_0),
\label{deriv_approx_central} \\
&D_\pm = \frac{2}{(\Delta_+ + \Delta_-)\Delta_\pm}.
\end{eqnarray}

Thus, space discretization a 1D homogeneous parabolic PDE yields
Eq.\eqref{space_discretized} with $\b b = 0$, and a tridiagonal matrix
$L$.

In multidimensional case the discretization is done similarly, but we
additionally need to take care of mixed second derivatives. Since only uniform
grids are currently supported in multidimensional case, we apply the usual
approximation of mixed derivatives:

\begin{equation}
\frac{\d^2 f}{\d x \d y}(x_0, y_0) \approx \frac{
f(x_+, y_+) - f(x_-, y_+) - f(x_+, y_-) + f(x_-, y_-)} {4\Delta_x\Delta_y},
\end{equation}

where $x_\pm = x_0 \pm \Delta_x, y_\pm = y_0 \pm \Delta_y$ are adjacent grid
points.

Considering that $\b V$ in Eq.\eqref{space_discretized} is a vector, i.e. a
flattened multidimensional value grid, the matrix $L$ is now banded,
with contributions coming from adjacent points in every dimension.

### Boundary conditions in 1D PDEs

Consider first the Robin boundary conditions, Eq. \eqref{boundcond}, for a
one-dimensional PDE. When discretizing Eq. \eqref{boundcond},
we approximate the derivative to the second order of accuracy, just like we did
with the PDE terms. The central approximation, Eq. \eqref{deriv_approx_central},
is however not applicable, because it uses the values at adjacent points on both
sides. Instead, we express the boundary derivative via the two closest points on
one side. Consider the lower boundary $x_0$, and the two
closest grid points $x_1 = x_0 + \Delta_0$, and $x_2 = x_1 + \Delta_1$:
\begin{eqnarray}
f(x_1) &\approx& f(x_0) +  f'(x_0) \Delta_0
    + \frac12 f''(x_0) \Delta_0^2,\\
f(x_2) &\approx& f(x_0) + f'(x_0) (\Delta_0 + \Delta_1)
    + \frac12 f''(x_0) (\Delta_0 + \Delta_1)^2
\end{eqnarray}

Eliminating $f''(x_0)$ from this system, we obtain
\begin{equation}
f'(x_0) \approx \frac
{
  (\Delta_0 + \Delta_1)^2 f(x_1) - \Delta_0^2 f(x_2)
  - \Delta_0 ( 2\Delta_0 + \Delta_1 ) f(x_0)
}{
 \Delta_0\Delta_1 ( \Delta_0 + \Delta_1 )
}
\end{equation}

(see e.g. [2], §2.3; only uniform grids are considered there though).

Similarly, we express the derivative at the upper boundary $x_n$ via the two
closest points on the left of it: $x_{n-1} = x_n - \Delta_{n-1}$ and
$x_{n-2} = x_{n-1} - \Delta_{n-2}$. The expression differs essentially only in
the sign:
\begin{equation}
f'(x_n) \approx -\frac
{
  (\Delta_{n-1} + \Delta_{n-2})^2 f(x_{n-1}) - \Delta_{n-1}^2 f(x_{n-2})
  - \Delta_{n-1} ( 2\Delta_{n-1} + \Delta_{n-2} ) f(x_n)
}{
 \Delta_{n-1}\Delta_{n-2} ( \Delta_{n-1} + \Delta_{n-2} )
}
\end{equation}

Substituting these two equations into Eq. \eqref{boundcond} (taking into
account that $\d V/\d \b n = - \d V/\d x$ on the lower boundary, and
$\d V/\d \b n = \d V/\d x$ on the upper boundary), we get the following
discretized versions of it:
\begin{eqnarray}
V_0 = \xi_1 V_1 + \xi_2 V_2 + \eta,\label{lower_bound_discr}\\
V_n = \bar\xi_1 V_{n-1} + \bar\xi_2 V_{n-2} + \bar\eta,\label{upper_bound_discr}
\end{eqnarray}

where
\begin{eqnarray}
&&\xi_1 = \beta (\Delta_0 + \Delta_1)^2 / \kappa, \\
&&\xi_2 = -\beta \Delta_0^2 / \kappa, \\
&&\eta = \gamma \Delta_0 \Delta_1 (\Delta_0 + \Delta_1) / \kappa, \\
&&\kappa = \alpha \Delta_0 \Delta_1 (\Delta_0 + \Delta_1) +
   \beta \Delta_1 (2\Delta_0 + \Delta_1).
\end{eqnarray}

The expressions for $\bar\xi_1, \bar\xi_2$ and $\bar\eta$ are exactly the
same, except $\Delta_{1, 2}$ is replaced by $\Delta_{n-1, n-2}$, and of
course $\alpha, \beta$ and $\gamma$ come from the upper boundary condition.

The evolution of the values on the inner part of the grid, $V_0, \ldots
V_{n-1}$ is defined by a tridiagonal matrix, as discussed in the previous
section:
\begin{equation}
\frac {d V_i}{dt} = L_{i, i-1} V_{i-1} + L_{i, i} V_{i} + L_{i, i+1} V_{i+1},
\qquad i = 1\ldots N-1
\end{equation}

Substituting Eqs. \eqref{lower_bound_discr}, \eqref{upper_bound_discr}, we obtain
for the inner part $\b V_{inner} = [V_1, ... V_{n-1}]^T$:

\begin{equation}
\frac {d \b V_{inner}}{dt} = {\tilde L} \b V_{inner} + \b b,
\label {dVdt_with_boundary}
\end{equation}

where 
\begin{eqnarray} 
&&\tilde L_{11} = L_{11} + \xi_1 L_{01}, \qquad \tilde L_{12} =
L_{12} + \xi_2 L_{01},\label{bound_corr_first} \\ &&\tilde L_{n-1, n-1} =
L_{n-1,n-1} + \bar\xi_1 L_{n,n-1}, \qquad \tilde L_{n-1, n-2} = L_{n-1,n-2} +
\bar\xi_2 L_{n,n-1}, \\ &&\tilde L_{ij} = L_{ij} \qquad i=2\ldots n-2,\\ &&b_1 =
L_{01} \eta, \qquad b_{n-1} = L_{n, n-1}\bar \eta, \\ &&b_i = 0 \qquad i=2\ldots
n-2.\label{bound_corr_last} 
\end{eqnarray}

Note that in case of Dirichlet conditions ($\alpha = 1, \beta = 0$) this
simplifies greatly: $\xi_{1,2} = 0$ (so there are no corrections to
$L$), and $\eta = \gamma$.

Thus to take into account the Robin boundary conditions we

*   apply the corrections to the time evolution equation given by Eqs.
    \eqref{bound_corr_first}-\eqref{bound_corr_last},
*   apply the chosen time marching scheme to find the "inner" part of the values
    vector $\b V_{inner}(t_{i+1})$ from Eq.
    \eqref{dVdt_with_boundary} given $\b V_{inner}(t_i)$,
*   and finally, find $V_0$ and $V_n$ from Eqs. \eqref{lower_bound_discr},
    \eqref{upper_bound_discr} at time $t + \delta t$, and append them to
    $\b V_{inner}(t_{i+1})$ to get the resulting vector $\b V(t_{i+1})$.

With "default" boundary conditions this approach isn't applicable: they involve
a time derivative, so $V_0$ or $V_n$ cannot be eliminated from
the system of equations like above. Instead, we space-discretize the boundary
condition as usual (except that the finite differences are non-central), and
include them in Eq. \eqref {dVdt_with_boundary}. Thus, $V_{inner}$ in
Eq. \eqref {dVdt_with_boundary} excludes boundaries with Robin conditions, but
includes the ones with "default" conditions. 

### Boundary conditions in multidimensional PDEs

The described approach generalizes in most aspects to the
multidimensional case:
$\b V_{inner}$ would be the value grid with all boundaries with Robin conditions
"trimmed", and Eqs. \eqref{bound_corr_first}-\eqref{bound_corr_last} would
similarly apply to the items next to the boundary.

However, the mixed derivative terms need special care. Consider a point
$V_{i, 1}$ near a boundary $(i, j=0)$ with a Robin boundary condition. In
presence of a mixed term, this point gets "influenced" by three points on the
boundary: $(i-1, 0), (i, 0), (i+1, 0).$ The approach outlined above would
require eliminating all of $V_{i-1, 0}, V_{i, 0}, V_{i+1, 0}$. In the corners,
we'd have to eliminate 5 variables instead of 3. This is quite cumbersome, and
may degrade performance.

Instead, we use the fact that the commonly used multidimensional time-marching
schemes apply the mixed term explicitly (See [1], page 7, and notice that
$A_0$, representing contributions of mixed terms, always gets multiplied by an
already known vector). This allows us to reformulate the space-discretized
problem as

\begin{equation}
\frac {d \b V_{inner}}{dt} = {\tilde L} \b V_{inner} + 
{L}_{mixed} \b V  +\b b,
\label {dVdt_with_boundary_multidim}
\end{equation}

where ${L}_{mixed}$ contains the contributions of mixed terms. When applying
a time marching scheme (e.g. Douglas scheme, see below), $\b V(t)$ is restored
from $\b V_{inner}(t)$ using Eqs. \eqref{lower_bound_discr},
\eqref{upper_bound_discr}. Note that this is only possible when
${L}_{mixed}$ is treated explicitly, otherwise $\b V_{inner}(t)$ is
unknown.


## Time marching schemes

Time marching schemes are algorithms for numerically solving the equation

\begin{equation}
\frac{d\b V}{dt} = L(t)\b V + \b b(t),\label{space_discretized2}
\end{equation}

for $\b V(t + \delta t)$ given $\b V(t),$ with a small $\delta t$.

Note that in case of constant parameters $L$ and $\b b$ this equation
has an exact solution involving the matrix exponent of $L$. Calculating
the matrix exponent is however infeasible in practice, primarily due to memory
constraints. The matrix $L$ is never explicitly constructed in the first
place. Recall that in 1D the matrix $L$ is tridiagonal, and in
multidimensional case it is banded. Only the non-zero diagonals are constructed,
and the time marching schemes make use of the sparseness of $L.$

### Explicit Scheme

The simplest scheme is the explicit scheme:

\begin{equation}
\frac{\b V_1 - \b V_0}{\delta t} = L_0 \b V_0 + \b b_0.
\end{equation}

Here and below we use the notation $X_\alpha = X(t + \alpha \delta t)$ where
$X$ is any function of time and $\alpha$ is a number between 0 and 1.

From there we obtain

\begin{equation}
\b V_1 = (1 + \delta tL_0) \b V_0 + \delta t \b b_0.
\end{equation}

The calculation boils down to multiplying a tridiagonal matrix by a vector.
Tensorflow has the
[tf.linalg.tridiagonal_matmul](https://www.tensorflow.org/api_docs/python/tf/linalg/tridiagonal_matmul)
Op, which can efficiently parallelize the multiplication on the GPU.

Unfortunately, the explicit scheme is applicable only with small time steps.
For a well-defined PDE, the matrix $\delta t L$ has negative eigenvalues
(This statement includes PDEs solved both in forward and backward directions,
i.e. $\delta t$ can be both positive and negative). When an eigenvalue
of $1 + \delta t L_0$ becomes less than $-1$, the contribution to
$\b V$ of the corresponding eigenvector grows exponentially with time. This
does not happen to the exact solution, so the error of the approximate solution
grows exponentially, i.e. the method is unstable. The stability requirements
are $\delta t \ll \Delta x^2 / D$ and $\delta t \ll \Delta x / \mu,$ where
$D$ and $\mu$ are the typical values of second and first order coefficients
of a PDE. This can constrain $\delta t$ to unacceptably small values.

### Implicit Scheme

The implicit scheme approximates the right hand side with its value at
$t + \delta t:$
\begin{equation}
\frac{\b V_1 - \b V_0}{\delta t} = L_1 \b V_1 + \b b_1,
\end{equation}

yielding
\begin{equation}
\b V_1 = (1 - \delta t L_0)^{-1} (\b V_0 + \delta t \b b_1).
\end{equation}

This takes care of the stability problem, because the eigenvalues of
$(1 - \delta tL_0)^{-1}$ are between 0 and 1. However, we now need to
solve a tridiagonal system of equation.

Tensorflow has the
[tf.linalg.tridiagonal_solve](https://www.tensorflow.org/api_docs/python/tf/linalg/tridiagonal_solve)
Op for this purpose. On GPU, it uses
[gtsv](https://docs.nvidia.com/cuda/cusparse/index.html#gtsv) routines from
Nvidia's CUDA Toolkit. These routines employ Parallel Cyclic Reduction algorithm
(see e.g. [5]) which enables certain level of parallelization.

### Weighted Implicit-Explicit Scheme and Crank-Nicolson Scheme

Weighted Implicit-Explicit Scheme is a combination of implicit and explicit
schemes:

\begin{equation}
\frac{\b V_1 - \b V_0}{\delta t} = \theta(L_0 \b V_0 + \b b_0) +
(1-\theta)(L_1 \b V_1 + \b b_1),
\end{equation}

where $\theta$ is a number between 0 and 1.

This yields
\begin{equation}
\b V_1 = (1 - (1-\theta)\delta t L_0)^{-1} [
(1+\theta \delta_t L_1)\b V_0 + \theta\delta t \b b_0 +
(1-\theta)\delta t \b b_1].
\end{equation}

The special case of $\theta = 1/2$ is the Crank-Nicolson (CN) scheme:
\begin{equation}
\b V_1 = (1 - \delta tL_0 / 2)^{-1} [
(1+\Delta_t L_1 / 2)\b V_0 + \delta t (\b b_0 + \b b_1) / 2].
\label{cn}
\end{equation}

CN scheme is stable (the eigenvalues are between -1 and 1), and,
unlike schemes with any other value of $\theta$, is second-order accurate in
$\delta t.$

One can verify (by comparing the Taylor expansions) that replacing
$L_{0,1}, \b b_{0,1}$ with $L_{1/2}, \b b_{1/2}$ in the right-hand
side of Eq.\eqref{space_discretized} retains the second order accuracy, while
saving some computation. Thus the final expression is
\begin{equation}
\b V_1 = (1 - \delta t L_{1/2} / 2)^{-1} [
(1+\delta t L_{1/2} / 2)\b V_0 + \delta t \b b_{1 / 2}].
\end{equation}


### Oscillations in Crank-Nicolson Scheme and Extrapolation Scheme

In financial applications, the initial or final data is often not smooth.
For example, the call option payoff has a discontinuity in the first
derivative at the strike. This can cause a severe drop in the accuracy of the
Crank-Nicolson scheme: it results in oscillations around the points of
discontinuity.

The reason of this effect is Crank-Nicolson scheme poorly approximating the
evolution of high-wavenumber components of the initial/final condition
function.

For the sake of discussion, assume $\delta t > 0$, $\b b(t) = 0$ and
$L(t) = -A$ (note that $A$ has positive eigenvalues in this
case). The exact solution of Eq. \eqref{space_discretized2} is then

\begin{equation}
\mathbf V(t + \delta t) = \exp(-A \delta t) \mathbf V(t).
\end{equation}

Various time marching schemes can be viewed as approximations to the exponent.
Crank-Nicolson scheme corresponds to the approximation

\begin{equation}
\exp(-y) \approx \frac{1-y/2}{1+y/2}.
\end{equation}

Here $y=\lambda \delta t$ and $\lambda$ is an
eigenvalue of $A$. This approximation, while being second order accurate
for $y \ll 1$, is clearly inaccurate for $y \gtrsim 1$, and because it is
negative there, the oscillations arise.

There are a few approaches to this problem found in the literature.

1) Replacing CN scheme with schemes based on Richardson extrapolation, i.e.
taking a linear combination of results of making time steps of different sizes,
with coefficients chosen to achieve desired accuracy. The simplest scheme
corresponds to the following approximation [3]:
\begin{equation}
\exp(-y) \approx \frac 2{(1 + y/2)^2} - \frac 1 {1 + y}.
\end{equation}
This means we are making two implicit half-steps, one implicit full step, and
combine the results. This approximation is second order accurate for small
$y$, just like CN, but is a more reasonable approximation of the exponent for
large $y$. The cost of this improvement is having to solve three tridiagonal
systems per step, compared to one system and one tridiagonal multiplication in
CN scheme. The implementation of this scheme can be found in `extrapolation.py`.
Higher orders of accuracy can be also achieved in this fashion. For example,
combining two CN half-steps and CN full step with coefficients 4/3 and -1/3
yields a third-order accurate approximation [6].

2) Subdiving the first time step into multiple smaller steps [7]. The idea is
that after the first step the irregularities in the initial
conditions are already smoothened to a large enough degree that CN scheme can be
applied to all further steps. But this smoothening itself needs to be modelled
more accurately. By reducing $\delta t$ we can ensure $y \lesssim 1$, for
all components of initial/final condition function.

3) Subdividing the first step into two fully implicit half-steps ("Rannacher
marching"), [8]. This is more computationally efficient than the two other
approaches. However, the fully implicit steps, while indeed damping the
oscillations, cause significant errors for low-wavenumber components, because of
its first-order accuracy on small wavenumbers.

The optimal approach seems to be a combination of all three ideas: apply the
extrapolation scheme to the first time step, or a few first steps, then proceed
with CN scheme. This is mentioned in the end of [8]. Such approach adds
minimal computational burden: only a finite number of extra tridiagonal systems
to solve. It ensures damping of oscillations, i.e. a reasonable treatment of
high-wavenumber components, while maintaining second-order accuracy of
low-wavenumber components.

The high-wavenumber components are dampened by approximately $(\delta t\max_i
(\lambda_i))^{-n_e},$ where $n_e$ is number of extrapolation steps taken. One
can approximate the maximum eigenvalue, and tune $n_e$ to achieve desired
level of oscillation damping.

The implementation of this can be found in
`oscillation_damped_crank_nicolson.py`.


### Multiple spatial dimensions, ADI

In case of multiple spatial dimensions, the matrix $L$ is banded, making
it much more difficult to do the matrix inversion in the Crank-Nicolson scheme.

The common workaround is the ADI (alternating direction explicit) method.
The time step is split into several substeps, and each substep treats only one
dimension explicitly. There is a number of time marching schemes based on this
idea. An overview of them can be found, for example, on pages 6-9 of [1]. Here
we summarize the simplest of them, the Douglas scheme, in our notations:

\begin{eqnarray}
\b Y_0 &=& (1 + L_0 \delta t) \b V_0 + \delta t \b b_0, \\
\b Y_j &=& \b Y_{j-1} + \theta \delta t (L^{(j)}_1 \b Y_j -
L^{(j)}_0 \b V_0 + \b b^{(j)}_1 - \b b^{(j)}_0), \qquad j = 1\ldots dim, \\
\b V_1 &=& \b Y_{dim}.
\end{eqnarray}

Here $L^{(j)}$ contains contributions to $L$ coming from derivatives
with respect to j-th dimension. Each $L^{(j)}$ thus have only three
nonzero diagonals. The contribution of the zeroth-order term
is split evenly between $L^{(j)}$. The contributions of mixed
second-order terms are not included in any $L^{(j)}$, and thus take part
only in the first substep, which lacks any matrix inversion.
$\b b^{(j)}$ are contributions from boundary conditions along the j-th axis
(boundary conditions are assumed to be the only source of $\b b$).
$\theta$ is a positive number.

The scheme is second-order accurate if there are no mixed derivative terms and
$\theta = 1/2,$ and first-order accurate otherwise. It is stable if
$\theta \geq 1/2$.

The implementation can be found in `douglas_adi.py`. The main computational
burden is solving large batches of tridiagonal systems plus transposing the
value grid Tensor on each substep. To understand why the latter is necessary,
recall that in implementation, $\b V$ is not a vector, but a Tensor of shape
`batch_shape + grid_shape`. When making the substep which treats j-th dimension
implicitly, we should treat all the other dimensions as batch dimensions.

For example, consider a 3D case where the shape is
`batch_shape + (z_size, y_size, x_size)`. When making the substep
that treats `y` implicitly, we transpose the value grid to the shape
`batch_shape + (z_size, x_size, y_size)`, and send it to
`tf.linalg.tridiagonal_solve`. From the point of view of the latter, the batch
shape is `batch_shape + (z_size, x_size)`, exactly as we need. After that we
transpose the result back to the original shape.

# References

[1] [Tinne Haentjens, Karek J. in't Hout. ADI finite difference schemes for
the Heston-Hull-White PDE](https://arxiv.org/abs/1111.4087)

[2] [S. Mazumder. Numerical Methods for Partial Differential Equations. ISBN
9780128498941. 2015.](https://www.sciencedirect.com/book/9780128498941/numerical-methods-for-partial-differential-equations)

[3] [D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods
for Parabolic Partial Differential Equations. I. 1978
SIAM Journal on Numerical Analysis. 15. 1212-1224.](https://epubs.siam.org/doi/abs/10.1137/0715082)

[4] [Douglas Jr., Jim (1962), "Alternating direction methods for three space
variables", Numerische Mathematik, 4 (1): 41-63.](https://dl.acm.org/citation.cfm?id=2722576)

[5] [D. Göddeke, R. Strzodka, "Cyclic reduction tridiagonal solvers on GPUs
applied to mixed precision multigrid", IEEE Transactions on Parallel and
Distributed System, vol. 22, pp. 22-32, Jan. 2011.](https://ieeexplore.ieee.org/abstract/document/5445081)

[6] [A. R. Gourlay and J. Ll. Morris. The Extrapolation of First Order Methods
for Parabolic Partial Differential Equations, II. SIAM Journal on Numerical
Analysis Vol. 17, No. 5 (Oct., 1980), pp. 641-655.](https://www.jstor.org/stable/2156665?seq=1)

[7] [D.Britz, O.Østerby, J.Strutwolf. Damping of Crank–Nicolson error
oscillations. Computational Biology and Chemistry, Vol. 27, Iss. 3, July 2003,
Pages 253-263.](https://www.sciencedirect.com/science/article/pii/S009784850200075X)

[8] [Giles, Michael & Carter, Rebecca. (2005). Convergence analysis of
Crank-Nicolson and Rannacher time-marching. J. Comput. Finance. 9.](https://www.researchgate.net/publication/228524629_Convergence_analysis_of_Crank-Nicolson_and_Rannacher_time-marching)
