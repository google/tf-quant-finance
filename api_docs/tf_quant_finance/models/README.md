# Diffusion Models.

This module is a toolbox for generating and calibrating Diffusion
Models. A model is derived from the ```ItoProcess``` class. The model is
described in terms of the underying stochastic differential equation (SDE):

$$dX_i = a_i(t, X) dt + \sum_{i, j = 1}^n S_{ij}(t, X) dW_j,$$

where $$n$$ is the dimensionality of the process $$a_i$$ and $$S_{ij}$$ are the
drift and volatility terms of the SDE. $$\{ W_i \}_{i=1}^n$$ is an
$$n$$-dimensional Wiener process.

For example, the Geometric Brownian Motion is an underlying of the Black-Scholes
model.

A minimal description of the model class should contain the following methods:

  *   ```dim``` - dimensionality of the underlying SDE;
  *   ```dtype``` - dtype of the model coefficients;
  *   ```name``` - name of the model class;
  *   ```drift_fn``` - drift rate of the process expressed as a callable
    which maps time and position to a vector. (corresponds to the $$a_i$$ above);
  *   ```volatility_fn``` - volatility of the process expressed as a callable
    which maps time and position to a volatility matrix(corresponds to the $$S_{ij}$$ above);
  *   ```sample_paths``` - return sample paths of the process at specified time
  points. The base class provides Euler scheme sampling if the drift and
  volatility functions are defined;
  * ```fd_solver_backward``` - returns a finite difference method for solving
  the [Feynman Kac PDE](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula)
  associated with the SDE. This equation is a slight generalization of the
  [Kolmogorov backward equation](https://en.wikipedia.org/wiki/Kolmogorov_backward_equations_(diffusion))
  with the inclusion of a discounting function. The Feynman Kac PDE is:

  $$V_t + Sum[mu_i(t, x) V_i, 1<=i<=n] + \frac{1}{2} Sum[ D_{ij} V_{ij}, 1 <= i,j <= n] - r(t, x) V = 0$$

  with the final value condition $$V(T, x) = u(x)$$.

  The corresponding Kolmogorov forward/Fokker Plank equation  is

  $$\frac{\partial}{\partial t} V(t, x) +  \sum_{i=1}^n \frac{\partial}{\partial x_i}  a_i(t, X) V(t, x) + \sum_{i, j, k = 1}^n \frac{\partial^2}{\partial x_i \partial x_j} S_{ik} S_{jk} V(t, x) = 0.$$

  with the initial value condition $$V(0, x) = u(x)$$.

  A minimum description of the model should only include ```dim```, ```dtype```,
  and ```name```. In order to use the provided ```sample_paths``` method,
  both ```drift_fn``` and ```volatility_fn``` should be defined.


TODO(b/140290854): Provide description of model calibration procedure.
TODO(b/140313472): Provide description of Ito process algebra API.                                                            


