<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pde.fd_solvers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_quant_finance.math.pde.fd_solvers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/math/pde/fd_solvers.py">View source</a>



Functions for solving linear parabolic PDEs.



## Functions

[`douglas_adi_step(...)`](../../../tf_quant_finance/math/pde/fd_solvers/douglas_adi_step.md): Creates a stepper function with Crank-Nicolson time marching scheme.

[`oscillation_damped_crank_nicolson_step(...)`](../../../tf_quant_finance/math/pde/fd_solvers/oscillation_damped_crank_nicolson_step.md): Scheme similar to Crank-Nicolson, but ensuring damping of oscillations.

[`solve_backward(...)`](../../../tf_quant_finance/math/pde/fd_solvers/solve_backward.md): Evolves a grid of function values backwards in time according to a PDE.

[`solve_forward(...)`](../../../tf_quant_finance/math/pde/fd_solvers/solve_forward.md): Evolves a grid of function values forward in time according to a PDE.

