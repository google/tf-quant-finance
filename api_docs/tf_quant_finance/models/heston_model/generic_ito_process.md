<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.heston_model.generic_ito_process" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_quant_finance.models.heston_model.generic_ito_process

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/generic_ito_process.py">View source</a>



Defines class to describe any Ito processes.


Uses Euler scheme for sampling and ADI scheme for solving the associated
Feynman-Kac equation.

## Modules

[`euler_sampling`](../../../tf_quant_finance/models/euler_sampling.md) module: The Euler sampling method for ito processes.

[`fd_solvers`](../../../tf_quant_finance/math/pde/fd_solvers.md) module: Functions for solving linear parabolic PDEs.

[`ito_process`](../../../tf_quant_finance/models/heston_model/generic_ito_process/ito_process.md) module: Defines an interface for Ito processes.

## Classes

[`class GenericItoProcess`](../../../tf_quant_finance/models/GenericItoProcess.md): Generic Ito process defined from a drift and volatility function.

