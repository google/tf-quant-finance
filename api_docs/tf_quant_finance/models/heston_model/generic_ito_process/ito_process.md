<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.heston_model.generic_ito_process.ito_process" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_quant_finance.models.heston_model.generic_ito_process.ito_process

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/ito_process.py">View source</a>



Defines an interface for Ito processes.


Ito processes underlie most quantitative finance models. This module defines
a framework for describing Ito processes. An Ito process is usually defined
via an Ito SDE:

```
  dX = a(t, X_t) dt + b(t, X_t) dW_t

```

where `a(t, x)` is a function taking values in `R^n`, `b(t, X_t)` is a function
taking values in `n x n` matrices. For a complete mathematical definition,
including the regularity conditions that must be imposed on the coefficients
`a(t, X)` and `b(t, X)`, see Ref [1].

#### References:
  [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.

## Classes

[`class ItoProcess`](../../../../tf_quant_finance/models/ItoProcess.md): Interface for specifying Ito processes.

