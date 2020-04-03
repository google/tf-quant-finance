<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.dates.daycounts" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_quant_finance.experimental.dates.daycounts

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/daycounts.py">View source</a>



Day Count Conventions.


Day count conventions are a system for determining how a coupon accumulates over
a coupon period. They can also be seen as a method for converting date
differences to elapsed time. For example, suppose we need to calculate the total
interest accrued over a period of 5 months starting from 6 Jan, 2020 to
8 June, 2020 given that the interest rate is quoted at 4% annually on a
principal of $100. Without the day count convention, we do not know how to
divide the total annual interest of $4 for the five month period. As an example
of the ambiguity, should the pro-rating be done by the total number of days
or by total number of months (or by some other metric)? The answer to this is
provided by assigning a specific day count convention to the quoted rate. For
example, one could use the Money market basis (Actual/360) which states that the
elapsed period for interest accrual between two dates D1 and D2 is the ratio
of the actual number of days between D1 and D2 and 360. For our example, it
leads to `154 / 360 = 0.4278`. Hence the accumulated interest is
`100 * 0.04 * 0.4278 = $1.71. For more details on the many conventions used, see
Ref. [1] and [2].

The functions in this module provide implementations of the commonly used day
count conventions. Some of the conventions also require a knowledge of the
payment schedule to be specified (e.g. Actual/Actual ISMA as in Ref [3] below.).

## References

[1] Wikipedia Contributors. Day Count Conventions. Available at:
  https://en.wikipedia.org/wiki/Day_count_convention
[2] ISDA, ISDA Definitions 2006.
  https://www.isda.org/book/2006-isda-definitions/
[3] ISDA, EMU and Market Conventions: Recent Developments,
  https://www.isda.org/a/AIJEE/1998-ISDA-memo-%E2%80%9CEMU-and-Market-Conventions-Recent-Developments%E2%80%9D.pdf

## Modules

[`dt`](../../../tf_quant_finance/experimental/dates/daycounts/dt.md) module: DateTensor definition.

[`du`](../../../tf_quant_finance/experimental/dates/date_utils.md) module: Utilities for working with dates.

[`periods`](../../../tf_quant_finance/experimental/dates/periods.md) module: PeriodTensor definition.

## Functions

[`actual_360(...)`](../../../tf_quant_finance/experimental/dates/daycounts/actual_360.md): Computes the year fraction between the specified dates.

[`actual_365_actual(...)`](../../../tf_quant_finance/experimental/dates/daycounts/actual_365_actual.md): Computes the year fraction between the specified dates.

[`actual_365_fixed(...)`](../../../tf_quant_finance/experimental/dates/daycounts/actual_365_fixed.md): Computes the year fraction between the specified dates.

[`thirty_360_isda(...)`](../../../tf_quant_finance/experimental/dates/daycounts/thirty_360_isda.md): Computes the year fraction between the specified dates.

