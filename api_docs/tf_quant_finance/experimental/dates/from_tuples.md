<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.dates.from_tuples" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.experimental.dates.from_tuples

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/date_tensor.py">View source</a>



Creates DateTensor from a sequence of year-month-day Tuples.

```python
tf_quant_finance.experimental.dates.from_tuples(
    year_month_day_tuples, validate=True
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`year_month_day_tuples`</b>: Sequence of (year, month, day) Tuples. Months are
  1-based; constants from Months enum can be used instead of ints. Days are
  also 1-based.
* <b>`validate`</b>: Whether to validate the dates.


#### Returns:

DateTensor object.


#### Example
```python
date_tensor = from_tuples([(2015, 4, 15), (2017, 12, 30)])
```