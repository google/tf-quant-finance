<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.dates.from_np_datetimes" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.experimental.dates.from_np_datetimes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/date_tensor.py">View source</a>



Creates DateTensor from a Numpy array of dtype datetime64.

```python
tf_quant_finance.experimental.dates.from_np_datetimes(
    np_datetimes
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`np_datetimes`</b>: Numpy array of dtype datetime64.


#### Returns:

DateTensor object.


#### Example
```python
import datetime
import numpy as np

date_tensor_np = np.array(
  [[datetime.date(2019, 3, 25), datetime.date(2020, 6, 2)],
   [datetime.date(2020, 9, 15), datetime.date(2020, 12, 27)]],
   dtype=np.datetime64)

date_tensor = from_np_datetimes(date_tensor_np)
```