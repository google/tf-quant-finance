<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.dates.periods.tensor_wrapper.TensorWrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="boolean_mask"/>
<meta itemprop="property" content="broadcast_to"/>
<meta itemprop="property" content="concat"/>
<meta itemprop="property" content="expand_dims"/>
<meta itemprop="property" content="identity"/>
<meta itemprop="property" content="reshape"/>
<meta itemprop="property" content="squeeze"/>
<meta itemprop="property" content="stack"/>
<meta itemprop="property" content="transpose"/>
<meta itemprop="property" content="where"/>
</div>

# tf_quant_finance.experimental.dates.periods.tensor_wrapper.TensorWrapper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>



Base class for Tensor wrappers.

<!-- Placeholder for "Used in" -->

Implements ops that manipulate the backing tensors of Tensor wrappers
(e.g. DateTensor, PeriodTensor). These ops are mostly about reshaping the
backing tensors, such as tf.reshape, tf.expand_dims, tf.stack, etc. Also
includes indexing and slicing.

Inheritors must implement _apply_op(self, op_fn) and provide a static method
_apply_sequence_to_tensor_op(op_fn, tensors). For example:

```python
class MyWrapper(TensorWrapper):
  def __init__(self, backing_tensor):
     self._backing_tensor = backing_tensor

  def _apply_op(self, op_fn):
    new_backing_tensor = op_fn(self._backing_tensor)
    return MyWrapper(new_backing_tensor)

  @staticmethod
  def _apply_sequence_to_tensor_op(op_fn, tensors):
    new_backing_tensor = op_fn([t._backing_tensor for t in tensors])
    return MyWrapper(new_backing_tensor)
```

Then 'MyWrapper` can be used as follows:

```python
m1 = MyWrapper(tf.constant([[1, 2, 3], [4, 5, 6]]))
m2 = MyWrapper(...)
m3 = m1[0, 1:-1]
m4 = m1.expand_dims(axis=-1)
m5 = MyWrapper.concat((m1, m2), axis=-1)
# etc.
```

## Methods

<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
__getitem__(
    key
)
```

Implements indexing.


<h3 id="boolean_mask"><code>boolean_mask</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
boolean_mask(
    mask, axis=None
)
```

See tf.boolean_mask.


<h3 id="broadcast_to"><code>broadcast_to</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
broadcast_to(
    shape
)
```

See tf.broadcast_to.


<h3 id="concat"><code>concat</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
@classmethod
concat(
    cls, tensor_wrappers, axis
)
```

See tf.concat.


<h3 id="expand_dims"><code>expand_dims</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
expand_dims(
    axis
)
```

See tf.expand_dims.


<h3 id="identity"><code>identity</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
identity()
```

See tf.identity.


<h3 id="reshape"><code>reshape</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
reshape(
    shape
)
```

See tf.reshape.


<h3 id="squeeze"><code>squeeze</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
squeeze(
    axis=None
)
```

See tf.squeeze.


<h3 id="stack"><code>stack</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
@classmethod
stack(
    cls, tensor_wrappers, axis=0
)
```

See tf.stack.


<h3 id="transpose"><code>transpose</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
transpose(
    perm=None
)
```

See tf.transpose.


<h3 id="where"><code>where</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/tensor_wrapper.py">View source</a>

```python
@classmethod
where(
    cls, condition, tensor_wrapper_1, tensor_wrapper_2
)
```

See tf.where. Only three-argument version is supported here.




