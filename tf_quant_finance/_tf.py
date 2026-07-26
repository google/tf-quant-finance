"""TF-compatibility shim backed by JAX.

Temporary vehicle for the TF->JAX migration: lets files keep using
``import tensorflow as tf`` (swapped to ``from tf_quant_finance import _tf as tf``)
while ``tf.convert_to_tensor`` / ``tf.math.exp`` / ``tf.float32`` / ``tf.while_loop``
etc. resolve to JAX equivalents. Deleted in Phase 6 once modules are natively JAX.

`ponytail:` this is deliberate scaffolding with a known ceiling (semantic gaps in
random key threading and GradientTape); upgrade path = convert modules natively
and drop the import. Not part of the public API.
"""
import sys
import types as _ptypes
import contextlib

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as _jsp
import jax.scipy.linalg as _jspl
import jax.scipy.special as _jsp_special
import jax.lax as _lax
from jax import lax as lax
import jax.ops as _jops

jax.config.update("jax_enable_x64", True)

_this = sys.modules[__name__]


def _drop_name(fn):
    """Wrap a jnp op so it accepts (and ignores) TF's ubiquitous name= kwarg."""
    def wrapper(*args, name=None, **kwargs):
        return fn(*args, **kwargs)
    wrapper.__name__ = getattr(fn, "__name__", "op")
    try:
        import functools
        wrapper = functools.wraps(fn)(wrapper)
    except Exception:
        pass
    return wrapper


# ---------------------------------------------------------------------------
# Top-level math ops: mirror jnp onto this module (tf.exp, tf.zeros, tf.where, ...)
# Wrap callables to tolerate TF's name= kwarg (no jnp op accepts it).
# ---------------------------------------------------------------------------
for _n in dir(jnp):
    if _n.startswith("_"):
        continue
    if hasattr(_this, _n):
        continue
    _v = getattr(jnp, _n)
    if callable(_v) and not isinstance(_v, type) and not hasattr(_v, "__path__"):
        _v = _drop_name(_v)
    setattr(_this, _n, _v)

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------
float16 = jnp.float16
float32 = jnp.float32
float64 = jnp.float64
bfloat16 = jnp.bfloat16
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
bool = jnp.bool_  # tf uses tf.bool, not tf.bool_
complex64 = jnp.complex64
complex128 = jnp.complex128


def as_dtype(x):
    return jnp.dtype(x)


# Type aliases used in annotations across the codebase (tf.Tensor, tf.DType)
Tensor = jnp.ndarray
DType = np.dtype


class TensorSpec:
    def __init__(self, shape=None, dtype=jnp.float32, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name


def Variable(initial_value=None, dtype=None, trainable=True, name=None, **kw):
    """tf.Variable shim: returns a mutable numpy array (for test counters etc.)."""
    import numpy as _np
    d = dtype if dtype is not None else _np.float32
    return _np.asarray(initial_value, dtype=d)


def assign_add(ref, value, **kw):
    """tf.compat.v1.assign_add shim: in-place add on numpy array."""
    import numpy as _np
    ref += _np.asarray(value, dtype=ref.dtype)
    return ref
Module = object



def dtype(x):
    return jnp.asarray(x).dtype


dtypes = _ptypes.SimpleNamespace(
    float16=float16, float32=float32, float64=float64, bfloat16=bfloat16,
    int8=int8, int16=int16, int32=int32, int64=int64,
    uint8=uint8, uint16=uint16, uint32=uint32, uint64=uint64,
    bool=bool, complex64=complex64, complex128=complex128, as_dtype=as_dtype,
    DType=np.dtype,
)

# ---------------------------------------------------------------------------
# tensor construction / conversion (TF ops accept name=/dtype_hint=; jnp doesn't)
# ---------------------------------------------------------------------------
def convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):
    d = dtype if dtype is not None else dtype_hint
    return jnp.asarray(value, dtype=d)


def constant(value, dtype=None, shape=None, name=None):
    v = jnp.asarray(value, dtype=dtype)
    if shape is not None:
        v = jnp.broadcast_to(v, shape)
    return v


def cast(x, dtype, name=None):
    return jnp.asarray(x).astype(dtype)

dtypes.cast = cast


def _drop_name(fn):
    """Wrap a jnp op so it accepts (and ignores) TF's name= kwarg."""
    def wrapper(*args, name=None, **kwargs):
        return fn(*args, **kwargs)
    wrapper.__name__ = getattr(fn, "__name__", "op")
    return wrapper


zeros = _drop_name(jnp.zeros)
ones = _drop_name(jnp.ones)
zeros_like = _drop_name(jnp.zeros_like)
ones_like = _drop_name(jnp.ones_like)
eye = _drop_name(jnp.eye)


def _eye(num_rows, num_columns=None, batch_shape=None, dtype=jnp.float32, name=None):
    e = jnp.eye(num_rows, num_columns, dtype=dtype)
    if batch_shape:
        e = jnp.broadcast_to(e, tuple(batch_shape) + e.shape)
    return e


eye = _eye
fill = _drop_name(jnp.full)
# tf.range(start, limit=None, delta=1, dtype=None, name=None) -> jnp.arange
def _tf_range(start=None, limit=None, delta=None, dtype=None, name=None, **kw):
    # tf.range(start, limit=None, delta=1, dtype) -> jnp.arange(start, stop, step)
    del name, kw
    if isinstance(start, (int, float)) and limit is None and delta is None:
        # tf.range(n) -> 0..n-1
        return jnp.arange(start, dtype=dtype)
    if delta is None:
        return jnp.arange(start, limit, dtype=dtype)
    return jnp.arange(start, limit, delta, dtype=dtype)


range = _tf_range
linspace = _drop_name(jnp.linspace)



def is_tensor(x):
    return isinstance(x, (np.ndarray, jnp.ndarray))


def get_static_value(x):
    try:
        return np.asarray(x)
    except Exception:
        return None


def executing_eagerly():
    return True


newaxis = None  # tf.newaxis

# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------

class TensorShape:
    """Minimal stand-in for tf.TensorShape: a tuple of dims (all static)."""

    def __init__(self, dims):
        if dims is None:
            self._dims = None
        elif isinstance(dims, TensorShape):
            self._dims = tuple(dims._dims)
        else:
            self._dims = tuple(int(d) if d is not None else None for d in dims)

    def __iter__(self):
        return iter(self._dims)

    def __len__(self):
        return len(self._dims)

    def __getitem__(self, i):
        return self._dims[i]

    def as_list(self):
        return list(self._dims)

    @property
    def rank(self):
        return len(self._dims)

    def is_fully_defined(self):
        import builtins
        return builtins.all(d is not None for d in self._dims)

    def num_elements(self):
        n = 1
        for d in self._dims:
            n *= d
        return n

    def __repr__(self):
        return f"TensorShape({self._dims})"


def shape(input, out_type=None, name=None):
    # tf.shape returns a (dynamic) shape tensor; in JAX shapes are static.
    out_type = out_type or jnp.int32
    return jnp.asarray(jnp.asarray(input).shape, dtype=out_type)


def size(x, out_type=None, name=None):
    return int(jnp.asarray(x).size)


def rank(x):
    return int(jnp.asarray(x).ndim)


# ---------------------------------------------------------------------------
# name_scope: no-op context manager (JAX has no graph names)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def name_scope(*args, **kwargs):
    yield


# ---------------------------------------------------------------------------
# compat / types / nn / sparse / xla stub namespaces
# ---------------------------------------------------------------------------
compat = _this  # tf.compat.v1 / v2 -> resolves back to the same surface
compat.v1 = _this  # type: ignore[attr-defined]
compat.v2 = _this  # type: ignore[attr-defined]


class _TypesNS:
    experimental = _ptypes.SimpleNamespace(
        TensorLike=object  # type alias only; replaced in types/data_types.py
    )


types = _TypesNS()

nn = _ptypes.SimpleNamespace(
    relu=jax.nn.relu, sigmoid=jax.nn.sigmoid, softmax=jax.nn.softmax,
    softplus=jax.nn.softplus, gelu=jax.nn.gelu, log_softmax=jax.nn.log_softmax,
)
sparse = _ptypes.SimpleNamespace()
xla = _ptypes.SimpleNamespace()
nest = _ptypes.SimpleNamespace(
    map_structure=lambda f, s: jax.tree_util.tree_map(f, s),
    flatten=lambda s: jax.tree_util.tree_leaves(s),
)
data = _ptypes.SimpleNamespace()


class _ProtoMsg:
    """Stub for TF protobuf message classes (tf.train.Example etc.).
    ponytail: real serialization is unimplemented; only here so experimental.io
    imports under JAX. Reimplement with numpy/plain files if needed (Phase 3)."""
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def SerializeToString(self):
        return b""
    def ParseFromString(self, data):
        return None


GraphDef = _ProtoMsg  # tf.compat.v1.GraphDef placeholder


train = _ptypes.SimpleNamespace(
    Example=_ProtoMsg, Feature=_ProtoMsg, Features=_ProtoMsg,
    BytesList=_ProtoMsg, FloatList=_ProtoMsg, Int64List=_ProtoMsg,
)


class _TFRecordWriter:
    def __init__(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs):
        pass
    def close(self):
        pass


io = _ptypes.SimpleNamespace(
    TFRecordWriter=_TFRecordWriter,
    TFRecordDataset=lambda *a, **k: iter(()),
    gfile=lambda *a, **k: None,
)
summary = _ptypes.SimpleNamespace()


# ---------------------------------------------------------------------------
# tf.TensorArray: register as a JAX pytree so it is a valid (functional) carry
# for lax.while_loop/scan. This lets the models' while_loop+TensorArray sampling
# paths run unchanged via the shim instead of per-model rewrites.
# ---------------------------------------------------------------------------
class TensorArray:
    def __init__(self, dtype=None, size=None, element_shape=None, dynamic_size=None,
                 clear_after_read=None, infer_shape=None, name=None, **kw):
        self.dtype = jnp.dtype(dtype) if dtype is not None else jnp.float32
        self.size = int(size) if size is not None else 0
        es = element_shape
        if es is None:
            self.element_shape = ()
        elif hasattr(es, "shape"):
            self.element_shape = tuple(np.asarray(es).tolist())
        else:
            self.element_shape = tuple(int(d) for d in es)
        self._data = jnp.zeros((self.size,) + self.element_shape, dtype=self.dtype)

    def write(self, index, value):
        import copy
        new = copy.copy(self)
        new._data = self._data.at[index].set(jnp.asarray(value, dtype=self.dtype))
        return new

    def read(self, index):
        return self._data[index]

    def stack(self):
        return self._data

    def unstack(self, value):
        import copy
        new = copy.copy(self)
        new._data = jnp.asarray(value)
        return new

    def tree_flatten(self):
        return (self._data,), (self.dtype, self.size, self.element_shape)

    @classmethod
    def tree_unflatten(cls, aux, children):
        new = cls.__new__(cls)
        new.dtype, new.size, new.element_shape = aux
        new._data = children[0]
        return new


jax.tree_util.register_pytree_node_class(TensorArray)


# ---------------------------------------------------------------------------
# tf.math.* namespace
# ---------------------------------------------------------------------------
math = _ptypes.SimpleNamespace()
for _n in dir(jnp):
    if _n.startswith("_"):
        continue
    _v = getattr(jnp, _n)
    if callable(_v) and not isinstance(_v, type) and not hasattr(_v, "__path__"):
        _v = _drop_name(_v)
    setattr(math, _n, _v)
# tf.math.reduce_* -> jnp without the reduce_ prefix
math.reduce_sum = jnp.sum
math.reduce_mean = jnp.mean
math.reduce_max = jnp.max
math.reduce_min = jnp.min
math.reduce_prod = jnp.prod
math.reduce_any = jnp.any
math.reduce_all = jnp.all
math.reduce_std = jnp.std
math.reduce_variance = jnp.var
math.reduce_logsumexp = _jsp.special.logsumexp
# tf.math uses is_nan/is_finite (underscores); jnp uses isnan/isfinite
math.is_nan = jnp.isnan
math.is_finite = jnp.isfinite
is_nan = jnp.isnan
is_finite = jnp.isfinite
math.cast = cast
# special funcs
math.erf = _jsp_special.erf
math.erfc = _jsp_special.erfc
math.erfinv = _jsp_special.erfinv
math.igamma = _jsp_special.gammainc
math.igammac = _jsp_special.gammaincc
math.sigmoid = jax.nn.sigmoid
math.cumprod = jnp.cumprod
def _cumsum(x, axis=0, exclusive=False, reverse=False, name=None):
    x = jnp.asarray(x)
    if reverse:
        x = jnp.flip(x, axis)
    out = jnp.cumsum(x, axis=axis)
    if exclusive:
        # drop the first element along axis, prepend a zero
        import builtins
        zeros_shape = list(out.shape)
        zeros_shape[axis] = 1
        z = jnp.zeros(zeros_shape, dtype=out.dtype)
        out = jnp.concatenate([z, out], axis=axis)
        sl = [builtins.slice(None)] * out.ndim
        sl[axis] = builtins.slice(None, -1)
        out = out[tuple(sl)]
    if reverse:
        out = jnp.flip(out, axis)
    return out


math.cumsum = _cumsum
cumsum = _cumsum
math.top_k = lambda a, k=1, sorted=True: jnp.argsort(a)[-k:]  # best-effort
def _divide_no_nan(x, y, name=None):
    # ponytail: safe denom avoids JAX '0*inf=nan' in the masked-division VJP.
    y = jnp.asarray(y)
    safe_y = jnp.where(y != 0, y, 1.0)
    return jnp.where(y != 0, x / safe_y, 0.0)


math.divide_no_nan = _divide_no_nan
divide_no_nan = _divide_no_nan
math.segment_sum = lambda data, segments, **kw: _jops.segment_sum(data, segments)
math.segment_prod = lambda data, segments, **kw: _jops.segment_prod(data, segments)
math.nextafter = _np_nextafter = np.nextafter
# ponytail: tf.math.brentq has no jax builtin; delegate to the repo's own root
# search or scipy. Wired when a caller actually needs it.
def _brentq_missing(*args, **kwargs):
    raise NotImplementedError("tf.math.brentq: use jax.scipy or repo root_search")
math.brentq = _brentq_missing


# ---------------------------------------------------------------------------
# tf.linalg.* namespace
# ---------------------------------------------------------------------------
def _matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, name=None, **kw):
    if transpose_a or adjoint_a:
        a = jnp.swapaxes(a, -1, -2)
    if transpose_b or adjoint_b:
        b = jnp.swapaxes(b, -1, -2)
    return jnp.matmul(a, b)


linalg = _ptypes.SimpleNamespace(
    matmul=_matmul,
    matvec=lambda m, v, **kw: jnp.matmul(m, v[..., None])[..., 0],
    cholesky=jnp.linalg.cholesky,
    inv=jnp.linalg.inv,
    pinv=jnp.linalg.pinv,
    eigh=jnp.linalg.eigh,
    eigvalsh=jnp.linalg.eigvalsh,
    eig=jnp.linalg.eig,
    svd=jnp.linalg.svd,
    norm=jnp.linalg.norm,
    einsum=jnp.einsum,
    det=jnp.linalg.det,
    solve=jnp.linalg.solve,
    qr=jnp.linalg.qr,
    tensor_diag=lambda v, **kw: _create_diag(v),
    diag=lambda v, k=0, **kw: _create_diag(v, k),
    set_diag=lambda m, v, **kw: m.at[..., :].set(v) if hasattr(m, "at") else m,
    tridiagonal_solve=_lax.linalg.tridiagonal_solve if hasattr(_lax.linalg, "tridiagonal_solve") else None,
    tridiagonal_matmul=None,
    expm=_jspl.expm,
)
def _band_part(m, num_lower, num_upper):
    # tf.linalg.band_part(m, num_lower, num_upper): keep `num_lower` sub-diagonals
    # and `num_upper` super-diagonals. num_lower<0 = all below, num_upper<0 = all above.
    m = jnp.asarray(m)
    n = m.shape[-1]
    row = jnp.arange(n)[:, None]   # (n, 1)
    col = jnp.arange(n)[None, :]   # (1, n)
    mask = ((num_lower < 0) | ((row - col) <= num_lower)) & \
           ((num_upper < 0) | ((col - row) <= num_upper))
    if m.ndim > 2:
        mask = jnp.broadcast_to(mask, m.shape)
    return m * mask
linalg.band_part = _band_part


def _create_diag(v, k=0):
    """tf.linalg.diag: create a (batched) diagonal matrix from `v`.

    jnp.diag only creates from 1-D input and extracts from 2-D; TF creates
    (batched) diagonal matrices for any rank."""
    v = jnp.asarray(v)
    if v.ndim == 1:
        return jnp.diag(v, k)
    n = v.shape[-1]
    eye = jnp.eye(n, dtype=v.dtype)
    return v[..., :, None] * eye  # [..., n, n]


linalg.diag = _create_diag


def _tridiagonal_matmul(diagonals, rhs, diagonals_format='sequence', **kw):
    """tf.linalg.tridiagonal_matmul: multiply tridiagonal matrix by rhs."""
    if isinstance(diagonals, (tuple, list)) and len(diagonals) == 3:
        super_d, diag, sub = (jnp.asarray(d) for d in diagonals)
    else:
        d = jnp.asarray(diagonals)
        super_d, diag, sub = d[..., 0, :], d[..., 1, :], d[..., 2, :]
    rhs = jnp.asarray(rhs)
    result = diag[..., :, None] * rhs
    # super[i] = M[i,i+1]: result[i] += super[i]*rhs[i+1]  (for i < m-1)
    result = result.at[..., :-1, :].add(super_d[..., :-1, None] * rhs[..., 1:, :])
    # sub[i] = M[i+1,i]: result[i+1] += sub[i]*rhs[i]  (for i < m-1)
    result = result.at[..., 1:, :].add(sub[..., :-1, None] * rhs[..., :-1, :])
    return result


linalg.tridiagonal_matmul = _tridiagonal_matmul
linalg.tensor_diag = _create_diag


def _tridiagonal_solve(diagonals, rhs, partial_pivots=True,
                      perturbation_singular=0.0, name=None, **kw):
    # TF: diagonals = (superdiag, diag, subdiag) or matrix [...,3,k].
    # jax 0.9.2 lax.linalg.tridiagonal_solve(dl, d, du, b) requires:
    #   - dl/d/du all SAME shape [..., m] (NOT m-1)
    #   - b shape [..., m, nrhs] (always needs the trailing nrhs dim)
    del partial_pivots, perturbation_singular, name, kw
    import jax.lax as _ll
    if isinstance(diagonals, (tuple, list)) and len(diagonals) == 3:
        super_d, diag, sub = (jnp.asarray(d) for d in diagonals)
    else:
        d = jnp.asarray(diagonals)
        super_d, diag, sub = d[..., 0, :], d[..., 1, :], d[..., 2, :]
    rhs = jnp.asarray(rhs)
    # Add trailing nrhs=1 dim if rhs lacks it (jax requires [..., m, nrhs]).
    squeeze_rhs = (rhs.ndim <= diag.ndim)
    if squeeze_rhs:
        rhs = rhs[..., None]
    out = _ll.linalg.tridiagonal_solve(sub, diag, super_d, rhs)
    if squeeze_rhs:
        out = out[..., 0]
    return out


linalg.tridiagonal_solve = _tridiagonal_solve
# ponytail: LinearOperator* (2 uses) not shimmed; convert those call sites natively.
linalg.LinearOperatorFullMatrix = NotImplementedError
linalg.LinearOperatorBlockDiag = NotImplementedError


# ---------------------------------------------------------------------------
# tf.random.* : functional PRNG. Uses a module-global key (thread-unsafe).
# ponytail: ceiling = single global key is not reproducible under parallelism;
# callers should migrate to explicit PRNGKey args (Phase 2).
# ---------------------------------------------------------------------------
_global_key = jax.random.PRNGKey(0)


def _next_key():
    global _global_key
    _global_key, k = jax.random.split(_global_key)
    return k


def _to_key(seed):
    """Coerce a TF-style seed (Python int, [2]-array, or jax key) to a PRNGKey."""
    if seed is None:
        return _next_key()
    if isinstance(seed, (int, np.integer)):
        return jax.random.PRNGKey(int(seed))
    seed = np.asarray(seed)
    if seed.shape == (2,) and np.issubdtype(seed.dtype, np.integer):
        return jax.random.PRNGKey(int(seed[0]))  # ponytail: use first component
    if seed.ndim == 1 and seed.shape[0] == 2:
        return seed
    return jax.random.PRNGKey(0)


def _stateless_uniform(shape, seed, minval=0.0, maxval=1.0, dtype=jnp.float32, name=None, **kw):
    return jax.random.uniform(_to_key(seed), shape, minval=minval, maxval=maxval, dtype=dtype)


def _stateless_normal(shape, seed, mean=0.0, stddev=1.0, dtype=jnp.float32, name=None, **kw):
    return jax.random.normal(_to_key(seed), shape, dtype=dtype) * stddev + mean


def _stateless_gamma(shape, seed, alpha, dtype=jnp.float32, name=None, **kw):
    return jax.random.gamma(_to_key(seed), alpha, shape, dtype=dtype)


def _stateless_poisson(shape, seed, lam, dtype=jnp.float32, name=None, **kw):
    return jax.random.poisson(_to_key(seed), lam, shape, dtype=dtype)


random = _ptypes.SimpleNamespace(
    set_seed=lambda s: globals().__setitem__("_global_key", jax.random.PRNGKey(int(s))),
    uniform=lambda shape, minval=0.0, maxval=1.0, dtype=jnp.float32, seed=None, name=None: jax.random.uniform(_to_key(seed), shape, minval=minval, maxval=maxval, dtype=dtype),
    normal=lambda shape, mean=0.0, stddev=1.0, dtype=jnp.float32, seed=None, name=None: jax.random.normal(_to_key(seed), shape, dtype=dtype) * stddev + mean,
    gamma=lambda shape, alpha, dtype=jnp.float32, seed=None, name=None: jax.random.gamma(_to_key(seed), alpha, shape, dtype=dtype),
    poisson=lambda lam, shape, dtype=jnp.float32, seed=None, name=None: jax.random.poisson(_to_key(seed), lam, shape, dtype=dtype),
    shuffle=lambda value, seed=None: jax.random.permutation(_to_key(seed), value),
    stateless_uniform=_stateless_uniform,
    stateless_normal=_stateless_normal,
    stateless_gamma=_stateless_gamma,
    stateless_poisson=_stateless_poisson,
)


# ---------------------------------------------------------------------------
# control flow
# ---------------------------------------------------------------------------
def while_loop(cond, body, loop_vars, parallel_iterations=None, maximum_iterations=None,
               swap_memory=None, name=None, return_same_structure=None,
               shape_invariants=None):
    del parallel_iterations, swap_memory, name, return_same_structure, shape_invariants
    # TF convention: cond/body are called as cond(*loop_vars). lax.while_loop
    # passes the carry as a single positional, so adapt by (un)packing.
    # TF unpacks the carry into cond(*loop_vars). A single ndarray is one arg;
    # tuples/lists are unpacked. Dataclass-like objects (registered pytrees via
    # @utils.dataclass) are treated as single args (not unpacked).
    is_seq = isinstance(loop_vars, (list, tuple))
    vars_tuple = tuple(loop_vars) if is_seq else (loop_vars,)

    def cond_jax(carry):
        return cond(*carry) if is_seq else cond(carry)

    def body_jax(carry):
        return body(*carry) if is_seq else body(carry)

    if maximum_iterations is not None:
        maxit = maximum_iterations

        def cond2(carry):
            i, rest = carry[0], carry[1:]
            c = cond(*rest) if is_seq else cond(rest[0])
            return jnp.logical_and(i < maxit, c)

        def body2(carry):
            i, rest = carry[0], carry[1:]
            nxt = body(*rest) if is_seq else body(rest[0])
            nxt_t = tuple(nxt) if isinstance(nxt, (list, tuple)) else (nxt,)
            return (i + 1,) + nxt_t

        out = _lax.while_loop(cond2, body2, (jnp.asarray(0),) + vars_tuple)
        rest = out[1:]
        return rest if (is_seq and len(rest) > 1) else rest[0]

    out = _lax.while_loop(cond_jax, body_jax, vars_tuple)
    return out if is_seq else out[0]


def cond(pred, true_fn, false_fn, *args, **kw):
    # tf.cond signature varies; route to lax.cond lazily.
    return _lax.cond(pred, true_fn, false_fn)


def scan(f, init, xs=None, reverse=False, **kw):
    return _lax.scan(f, init, xs, reverse=reverse)


def map_fn(f, elems, dtype=None, **kw):
    return jax.vmap(f)(elems)


def vectorized_map(f, xs, fallback_to_while_loop=True, **kw):
    return jax.vmap(f)(xs)


def function(func=None, input_signature=None, **kw):
    # tf.function: drop jit by default (callers can add @jax.jit explicitly).
    def _identity(f):
        return f
    if func is None:
        return _identity
    return func


def py_function(func=None, **kw):
    def _wrap(f):
        return f
    if func is None:
        return _wrap
    return func


# ---------------------------------------------------------------------------
# tf.test — back TestCase by absltest so @parameterized works and we get
# assertAllClose / assertNear natively across the repo. Repo tests use the
# pattern `class X(parameterized.TestCase, tf.test.TestCase)`, which requires
# this base to be absltest (not parameterized) to keep a consistent MRO.
# ---------------------------------------------------------------------------
import unittest as _unittest
from absl.testing import absltest as _absltest


class TestCase(_absltest.TestCase):
    """tf.test.TestCase stand-in: absltest.TestCase + tf-style evaluate()."""

    def evaluate(self, tensors):
        # Preserve namedtuples and dataclass-like objects; only convert
        # plain lists/tuples and leaf tensors to numpy.
        if hasattr(tensors, "_fields"):
            return type(tensors)(*[self.evaluate(v) for v in tensors])
        if hasattr(tensors, "__attrs_attrs__"):
            return type(tensors)(*[self.evaluate(getattr(tensors, a.name)) for a in tensors.__attrs_attrs__])
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(np.asarray(t) for t in tensors)
        return np.asarray(tensors)

    # TF-specific asserts that absltest.TestCase lacks.
    def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=rtol, atol=atol, err_msg=msg)

    def assertNear(self, a, b, err, msg=None):
        self.assertLess(abs(float(np.asarray(a)) - float(np.asarray(b))), err, msg=msg)

    def assertAllEqual(self, a, b, msg=None):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=msg)

    def assertArrayNear(self, a, b, tol, msg=None):
        np.testing.assert_allclose(np.asarray(a, dtype=float),
                                   np.asarray(b, dtype=float),
                                   rtol=tol, atol=tol, err_msg=msg)

    def assertNDArrayNear(self, a, b, tol, msg=None):
        np.testing.assert_allclose(np.asarray(a, dtype=float),
                                   np.asarray(b, dtype=float),
                                   rtol=tol, atol=tol, err_msg=msg)

    def assertAllFinite(self, a, msg=None):
        self.assertTrue(np.all(np.isfinite(np.asarray(a))), msg=msg)

    def assertShapeEqual(self, a, b, msg=None):
        self.assertEqual(np.asarray(a).shape, np.asarray(b).shape, msg=msg)

    def assertDTypeEqual(self, a, dtype, msg=None):
        self.assertEqual(np.asarray(a).dtype, dtype, msg=msg)

    def assertDeviceEqual(self, *a, **k):
        pass

    def get_temp_dir(self):
        import tempfile
        return tempfile.gettempdir()

    def cached_session(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()

    def session(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()


test = _ptypes.SimpleNamespace(
    TestCase=TestCase,
    main=_unittest.main,
    SkipTest=_unittest.SkipTest,
    TestCase_source=TestCase,
)


# tf.python.framework.test_util: graph/eager mode decorators. JAX is eager-only,
# so these are no-ops.
def _cls_noop(x=None):
    """Class decorator that works as @deco, @deco(), or @deco('reason')."""
    if x is None or isinstance(x, str):
        return lambda c: c
    return x


def _fn_noop(fn=None, *args, **kw):
    if fn is None or isinstance(fn, str):
        return lambda f: f
    return fn


test_util = _ptypes.SimpleNamespace(
    run_all_in_graph_and_eager_modes=_cls_noop,
    run_in_graph_and_eager_modes=_fn_noop,
    run_deprecated_v1=_cls_noop,
    run_v1_only=_cls_noop,
    run_in_graph_mode=_fn_noop,
    run_in_eager_mode=_fn_noop,
    deprecated_graph_mode_only=_fn_noop,
)


# ---------------------------------------------------------------------------
# debugging: assertions are no-ops under JAX (or raise eagerly on static vals)
# ---------------------------------------------------------------------------
def _chk(cond, msg=None):
    """Eager validation: raise InvalidArgumentError if cond is concretely False;
    no-op if cond is a tracer (can't evaluate under jit)."""
    try:
        ok = bool(np.asarray(cond).all())
    except Exception:
        return None
    if not ok:
        raise InvalidArgumentError(msg or "assertion failed")
    return None


debugging = _ptypes.SimpleNamespace(
    assert_positive=lambda x, message=None, **k: _chk(np.asarray(x) > 0, message),
    assert_non_negative=lambda x, message=None, **k: _chk(np.asarray(x) >= 0, message),
    assert_negative=lambda x, message=None, **k: _chk(np.asarray(x) < 0, message),
    assert_non_positive=lambda x, message=None, **k: _chk(np.asarray(x) <= 0, message),
    assert_less=lambda a, b, message=None, **k: _chk(np.asarray(a) < np.asarray(b), message),
    assert_less_equal=lambda a, b, message=None, **k: _chk(np.asarray(a) <= np.asarray(b), message),
    assert_greater=lambda a, b, message=None, **k: _chk(np.asarray(a) > np.asarray(b), message),
    assert_greater_equal=lambda a, b, message=None, **k: _chk(np.asarray(a) >= np.asarray(b), message),
    assert_equal=lambda a, b, message=None, **k: _chk(np.asarray(a) == np.asarray(b), message),
    assert_none_equal=lambda a, b, message=None, **k: _chk(np.asarray(a) != np.asarray(b), message),
    assert_all_finite=lambda x, message=None, **k: _chk(np.isfinite(np.asarray(x, dtype=float)), message),
    assert_near=lambda a, b, rtol=None, atol=None, message=None, **k: _chk(
        np.isclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                  rtol=rtol or 1e-6, atol=atol or 0.0), message),
    assert_all_close=lambda a, b, rtol=None, atol=None, message=None, **k: _chk(
        np.isclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                  rtol=rtol or 1e-6, atol=atol or 1e-6), message),
    assert_rank=lambda *a, **k: None, assert_type=lambda *a, **k: None,
    is_strictly_increasing=lambda x, message=None, **k: _chk(jnp.all(jnp.diff(jnp.asarray(x)) > 0), message),
    is_non_decreasing=lambda x, message=None, **k: _chk(jnp.all(jnp.diff(jnp.asarray(x)) >= 0), message),
)


def _assert_noop(condition, data, summarize=None, name=None):
    # tf.debugging.Assert: in TF graph mode this raises at runtime. Under JAX/eager
    # we validate eagerly when the condition is concrete (so assertRaises tests
    # pass); traced conditions can't be evaluated, so no-op.
    try:
        ok = bool(np.asarray(condition).all())
    except Exception:
        return None
    if not ok:
        raise InvalidArgumentError(repr(data))
    return None


debugging.Assert = _assert_noop
Assert = _assert_noop
compat.v1.debugging = debugging
assert_equal = debugging.assert_equal
assert_greater = debugging.assert_greater
assert_less = debugging.assert_less
assert_less_equal = debugging.assert_less_equal
assert_greater_equal = debugging.assert_greater_equal


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class Error(Exception):
    pass


class InvalidArgumentError(ValueError):
    pass


class NotFoundError(ValueError):
    pass


class UnimplementedError(RuntimeError):
    pass


errors = _ptypes.SimpleNamespace(
    InvalidArgumentError=InvalidArgumentError,
    NotFoundError=NotFoundError,
    UnimplementedError=UnimplementedError,
    Error=Error,
    AbortError=RuntimeError,
)


class _UnconnectedGradients:
    NONE = "none"
    ZERO = "zero"


UnconnectedGradients = _UnconnectedGradients


# ---------------------------------------------------------------------------
# gradients (best-effort; real rewrites in Phase 2/3)
# ---------------------------------------------------------------------------
class GradientTape:
    """Minimal shim: records watched tensors, computes grad via jax.grad on .grad().
    ponytail: only supports the single-output scalar case via jax.grad; callers
    needing jacobians/hessians get rewritten natively."""

    def __init__(self, persistent=False, watch_accessed_variables=True):
        self._persistent = persistent
        self._watched = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def watch(self, t):
        self._watched.append(t)

    def gradient(self, target, sources, output_gradients=None):
        def fn(x):
            return jnp.sum(target) if not callable(target) else target()
        # best-effort: treat sources as a single arg
        if isinstance(sources, (list, tuple)):
            grads = jax.grad(fn)(*sources) if len(sources) > 1 else jax.grad(fn)(sources[0])
            return grads
        return jax.grad(fn)(sources)


def gradients(ys, xs, **kw):
    return jax.grad(lambda x: jnp.sum(ys))(xs) if not isinstance(xs, (list, tuple)) else jax.grad(lambda *a: jnp.sum(ys))(*xs)


def custom_gradient(f):
    """tf.custom_gradient shim: call f and drop the (value, grad_fn) tuple,
    returning just the value. JAX autodiff replaces the custom grad."""
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and callable(result[1]):
            return result[0]
        return result
    return wrapper


# ---------------------------------------------------------------------------
# misc helpers used in places
# ---------------------------------------------------------------------------
def repeat(input, repeats, axis=None):
    return jnp.repeat(input, repeats, axis=axis)


# tf.gather(params, indices, axis) -> jnp.take (this jax has no jnp.gather).
def gather(params, indices, axis=0, batch_dims=0, name=None,
           validate_indices=None):
    del name, validate_indices
    params = jnp.asarray(params)
    indices = jnp.asarray(indices)
    if batch_dims:
        # Pair the leading `batch_dims` axes of params and indices, gather along `axis`.
        def _g(p, idx):
            if axis == 0:
                return p[idx]
            # Axis relative to the de-batched array: positive axes subtract
            # batch_dims (leading dims removed by vmap); negative axes unchanged
            # (trailing dims unaffected).
            rel_axis = axis - batch_dims if axis >= 0 else axis
            return jnp.take(p, idx, axis=rel_axis)
        for _ in range(batch_dims):
            _g = jax.vmap(_g)
        return _g(params, indices)
    return jnp.take(params, indices, axis=axis)


def searchsorted(sorted_seq, values, side='left', out_type=None, name=None):
    sorted_seq = jnp.asarray(sorted_seq)
    values = jnp.asarray(values)
    if sorted_seq.ndim <= 1:
        return jnp.searchsorted(sorted_seq, values, side=side)
    # Batched: vmap a 1-D searchsorted over the shared leading batch axes.
    v = lambda s, val: jnp.searchsorted(s, val, side=side)
    for _ in range(sorted_seq.ndim - 1):
        v = jax.vmap(v)
    return v(sorted_seq, values)


# tf.concat(values, axis): accept a sequence OR positional tensors, and coerce
# list elements to arrays (jnp.concatenate rejects raw lists).
# When concatenating shapes (empty + int dims), keep integer dtype instead of
# letting an empty array default to float64 (common TF->JAX shape bug).
def concat(values, axis=0, name=None, *more, **kwargs):
    if isinstance(values, (list, tuple)):
        seq = values
    elif more:
        seq = (values,) + more
    else:
        try:
            seq = list(values)  # generator / iterable (TF accepted these)
        except TypeError:
            seq = [values]
    arrs = [jnp.asarray(v) for v in seq]
    import builtins
    has_nonempty = False
    all_int_compatible = True
    for a in arrs:
        if a.size:
            has_nonempty = True
        if not (a.size == 0 or np.issubdtype(a.dtype, np.integer)):
            all_int_compatible = False
    if has_nonempty and all_int_compatible:
        arrs = [a.astype(jnp.int32) for a in arrs]
    return jnp.concatenate(arrs, axis=axis)


stack = _drop_name(jnp.stack)


def _pad(tensor, paddings, mode='CONSTANT', name=None, constant_values=0, **kw):
    mode_map = {'CONSTANT': 'constant', 'REFLECT': 'reflect', 'SYMMETRIC': 'symmetric'}
    return jnp.pad(tensor, jnp.asarray(paddings), mode=mode_map.get(str(mode).upper(), str(mode).lower()))


pad = _pad

# tf.transpose uses `perm=`; jnp uses `axes=`.
def transpose(a, perm=None, name=None, conjugate=False):
    return jnp.transpose(a, axes=perm)


matmul = _matmul


floor_div = jnp.floor_divide
realdiv = jnp.true_divide
cumprod = jnp.cumprod
argmax = jnp.argmax
argmin = jnp.argmin
is_finite = jnp.isfinite
is_nan = jnp.isnan
is_inf = jnp.isinf
logical_not = jnp.logical_not
logical_or = jnp.logical_or
logical_and = jnp.logical_and
logical_xor = jnp.logical_xor
floormod = jnp.mod
truediv = jnp.true_divide
unsorted_segment_max = lambda data, segment_ids, num_segments=None, **kw: jax.ops.segment_max(data, segment_ids)
unsorted_segment_sum = lambda data, segment_ids, num_segments=None, **kw: jax.ops.segment_sum(data, segment_ids)
unsorted_segment_min = lambda data, segment_ids, num_segments=None, **kw: jax.ops.segment_min(data, segment_ids)
unsorted_segment_prod = lambda data, segment_ids, num_segments=None, **kw: jax.ops.segment_prod(data, segment_ids)


def _unique(values, out_idx=None, name=None):
    """tf.unique returns (unique_values, idx). jnp.unique returns only values."""
    values = jnp.asarray(values)
    # jnp.unique with return_index=True returns (unique, indices, counts)
    # but indices are of first occurrence, not the TF mapping. Use a manual approach.
    uniq = jnp.unique(values)
    # Build idx: for each element in values, find its index in uniq.
    idx = jnp.searchsorted(uniq, values)
    return uniq, idx


unique = _unique


gather_nd = jnp.take  # best-effort; callers needing advanced gather_nd convert natively


def _gather_nd(params, indices, name=None, batch_dims=0, **kw):
    """tf.gather_nd: gather elements at N-dimensional indices.
    indices shape [..., num_dims] -> output shape indices.shape[:-1] + params.shape[num_dims:]."""
    params = jnp.asarray(params)
    indices = jnp.asarray(indices)
    if indices.ndim == 1:
        return params[tuple(indices)]
    # Split indices into per-dimension arrays and use tuple indexing
    num_dims = indices.shape[-1]
    idx_tuple = tuple(indices[..., d] for d in range(num_dims))
    return params[idx_tuple]


gather_nd = _gather_nd


one_hot = _drop_name(jax.nn.one_hot)
reverse = _drop_name(jnp.flip)


def boolean_mask(tensor, mask, axis=None, name=None):
    # tf.boolean_mask: select elements where mask is True (mask matches leading dims).
    mask = jnp.asarray(mask)
    tensor = jnp.asarray(tensor)
    if axis is None:
        return tensor[mask]
    return jnp.compress(mask, tensor, axis=axis)


def _complex(real, imag=None, name=None):
    if imag is None:
        return jnp.asarray(real, dtype=jnp.complex128)
    return jnp.asarray(real, dtype=jnp.complex128) + 1j * jnp.asarray(imag)


complex = _complex


def tensor_scatter_nd_update(tensor, indices, updates, name=None):
    tensor = jnp.asarray(tensor)
    indices = jnp.asarray(indices)
    if indices.ndim <= 1:
        return tensor.at[indices].set(jnp.asarray(updates))
    idx_tuple = tuple(indices[:, d] for d in range(indices.shape[-1]))
    return tensor.at[idx_tuple].set(jnp.asarray(updates))


def placeholder_with_default(input, shape=None, name=None):
    return jnp.asarray(input)


def _global_variables_initializer():
    return None


def _global_variables():
    return []


compat.v1.global_variables_initializer = _global_variables_initializer
compat.v1.global_variables = _global_variables
def _scatter_nd(indices, updates, shape, name=None):
    updates = jnp.asarray(updates)
    out = jnp.zeros(shape, dtype=updates.dtype)
    indices = jnp.asarray(indices)
    if indices.ndim <= 1:
        return out.at[indices].set(updates)
    # indices shape [N, ndim]: N index tuples -> tuple of ndim index arrays.
    idx_tuple = tuple(indices[:, d] for d in range(indices.shape[-1]))
    return out.at[idx_tuple].set(updates)


scatter_nd = _scatter_nd
bitwise = _ptypes.SimpleNamespace(
    left_shift=jnp.left_shift, right_shift=jnp.right_shift,
    bitwise_and=jnp.bitwise_and, bitwise_or=jnp.bitwise_or,
    bitwise_xor=jnp.bitwise_xor, invert=jnp.bitwise_not,
)


def tensordot(a, b, axes):
    return jnp.tensordot(a, b, axes)


def einsum(*args, **kw):
    return jnp.einsum(*args, **kw)


def sort_(a, axis=-1, direction="ASCENDING"):
    return jnp.sort(a, axis=axis)


def identity(a):
    return a


def add_n(tensors):
    out = tensors[0]
    for t in tensors[1:]:
        out = out + t
    return out


def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False, name=None):
    # ponytail: real TensorProto serialization unimplemented (experimental.io only);
    # returns the array so the module imports. Reimplement if IO is needed.
    return np.asarray(values)


def make_ndarray(tensor_proto):
    return np.asarray(tensor_proto)


# ---------------------------------------------------------------------------
# reduce_* at top level (tf.reduce_sum etc. -> jnp.sum etc.)
# TF spells the first arg `input_tensor=`; accept it.
# ---------------------------------------------------------------------------
def _make_reduce(fn):
    def wrapper(input_tensor=None, axis=None, keepdims=False, name=None,
               **kwargs):
        if input_tensor is None:
            input_tensor = kwargs.pop("x", kwargs.pop("tensor", None))
        return fn(input_tensor, axis=axis, keepdims=keepdims)
    wrapper.__name__ = getattr(fn, "__name__", "reduce")
    return wrapper


reduce_sum = _make_reduce(jnp.sum)
reduce_mean = _make_reduce(jnp.mean)
reduce_max = _make_reduce(jnp.max)
reduce_min = _make_reduce(jnp.min)
reduce_prod = _make_reduce(jnp.prod)
reduce_any = _make_reduce(jnp.any)
reduce_all = _make_reduce(jnp.all)
reduce_std = _make_reduce(jnp.std)
reduce_variance = _make_reduce(jnp.var)
reduce_logsumexp = _make_reduce(_jsp.special.logsumexp)


# tf.broadcast_static_shape / broadcast_dynamic_shape -> jnp.broadcast_shapes
def broadcast_static_shape(shape1, shape2, name=None):
    return tuple(jnp.broadcast_shapes(tuple(shape1), tuple(shape2)))


def broadcast_dynamic_shape(shape1, shape2, name=None):
    return jnp.broadcast_shapes(jnp.asarray(shape1), jnp.asarray(shape2))


# tf.slice(input, begin, size) -> lax.dynamic_slice
def slice(input, begin, size, name=None):
    import jax.lax as _l
    return _l.dynamic_slice(input, tuple(np.asarray(begin).tolist()),
                            tuple(np.asarray(size).tolist()))
math.reduce_sum = reduce_sum
math.reduce_mean = reduce_mean
math.reduce_max = reduce_max
math.reduce_min = reduce_min
math.reduce_prod = reduce_prod
math.reduce_any = reduce_any
math.reduce_all = reduce_all
math.reduce_std = reduce_std
math.reduce_variance = reduce_variance
math.reduce_logsumexp = reduce_logsumexp


# tf.where: 1-arg form returns indices of True; 3-arg form selects. Tolerate kwargs.
def where(condition, x=None, y=None, name=None):
    if x is None and y is None:
        return jnp.argwhere(condition)
    return jnp.where(condition, x, y)


# tf.control_dependencies: graph-mode control flow; no-op under JAX/eager.
@contextlib.contextmanager
def control_dependencies(control_inputs=None):
    yield



__version__ = "jax-shim"
