# tensor.py
class ScalarGrad:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    # -------------------
    # Basic ops
    # -------------------
    def __add__(self, other):
        other = other if isinstance(other, ScalarGrad) else ScalarGrad(other)
        return ScalarGrad(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, ScalarGrad) else ScalarGrad(other)
        return ScalarGrad(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return ScalarGrad(self.data**other, (self,), (other * self.data**(other - 1),))

    def relu(self):
        return ScalarGrad(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return ScalarGrad(other) + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * ScalarGrad(other)**-1
    def __rtruediv__(self, other): return ScalarGrad(other) * self**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build_topo(c)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for c, lg in zip(v._children, v._local_grads):
                c.grad += lg * v.grad

    def __repr__(self):
        return f"ScalarGrad(data={self.data:.4f}, grad={self.grad:.4f})"


class TensorGrad:
    def __init__(self, data):
        if isinstance(data, ScalarGrad):
            self.data = data
        elif isinstance(data, (int, float)):
            self.data = ScalarGrad(data)
        elif isinstance(data, TensorGrad):
            self.data = data.data
        elif isinstance(data, list):
            self.data = [x if isinstance(x, TensorGrad) else TensorGrad(x) for x in data]
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def _is_scalar(self): return isinstance(self.data, ScalarGrad)
    def _scalar(self): return self.data

    # -------------------
    # Elementwise ops
    # -------------------
    def __add__(self, other):
        other = other if isinstance(other, TensorGrad) else TensorGrad(other)
        if self._is_scalar() and other._is_scalar():
            return TensorGrad(self._scalar() + other._scalar())
        if self._is_scalar() or other._is_scalar():
            raise ValueError("Cannot add scalar TensorGrad and non-scalar TensorGrad.")
        if len(self.data) != len(other.data):
            raise ValueError(f"Add shape mismatch: {len(self.data)} vs {len(other.data)}")
        return TensorGrad([a + b for a, b in zip(self.data, other.data)])

    def __mul__(self, other):
        other = other if isinstance(other, TensorGrad) else TensorGrad(other)
        if self._is_scalar() and other._is_scalar():
            return TensorGrad(self._scalar() * other._scalar())
        if self._is_scalar() or other._is_scalar():
            raise ValueError("Cannot multiply scalar TensorGrad and non-scalar TensorGrad.")
        if len(self.data) != len(other.data):
            raise ValueError(f"Multiply shape mismatch: {len(self.data)} vs {len(other.data)}")
        return TensorGrad([a * b for a, b in zip(self.data, other.data)])

    def __neg__(self):
        if self._is_scalar(): return TensorGrad(self._scalar() * -1)
        return TensorGrad([-x for x in self.data])

    def __sub__(self, other): return self + (-other)
    def relu(self):
        if self._is_scalar(): return TensorGrad(self._scalar().relu())
        return TensorGrad([x.relu() for x in self.data])

    def dot(self, x):
        x = x if isinstance(x, TensorGrad) else TensorGrad(x)
        if self._is_scalar() and x._is_scalar():
            return TensorGrad(self._scalar() * x._scalar())
        if self._is_scalar() or x._is_scalar():
            raise ValueError("Dot product requires both operands to be scalars or both non-scalars.")
        if len(self.data) == 0 or len(x.data) == 0:
            raise ValueError("Dot product is undefined for empty tensors.")
        if self.data[0]._is_scalar():
            if len(self.data) != len(x.data):
                raise ValueError(f"Dot shape mismatch: {len(self.data)} vs {len(x.data)}")
            s = ScalarGrad(0.0)
            for a, b in zip(self.data, x.data):
                s = s + a._scalar() * b._scalar()
            return TensorGrad(s)
        return TensorGrad([row.dot(x) for row in self.data])

    def sum(self):
        if self._is_scalar(): return self._scalar()
        s = ScalarGrad(0.0)
        for x in self.data:
            s = s + (x._scalar() if x._is_scalar() else x.sum())
        return s

    def scalars(self):
        if self._is_scalar(): return [self._scalar()]
        out = []
        for x in self.data: out.extend(x.scalars())
        return out

    def __len__(self): return 1 if self._is_scalar() else len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    def __repr__(self):
      if self._is_scalar():
          return f"{self.data.data:.4f}"
      return f"[{', '.join(repr(x) for x in self.data)}]"

# tensor.py — add at bottom
def scalar(x): return ScalarGrad(float(x))
def tensor(data):
    if isinstance(data, (int, float)): return TensorGrad(ScalarGrad(data))
    return TensorGrad([tensor(x) for x in data])
