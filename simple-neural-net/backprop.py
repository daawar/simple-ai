import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: (
            None
        )  ##TODO understand this syntax and use of this assignment
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        assert isinstance(other, Value) or isinstance(other, (int, float)), (
            "Unsupported type for add operations- should be either int or float"
        )
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # calculate gradients in backward propagation
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward  ##TODO understand this lamba assignment
        return out

    def __mul__(self, other):
        assert isinstance(other, Value) or isinstance(other, (int, float)), (
            "Unsupported type for multiply operations- should be either int or float"
        )
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # w = u * v
            # L(w)
            # dL/du = dL/dw * dw/du (chain rule of multiplication)
            self.grad += (
                out.grad * other.data
            )  # Chain rule - upstream gradient wrt to Loss fucntion calculated so far and then Local gradient wrt current operation (d)
            other.grad += out.grad * self.data

        out._backward = _backward  ##TODO understand this lamba assignment
        return out

    # def __pow__(self, other):
    #     assert isinstance(other, (int, float)), 'only int or float exponents allowed'
    #     out = Value(self.data ** other, (self, other))

    #     return out

    def tanh(self):
        x = self.data
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # mimics the backpropagation function "backward" from pytorch library
    def backward(self):
        topo = []
        visited = (
            set()
        )  # ALERT-- Using visited = () gave error about not able to add to tuple

        def _toposort(v):
            if v not in visited:
                visited.add(v)
                for u in v._prev:
                    _toposort(u)
                topo.append(v)

        ### ALERT - Missing this step was not showing any gradients for any nodes ####
        _toposort(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# test code below
a = Value(1.0)
b = Value(2.0)
c = a + b  # (a, b) -+-> c
d = Value(4.0)
e = d * c * 0.5  # (d, c) -*-> e
print(e)

e.backward
