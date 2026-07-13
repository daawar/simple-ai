from backprop import Value
import graph
import random


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert isinstance(x, list), "input x should be a python array/list"
        z = list(zip(self.w, x))
        out = sum((wi * xi for wi, xi in z), self.b)
        return out


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, nin, nouts):
        # number of outputs of every layer is the number of input to next layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Neuron test code
n = Neuron(2)
x = [5.0, 2.0]
neout = n(x)
# print(neout)


# layer test code
l = Layer(2, 3)
lout = l(x)
# print(lout)

# MLP test code
a = 2
b = [2, 3, 3, 1]

m = MLP(a, b)
mout = m(x)
# print(mout[0])

h = mout[0].tanh()
print(h)

h.backward()

graph.display(h)
