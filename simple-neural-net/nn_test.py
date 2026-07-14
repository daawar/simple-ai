from backprop import Value
from graph import display
import math
from nn import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(3, [3, 4, 4, 1])

ypred = [mlp(x) for x in xs]

print(ypred)

# TODO implement power function for Value class, for now using math.pow
# loss = sum(((ypred[0][i] - ys[i]) ** 2) for i in range(len(ys)))


loss = sum(math.pow((ypred[i][0] - ys[i]).data, 2.0) for i in range(len(ys)))

print(loss)