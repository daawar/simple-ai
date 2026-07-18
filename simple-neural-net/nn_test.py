from backprop import Value
from graph import display
import math
from nn import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(3, [3, 4, 4, 1])

# ypred = [mlp(x) for x in xs]

# print(ypred)

# # TODO implement power function for Value class, for now using math.pow
# # loss = sum(((ypred[0][i] - ys[i]) ** 2) for i in range(len(ys)))

# loss = sum((yactual - ypredicted) ** 2 for ypredicted, yactual in zip(ypred, ys))


# loss = sum(math.pow((ypred[i][0] - ys[i][0]).data, 2.0) for i in range(len(ys)))
# print(loss)

# display(loss)

# training (adjust weights and biases to minimize loss)
# print(mlp.layers[0].neurons[0].w[0].grad)


# backpropagate and calculate all gradients
# loss.backward(?)
#
# display(loss)?

# #training (adjust weights and biases to minimize loss)
# print(mlp.layers[0].neurons[0].w[0].grad)


params = mlp.parameters()

print(len(params))

print(params)

# hyperparameters
step_size = 0.001
total_iters = 35
ypred = []

for iter in range(total_iters):
    # predict
    ypred = [mlp(x) for x in xs]
    # print(ypred)

    # calculate loss
    loss = sum((yactual - ypredicted) ** 2 for ypredicted, yactual in zip(ypred, ys))
    print(iter, loss.data)

    loss.backward()

    # print(f"{iter} Before update", params)
    for p in params:
        p.data -= p.grad * step_size

    # print(f"{iter} After update", params)


print(ypred)
