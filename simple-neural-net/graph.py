from backprop import Value
from graphviz import Digraph


def build_nodes_graph(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for u in v._prev:
                edges.add((u, v))
                build(u)

    build(root)
    return nodes, edges


def build_display_graph(root):
    nodes, edges = build_nodes_graph(root)
    g = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    for node in nodes:
        g.node(
            name=str(id(node)),
            label="{data %.4f | grad %.4f}" % (node.data, node.grad),
            shape="record",
        )

        if node._op:
            g.node(name=str(id(node)) + node._op, label=node._op)
            g.edge(str(id(node)) + node._op, str(id(node)))

    for n1, n2 in edges:
        g.edge(str(id(n1)), str(id(n2)) + n2._op)

    return g


# test nodes builder

a = Value(1.0)
b = Value(2.0)
c = a + b  # (a, b) -+-> c
d = Value(4.0)
e = d * c * 0.5  # (d, c) -*-> e
print(e)

e.backward()

# g = build_display_graph(e)
# g.render("gout")

# neuron mimic - using activation function tanh
w1 = Value(0.5)
x1 = Value(2.0)
w2 = Value(0.3)
x2 = Value(5.0)
b = Value(0.634846)

f = w1 * x1 + w2 * x2 + b
print(f)

# use activation function
h = f.tanh()
print(h)

# backpropagate updating gradients on the way back
h.backward()

g = build_display_graph(h)
g.render("gout")
