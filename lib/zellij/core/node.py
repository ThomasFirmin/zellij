import numpy as np


class Node(object):
    def __init__(self, operation, outputs):
        self.operation = operation  # Pytorch layer
        self.outputs = outputs  # list(Node) ou [None] si output

    def __str__(self):
        out = f"{self.operation} -> {[c.operation for c in self.outputs if c != None]}\n"
        for c in self.outputs:
            if c != None:
                out += c.__str__()

        return out

    def __repr__(self):
        return self.operation

    def is_eq(self, node):
        eq = self.operation == node.operation and len(node.outputs) == len(self.outputs)
        if len(self.outputs) == 0:
            eq = eq and len(node.outputs) == 0
        else:
            for n in self.outputs:
                eq = eq and self.is_in_outputs(n)
        return eq

    def is_in_outputs(self, node):
        for n in self.outputs:
            if n.is_eq(node):
                return True
        return False

    def is_in_list(self, l):
        for n in l:
            if n.is_eq(self):
                return True
        return False


def remove_node_from_list(l, node):
    removed = False
    for n in l:
        if n.is_eq(node):
            l.remove(n)
            removed = True
    if not removed:
        print("Node not found in outputs")
    return l


class DAGraph(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.root = nodes[0]
        self.leaf = self.set_leaf()

    def set_leaf(self):
        i = 0
        while i < len(self.nodes) and len(self.nodes[i].outputs) > 0:
            i += 1
        return self.nodes[i]

    def __str__(self):
        return self.root.__str__()

    def copy(self):
        new_nodes = [Node(self.leaf.operation, [])]
        old_nodes = self.nodes.copy()[::-1]
        old_nodes.remove(self.leaf)
        while len(old_nodes) > 0:
            for n in old_nodes:
                if not any(o in old_nodes for o in n.outputs):
                    new_outputs = []
                    for old_n in n.outputs:
                        i = len(new_nodes) - 1
                        found = False
                        while i >= 0 and not found:
                            if old_n.is_eq(new_nodes[i]):
                                new_outputs.append(new_nodes[i])
                                found = True
                            i -=1
                    new_nodes = [Node(n.operation, new_outputs)] + new_nodes
                    old_nodes.remove(n)
        return DAGraph(new_nodes)


def crossover(g1, g2):
    graph1 = g1.copy()
    graph2 = g2.copy()
    subset_1 = []
    subset_2 = []

    # Select a connected path from root to output
    current_node = graph1.root
    while len(current_node.outputs) != 0:
        idx = np.random.randint(len(current_node.outputs))
        current_node = current_node.outputs[idx]
        subset_1.append(current_node)

    current_node = graph2.root
    while len(current_node.outputs) != 0:
        idx = np.random.randint(len(current_node.outputs))
        current_node = current_node.outputs[idx]
        subset_2.append(current_node)

    index_1_1 = np.random.randint(len(subset_1))
    if index_1_1 + 1 < len(subset_1):
        index_1_2 = np.random.randint(index_1_1 + 1, len(subset_1))
    else:
        index_1_2 = index_1_1

    index_2_1 = np.random.randint(len(subset_2))
    if index_2_1 + 1 < len(subset_2):
        index_2_2 = np.random.randint(index_2_1 + 1, len(subset_2))
    else:
        index_2_2 = index_2_1
    subset_1 = subset_1[index_1_1:index_1_2 + 1]
    subset_2 = subset_2[index_2_1:index_2_2 + 1]

    child1 = [n for n in graph1.nodes if n not in subset_1]
    child2 = [n for n in graph2.nodes if n not in subset_2]

    subset_1[-1].outputs, subset_2[-1].outputs = subset_2[-1].outputs, subset_1[-1].outputs
    orphan_nodes_1 = []
    orphan_nodes_2 = []
    if len(subset_1) > 1:
        for i in range(len(subset_1) - 1, 0, -1):
            orphan_nodes_1 += remove_node_from_list(subset_1[i - 1].outputs, subset_1[i])
            subset_1[i - 1].outputs = [subset_1[i]]
    orphan_nodes_1 = list(dict.fromkeys(orphan_nodes_1))
    if len(subset_2) > 1:
        for i in range(len(subset_2) - 1, 0, -1):
            orphan_nodes_2 += remove_node_from_list(subset_2[i - 1].outputs, subset_2[i])
            subset_2[i - 1].outputs = [subset_2[i]]
    orphan_nodes_2 = list(dict.fromkeys(orphan_nodes_2))

    for i in range(len(child1)):
        if child1[i].is_in_outputs(subset_1[0]):
            child1[i].outputs = remove_node_from_list(child1[i].outputs, subset_1[0]) + [subset_2[0]] + orphan_nodes_1
        if child1[i].is_in_outputs(subset_1[-1]):
            child1[i].outputs = remove_node_from_list(child1[i].outputs, subset_1[-1]) + [subset_2[-1]]
    child1 += subset_2
    for i in range(len(child2)):
        if child2[i].is_in_outputs(subset_2[0]):
            child2[i].outputs = remove_node_from_list(child2[i].outputs, subset_2[0]) + [subset_1[0]] + orphan_nodes_2
        if child2[i].is_in_outputs(subset_2[-1]):
            child2[i].outputs = remove_node_from_list(child2[i].outputs, subset_2[-1]) + [subset_1[-1]]
    child2 += subset_1

    child1 = DAGraph(child1)
    child2 = DAGraph(child2)
    return child1, child2
