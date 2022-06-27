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
        return self.operation.__repr__()

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

    def __repr__(self):
        nodes_reprs = ""
        for n in self.nodes:
            nodes_reprs += n.operation.__repr__()  + f" -> {[c.operation for c in n.outputs]}\n"
        return nodes_reprs


    def copy(self):
        new_nodes = [Node(self.leaf.operation, [])]
        old_nodes = self.nodes.copy()[::-1]
        old_nodes.remove(self.leaf)
        while len(old_nodes) > 0:
            for n in old_nodes:
                if not any(o in old_nodes for o in n.outputs):
                    new_outputs = []
                    for old_n in n.outputs:
                        if old_n is not None:
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
