import numpy as np

class Node(object):
    def __init__(self, operation, outputs):
        self.operation = operation  # Pytorch layer
        self.outputs = list(set(outputs))  # list(Node) ou [None] si output

    def __str__(self):
        from zellij.core.variables import logger
        out = f"{self.operation} -> "
        try:
            out += f"{[c.operation for c in self.outputs if c != None]}\n"
        except RecursionError as e:
            print(e)
            print(f"Node: {self.operation}, with outputs: {self.outputs}")
        for c in self.outputs:
            if c is not None:
                out += c.__str__()
        return out

    def __repr__(self):
        return self.operation.__repr__()

    def is_eq(self, node):
        eq = self.operation == node.operation and len(node.outputs) == len(self.outputs)
        if len(self.outputs) == 0:
            pass
        else:
            for n in node.outputs:
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
        self.leaf = self.set_leaf(nodes)
        nodes.append(nodes.pop(nodes.index(self.leaf)))
        self.nodes = nodes
        self.root = nodes[0]

    def set_leaf(self, nodes):
        i = 0
        while i < len(nodes) and len(nodes[i].outputs) > 0:
            i += 1
        return nodes[i]

    def __str__(self):
        return self.root.__str__()

    def __repr__(self):
        nodes_reprs = ""
        for n in self.nodes:
            nodes_reprs += n.operation.__repr__()  + f" -> {[c.operation for c in n.outputs if c != None]}\n"
        return nodes_reprs


    def copy(self):
        new_nodes = [Node(self.leaf.operation, [])]
        old_nodes = self.nodes.copy()[::-1]
        old_nodes.remove(self.leaf)
        while len(old_nodes) > 0:
            for k in range(len(old_nodes)):
                n = old_nodes[k]
                if not any(o in old_nodes for o in n.outputs):
                    new_outputs = []
                    new_not_found = [True for _ in range(len(new_nodes))]
                    for old_n in n.outputs:
                        i = 0
                        found = False
                        while i < len(new_nodes) and not found:
                            if old_n.is_eq(new_nodes[i]) and new_not_found[i]:
                                new_outputs.append(new_nodes[i])
                                found = True
                                new_not_found[i] = False
                            i +=1
                    new_nodes = [Node(n.operation, new_outputs)] + new_nodes
                    old_nodes[k] = "Flagged"
            old_nodes = list(filter(("Flagged").__ne__, old_nodes))
        for n in new_nodes:
            n.outputs = list(set(n.outputs))
        return DAGraph(new_nodes)
