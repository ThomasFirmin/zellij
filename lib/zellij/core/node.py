import numpy as np


class Node(object):
    def __init__(self, operation, outputs):
        self.operation = operation  # Pytorch layer
        self.outputs = list(set(outputs))  # list(Node) ou [None] si output

    def __str__(self):
        from zellij.core.variables import logger
        out = f"{self.operation} -> "
        out += f"{[c.operation for c in self.outputs if c != None]}\n"
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


def remove_node_from_list(l, nodes):
    new_l = l[:]
    flagged = [False for _ in range(len(nodes))]
    for n in l:
        for idx in range(len(nodes)):
            if n.is_eq(nodes[idx]) and not flagged[idx]:
                new_l.remove(nodes[idx])
                flagged[idx] = True
    return new_l


def order_nodes(nodes, leaf):
    if len(nodes) > 0:
        flagged = [False for _ in range(len(nodes))]
        nodes.append(nodes.pop(nodes.index(leaf)))
        flagged[-1] = True
        while sum(flagged) < len(flagged):
            last_idx = [i for i, x in enumerate(flagged) if not x][-1]
            node = nodes[last_idx]
            outputs_indexes = []
            for out in node.outputs:
                outputs_indexes.append(nodes.index(out))
            if last_idx < min(outputs_indexes):
                flagged[last_idx] = True
            else:
                nodes.remove(node)
                nodes.insert(min(outputs_indexes), node)
                flagged.remove(flagged[last_idx])
                flagged.insert(min(outputs_indexes), False)
    return nodes

def select_random_subdag(g):
    subdag = []
    if len(g.nodes) > 2:
        while len(subdag) == 0:
            subdag = []
            current_node = g.root
            # Select a random path from root to leaf
            while len(current_node.outputs) != 0:
                idx = np.random.randint(len(current_node.outputs))
                current_node = current_node.outputs[idx]
                subdag.append(current_node)

            # If leaf is in path remove it
            if len(subdag[-1].outputs) == 0:
                subdag.pop()

            # If path is not empty
            if len(subdag) > 0:
                index_1_1 = np.random.randint(len(subdag))
                if index_1_1 + 1 < len(subdag):
                    index_1_2 = np.random.randint(index_1_1, len(subdag))
                else:
                    index_1_2 = index_1_1
                subdag = subdag[index_1_1:index_1_2 + 1]
    return subdag


class DAGraph(object):
    def __init__(self, nodes):
        self.leaf = self.set_leaf(nodes)
        self.nodes = order_nodes(nodes, self.leaf)
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
            nodes_reprs += n.operation.__repr__() + f" -> {[c.operation for c in n.outputs if c != None]}\n"
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
                            i += 1
                    new_nodes = [Node(n.operation, new_outputs)] + new_nodes
                    old_nodes[k] = "Flagged"
            old_nodes = list(filter(("Flagged").__ne__, old_nodes))
        for n in new_nodes:
            n.outputs = list(set(n.outputs))
        dag = DAGraph(new_nodes)
        return dag
