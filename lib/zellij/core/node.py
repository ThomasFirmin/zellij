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
    for n1 in nodes:
        for n2 in new_l:
            if n1 == n2:
                new_l.remove(n2)
                break
    return new_l


def order_nodes(nodes, leaf):
    if len(nodes) > 0:
        flagged = [False for _ in range(len(nodes))]
        nodes.append(nodes.pop(nodes.index(leaf)))
        flagged[-1] = True
        changes = 0
        while sum(flagged) < len(flagged) and changes < len(flagged) ** 2:
            last_idx = [i for i, x in enumerate(flagged) if not x][-1]
            node = nodes[last_idx]
            outputs_indexes = []
            for out in node.outputs:
                outputs_indexes.append(nodes.index(out))
            min_out = min(outputs_indexes)
            if last_idx < min_out:
                flagged[last_idx] = True
            else:
                nodes.remove(node)
                nodes.insert(min(outputs_indexes), node)
                flagged.remove(flagged[last_idx])
                flagged.insert(min(outputs_indexes), False)
            changes += 1
        if sum(flagged) < len(flagged):
            raise RecursionError
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
        self.check_orphans()

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

    def check_orphans(self):
        for i in range(1, len(self.nodes)):
            related = self.nodes[i] in self.root.outputs
            j = 1
            while not related and j < i:
                related = self.nodes[i] in self.nodes[j].outputs
                j += 1
            if not related:
                raise OrphanError(self.nodes[i], self.nodes)


class OrphanError(Exception):
    def __init__(self, node, nodes):
        self.message = "Orphan error: Graph has orphans: node {} in {}".format(node.operation, [n for n in nodes])

    def __str__(self):
        return self.message


class AdjMatrix(object):
    def __init__(self, operations, matrix):
        self.matrix = matrix
        self.operations = operations
        self.assert_adj_matrix()

    def assert_adj_matrix(self):
        assert isinstance(self.operations, list), f"""Operations should be a list, got {self.operations} instead."""
        assert isinstance(self.matrix, np.ndarray) and (self.matrix.shape[0] == self.matrix.shape[1]), f"""Matrix should be a 
        squared array. got {self.matrix} instead."""
        assert self.matrix.shape[0] == len(
            self.operations), f"""Matrix and operations should have the same dimension got {self.matrix.shape[0]} 
                and {len(self.operations)} instead. """
        assert np.sum(np.triu(self.matrix, k=1) != self.matrix) == 0, f"""The adjacency matrix should be upper-triangular with 0s on the
        diagonal. Got {self.matrix}. """
        for i in range(self.matrix.shape[0] - 1):
            assert sum(self.matrix[i]) > 0, f"""Node {i} does not have any child."""
        for j in range(1, self.matrix.shape[1]):
            assert sum(self.matrix[:, j]) > 0, f"""Node {j} does not have any parent."""
        if self.operations[0][0] != "Input":
            logger.error(self.operations)

    def copy(self):
        new_op = self.operations.copy()
        new_matrix = self.matrix.copy()
        return AdjMatrix(new_op, new_matrix)

    def __str__(self):
        matrix_str = f"NODES: {self.operations.__str__()} | MATRIX:{self.matrix.tolist().__str__()}"
        return matrix_str

    def __repr__(self):
        matrix_repr = f"NODES: {self.operations.__repr__()} | MATRIX:{self.matrix.tolist().__repr__()}"
        return matrix_repr



def fill_adj_matrix(matrix):
    for i in range(matrix.shape[0] - 1):
        new_row = matrix[i, i + 1:]
        while sum(new_row) == 0:
            new_row = np.random.choice(2, new_row.shape[0])
        matrix[i, i + 1:] = new_row
    for j in range(1, matrix.shape[1]):
        new_col = matrix[:j, j]
        while sum(new_col) == 0:
            new_col = np.random.choice(2, new_col.shape[0])
        matrix[:j, j] = new_col
    return matrix
