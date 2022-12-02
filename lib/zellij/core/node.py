import numpy as np

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

    def __eq__(self, other):
        if len(self.operations) == len(other.operations):
            return (other.matrix == self.matrix) and (sum([other.operations[i] != self.operations[i] for i in range(len(self.operations))]) == 0)
        else:
            return False
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
