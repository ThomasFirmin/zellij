# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-06T12:07:46+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-02T11:57:26+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin
import random

from zellij.core.addons import Mutator, Crossover, Selector
from zellij.core.node import AdjMatrix, fill_adj_matrix
from zellij.core.search_space import Searchspace, logger
from zellij.core.variables import Constant
import numpy as np
from deap import tools


class NeighborMutation(Mutator):
    def __init__(self, probability, search_space=None):
        assert (
                probability > 0 and probability <= 1
        ), f'Probability must be comprised in ]0,1], got {probability}'
        self.probability = probability

        super(Mutator, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):

        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
            for {self.__class__.__name__}, got {search_space}"""

            assert all(
                hasattr(val, "neighbor") for val in search_space.values
            ), f"""For {self.__class__.__name__} values of target object must
            have a `neighbor` method. When defining the :ref:`sp`,
            user must define the `neighbor` kwarg before the `mutation` kwarg.
            ex:\n
            >>> ContinuousSearchspace(values, loss, neighbor=..., mutation=...)
            """

        self._target = search_space

    def _build(self, toolbox):
        toolbox.register("mutate", self)

    def __call__(self, individual):
        # For each dimension of a solution draw a probability to be muted
        for val in self.target.values:
            if np.random.random() < self.probability and not isinstance(
                    val, Constant
            ):
                # Get a neighbor of the selected attribute
                individual[val._idx] = val.neighbor(individual[val._idx])

        return (individual,)


class DeapTournament(Selector):
    def __init__(self, size, search_space=None):
        assert size > 0, f"Size must be an int > 0, got {size}"
        self.size = size

        super(Selector, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):
        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
            for {self.__class__.__name__}, got {search_space}"""

            assert all(
                hasattr(val, "neighbor") for val in search_space.values
            ), f"""For {self.__class__.__name__} values of target object must
            have a `neighbor` method. When defining the :ref:`sp`,
            user must define the `neighbor` kwarg before the `mutation` kwarg.
            ex:\n
            >>> ContinuousSearchspace(values, loss, neighbor=..., mutation=...)
            """

        self._target = search_space

    def _build(self, toolbox):
        toolbox.register("select", tools.selTournament, tournsize=self.size)
        self.toolbox = toolbox

    def __call__(self, population, k):
        return list(
            map(self.toolbox.clone, self.toolbox.select(population, k=k))
        )


class DeapOnePoint(Crossover):
    def __init__(self, search_space=None):
        super(Crossover, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):
        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
            for {self.__class__.__name__}, got {search_space}"""

            assert all(
                hasattr(val, "neighbor") for val in search_space.values
            ), f"""For {self.__class__.__name__} values of target object must
            have a `neighbor` method. When defining the :ref:`sp`,
            user must define the `neighbor` kwarg before the `mutation` kwarg.
            ex:\n
            >>> ContinuousSearchspace(values, loss, neighbor=..., mutation=...)
            """

        self._target = search_space

    def _build(self, toolbox):
        toolbox.register("mate", tools.cxOnePoint)
        self.toolbox = toolbox

    def __call__(self, children1, children2):
        self.toolbox.mate(children1, children2)


class DAGTwoPoint(Crossover):

    def __init__(self, search_space=None, size=10):
        self.size = size
        super(DAGTwoPoint, self).__init__(search_space)

    def _build(self, toolbox):
        toolbox.register("mate", self)

    def __call__(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        for i in range(cxpoint1, cxpoint2):
            if isinstance(ind1[i], AdjMatrix):
                ind1[i], ind2[i] = self.adj_matrix_crossover(ind1[i], ind2[i])

        return ind1, ind2

    def adj_matrix_crossover(self, p1, p2):
        crossed = False
        while not crossed:
            op1 = p1.operations
            op2 = p2.operations
            m1 = p1.matrix
            m2 = p2.matrix

            s1 = list(set(np.random.choice(range(1, len(op1)), size=len(op1) - 1)))
            s2 = list(set(np.random.choice(range(1, len(op2)), size=len(op2) - 1)))
            s1.sort()
            s2.sort()

            # remove subsets
            it = 0
            for i1 in s1:
                m1 = np.delete(m1, i1 - it, axis=0)
                m1 = np.delete(m1, i1 - it, axis=1)
                it+=1
            it = 0
            for i2 in s2:
                m2 = np.delete(m2, i2 - it, axis=0)
                m2 = np.delete(m2, i2 - it, axis=1)
                it+=1

            # Select index new nodes
            old_s1 = np.array(list(set(range(len(op1))) - set(s1)))
            old_s2 = np.array(list(set(range(len(op2))) - set(s2)))
            new_s1 = [np.argmin(np.abs(old_s2 - s1[0]))]
            if new_s1[0] == old_s2[new_s1[0]]:
                new_s1[0] += 1
            for i1 in range(1, len(s1)):
                new_s1.append(min(s1[i1] - s1[i1-1] + new_s1[i1-1], len(old_s2) + len(new_s1)))
            new_s2 = [np.argmin(np.abs(old_s1 - s2[0]))]
            if new_s2[0] == old_s1[new_s2[0]]:
                new_s2[0] += 1
            for i2 in range(1, len(s2)):
                new_s2.append(min(s2[i2] - s2[i2 - 1] + new_s2[i2-1], len(old_s1) + len(new_s2)))
            m1_shape_before = m1.shape
            m2_shape_before = m2.shape
            m1 = np.insert(m1, np.clip(new_s2, 0, m1.shape[0]), 0, axis=0)
            m1 = np.insert(m1, np.clip(new_s2, 0, m1.shape[1]), 0, axis=1)
            m2 = np.insert(m2, np.clip(new_s1, 0, m2.shape[0]), 0, axis=0)
            m2 = np.insert(m2, np.clip(new_s1, 0, m2.shape[1]), 0, axis=1)
            m1_shape_after = m1.shape
            m2_shape_after = m2.shape
            for i in range(len(s1)):
                diff = new_s1[i] - s1[i]
                if diff >= 0:
                    length = min(m2.shape[0] - diff, p1.matrix.shape[0])
                    try:
                        m2[diff:diff+length, new_s1[i]] = p1.matrix[:length, s1[i]]
                        m2[new_s1[i], diff:diff+length] = p1.matrix[s1[i], :length]
                    except IndexError as e:
                        logger.error(f"Diff = {diff}, failed with {e}, with m2 = {m2}, p1.matrix = {p1.matrix}, i = {i}, "
                                     f"new_s1[i] = {new_s1[i]}, s1[i] = {s1[i]}, m2 shape before: {m2_shape_before}, "
                                     f"m2 shape after: {m2_shape_after}, new_s1= {new_s1}, s1 = {s1}, old_s1 = {old_s1}")
                if diff < 0:
                    length = min(m2.shape[0], p1.matrix.shape[0]+diff)
                    try:
                        m2[:length, new_s1[i]] = p1.matrix[-diff:-diff+length, s1[i]]
                        m2[new_s1[i], :length] = p1.matrix[s1[i], -diff:-diff+length]
                    except IndexError as e:
                        logger.error(f"Diff = {diff}, failed with {e}, with m2 = {m2}, p1.matrix = {p1.matrix}, i = {i}, "
                                     f"new_s1[i] = {new_s1[i]}, s1[i] = {s1[i]}, m2 shape before: {m2_shape_before}, "
                                     f"m2 shape after: {m2_shape_after}, new_s1= {new_s1}, s1 = {s1}")
            for i in range(len(s2)):
                diff = new_s2[i] - s2[i]
                if diff >= 0:
                    length = min(m1.shape[0] - diff, p2.matrix.shape[0])
                    try:
                        m1[diff:diff+length, new_s2[i]] = p2.matrix[:length, s2[i]]
                        m1[new_s2[i], diff:diff+length] = p2.matrix[s2[i], :length]
                    except IndexError as e:
                        logger.error(f"Diff = {diff}, failed with {e}, with m1 = {m1}, p2.matrix = {p2.matrix}, i = {i}, "
                                     f"new_s2[i] = {new_s2[i]}, s2[i] = {s2[i]}, m1 shape before: {m1_shape_before}, "
                                     f"m1 shape after: {m1_shape_after}, new_s2= {new_s2}, s2 = {s2}")
                if diff < 0:
                    length = min(m1.shape[0], p2.matrix.shape[0]+diff)
                    try:
                        m1[:length, new_s2[i]] = p2.matrix[-diff:-diff + length, s2[i]]
                        m1[new_s2[i], :length] = p2.matrix[s2[i], -diff:-diff + length]
                    except IndexError as e:
                        logger.error(f"Diff = {diff}, failed with {e}, with m1 = {m1}, p2.matrix = {p2.matrix}, i = {i}, "
                                     f"new_s2[i] = {new_s2[i]}, s2[i] = {s2[i]}, m1 shape before: {m1_shape_before}, "
                                     f"m1 shape after: {m1_shape_after}, new_s2= {new_s2}, s2 = {s2}")
            m1 = np.triu(m1, k=1)
            m1 = fill_adj_matrix(m1)
            m2 = np.triu(m2, k=1)
            m2 = fill_adj_matrix(m2)
            op1 = [op1[i] for i in range(len(op1)) if i not in s1]
            op2 = [op2[i] for i in range(len(op2)) if i not in s2]
            for i in range(len(new_s1)):
                op2 = op2[:new_s1[i]] + [p1.operations[s1[i]]] + op2[new_s1[i]:]
            for i in range(len(new_s2)):
                op1 = op1[:new_s2[i]] + [p2.operations[s2[i]]] + op1[new_s2[i]:]
            if max(len(op1), len(op2)) <= self.size:
                crossed = True
        return AdjMatrix(op1, m1), AdjMatrix(op2, m2)

    @Mutator.target.setter
    def target(self, search_space):
        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
                for {self.__class__.__name__}, got {search_space}"""
        self._target = search_space