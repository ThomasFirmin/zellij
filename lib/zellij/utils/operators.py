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
from zellij.core.node import remove_node_from_list, DAGraph
from zellij.core.search_space import Searchspace
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
                print(individual)
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

    def __init__(self, search_space=None):
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

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        for i in range(cxpoint1, cxpoint2):
            if isinstance(ind1[i], DAGraph):
                ind1[i], ind2[i] = self.dag_crossover(ind1[i], ind2[i])
        return ind1, ind2

    def dag_crossover(self, g1, g2):
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
                child1[i].outputs = remove_node_from_list(child1[i].outputs, subset_1[0]) + [
                    subset_2[0]] + orphan_nodes_1
            if child1[i].is_in_outputs(subset_1[-1]):
                child1[i].outputs = remove_node_from_list(child1[i].outputs, subset_1[-1]) + [subset_2[-1]]
        child1 += subset_2
        for i in range(len(child2)):
            if child2[i].is_in_outputs(subset_2[0]):
                child2[i].outputs = remove_node_from_list(child2[i].outputs, subset_2[0]) + [
                    subset_1[0]] + orphan_nodes_2
            if child2[i].is_in_outputs(subset_2[-1]):
                child2[i].outputs = remove_node_from_list(child2[i].outputs, subset_2[-1]) + [subset_1[-1]]
        child2 += subset_1

        child1 = DAGraph(child1)
        child2 = DAGraph(child2)
        return child1, child2

    @Mutator.target.setter
    def target(self, search_space):
        if search_space:
            assert isinstance(
                search_space, Searchspace
            ), f""" Target object must be a :ref:`sp`
                for {self.__class__.__name__}, got {search_space}"""
        self._target = search_space
