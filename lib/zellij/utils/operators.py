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
from zellij.core.node import remove_node_from_list, DAGraph, select_random_subdag
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
            if isinstance(ind1[i], DAGraph):
                ind1[i], ind2[i] = self.dag_crossover(ind1[i], ind2[i])
        return ind1, ind2

    def dag_crossover(self, g1, g2):
        crossed = False # prevent from too large graphs
        child1 = None
        child2 = None
        while not crossed:
            graph1 = g1.copy()
            graph2 = g2.copy()

            sub_dag_1 = select_random_subdag(graph1)
            sub_dag_2 = select_random_subdag(graph2)

            child1 = [n for n in graph1.nodes if n not in sub_dag_1]
            child2 = [n for n in graph2.nodes if n not in sub_dag_2]

            # Case of 2 empty dags: g1 and g2 are root -> leaf -> []
            if len(sub_dag_1) == 0 and len(sub_dag_2) == 0:
                crossed = True

            # g2 is root -> leaf -> []
            elif len(sub_dag_1) > 0 and len(sub_dag_2) == 0:
                # Orphans 1: all nodes which were in outputs of sub dag nodes
                orphan_nodes_1 = []
                for n in sub_dag_1:
                    orphans = remove_node_from_list(n.outputs, sub_dag_1)
                    orphan_nodes_1 += orphans
                    n.outputs = list(set(n.outputs) - set(orphans))
                # Remove duplicate orphans
                orphan_nodes_1 = list(set(orphan_nodes_1))
                # Add root subdag 1 to root dag 2 output
                child2[0].outputs.append(sub_dag_1[0])
                # Add leaf dag 2 to leaf subdag 1 output
                sub_dag_1[-1].outputs.append(child2[-1])
                # Add nodes from subdag 1 to dag 2
                child2 += sub_dag_1
                # For all node in dag 1, if node from subdag 1 in outputs replace them by orphans
                for n in child1:
                    outputs = remove_node_from_list(n.outputs, sub_dag_1)
                    if len(list(set(n.outputs) - set(outputs))) > 0:
                        n.outputs = outputs
                        if  n not in orphan_nodes_1 : # condition to prevent cycles
                            n.outputs = list(set(outputs).union(set(orphan_nodes_1)))
                if max(len(child1), len(child2)) <= self.size:
                    crossed = True

            elif len(sub_dag_1) == 0 and len(sub_dag_2) > 0:
                orphan_nodes_2 = []
                for n in sub_dag_2:
                    orphans = remove_node_from_list(n.outputs, sub_dag_2)
                    orphan_nodes_2 += orphans
                    n.outputs = list(set(n.outputs) - set(orphans))
                # Remove duplicate orphans
                orphan_nodes_2 = list(set(orphan_nodes_2))
                # Add root subdag 2 to root dag 1 output
                child1[0].outputs.append(sub_dag_2[0])
                # Add leaf dag 1 to leaf subdag 2 output
                sub_dag_2[-1].outputs.append(child1[-1])
                # Add nodes from subdag 2 to dag 1
                child1 += sub_dag_2
                # For all node in dag 2, if node from subdag 2 in outputs replace them by orphans
                for n in child2:
                    outputs = remove_node_from_list(n.outputs, sub_dag_2)
                    if len(list(set(n.outputs) - set(outputs))) > 0:
                        n.outputs = outputs
                        if n not in orphan_nodes_2:  # condition to prevent cycles
                            n.outputs = list(set(outputs).union(set(orphan_nodes_2)))
                if max(len(child1), len(child2)) <= self.size:
                    crossed = True

            else:
                orphan_nodes_1 = []
                for n in sub_dag_1:
                    orphans = remove_node_from_list(n.outputs, sub_dag_1)
                    orphan_nodes_1 += orphans
                    n.outputs = list(set(n.outputs) - set(orphans))
                orphan_nodes_1 = list(set(orphan_nodes_1))
                orphan_nodes_2 = []
                for n in sub_dag_2:
                    orphans = remove_node_from_list(n.outputs, sub_dag_2)
                    orphan_nodes_2 += orphans
                    n.outputs = list(set(n.outputs) - set(orphans))
                orphan_nodes_2 = list(set(orphan_nodes_2))
                # For all node in dag 1, if node from subdag 1 in outputs replace them by subdag 2 root
                for n in child1:
                    outputs = remove_node_from_list(n.outputs, sub_dag_1)
                    if len(list(set(n.outputs) - set(outputs))) > 0:
                        n.outputs = outputs
                        if  n not in orphan_nodes_1 : # condition to prevent cycles
                            n.outputs.append(sub_dag_2[0])
                # For all node in dag 1, if node from subdag 1 in outputs replace them by subdag 2 root
                for n in child2:
                    outputs = remove_node_from_list(n.outputs, sub_dag_2)
                    if len(list(set(n.outputs) - set(outputs))) > 0:
                        n.outputs = outputs
                        if  n not in orphan_nodes_2 : # condition to prevent cycles
                            n.outputs.append(sub_dag_1[0])
                # Add dag 2 orphans to subdag 1 leaf
                sub_dag_1[-1].outputs += orphan_nodes_2
                # Add dag 1 orphans to subdag 2 leaf
                sub_dag_2[-1].outputs += orphan_nodes_1
                # Add nodes from subdag 2 to dag 1
                child1 += sub_dag_2
                # Add nodes from subdag 1 to dag 2
                child2 += sub_dag_1
                if max(len(child1), len(child2)) <= self.size:
                    crossed = True

        for n in child1:
            n.outputs = list(set(n.outputs))
        for n in child2:
            n.outputs = list(set(n.outputs))
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
