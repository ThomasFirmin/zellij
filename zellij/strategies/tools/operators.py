# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.addons import Mutation, Crossover, Selection

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import Searchspace

import numpy as np
from deap import tools


class NeighborMutation(Mutation):
    """NeighborMutation

    Based on `DEAP <https://deap.readthedocs.io/>`_.
    It is a :ref:`spadd` which defines a mutation.
    The mutation itself is based on the neighbor :ref:`addons` defined
    for each :ref:`var` of the :ref:`sp`. See :ref:`nbh`.

    Parameters
    ----------
    probability : float
        Probaility for an individual to be mutated.

    Attributes
    ----------
    probability

    """

    def __init__(self, probability: float):
        super(Mutation, self).__init__()
        self.probability = probability

    @property
    def probability(self) -> float:
        return self._probability

    @probability.setter
    def probability(self, value: float):
        if value < 0 or value > 1:
            raise InitializationError(
                f"Probability must be comprised in ]0,1], got {value}"
            )
        else:
            self._probability = value

    @Mutation.target.setter
    def target(self, value: Searchspace):
        if value and value._do_neighborhood:
            self._target = value
        else:
            raise InitializationError(
                "The Searchspace and variables must implements a neighborhood."
            )

    def _build(self, toolbox):
        toolbox.register("mutate", self)

    def __call__(self, individual):
        # For each dimension of a solution draw a probability to be muted
        for idx, val in enumerate(self.target.variables):
            if np.random.random() < self.probability:
                # Get a neighbor of the selected attribute
                individual[idx] = val.neighborhood(individual[idx])  # type: ignore

        return (individual,)


class DeapTournament(Selection):
    """DeapTournament

    Based on `DEAP <https://deap.readthedocs.io/>`_ tournament.
    :ref:`spadd` defining a selection method.

    Parameters
    ----------
    size : int
        Size of the tournament.
    search_space : :ref:`sp`
        Targeted :ref:`sp`.

    Attributes
    ----------
    size

    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int):
        if value > 0:
            self._size = value
        else:
            raise InitializationError(f"Size must be an int > 0, got {value}")

    def _build(self, toolbox):
        toolbox.register("select", tools.selTournament, tournsize=self.size)
        self.toolbox = toolbox

    def __call__(self, population, k):
        return list(map(self.toolbox.clone, self.toolbox.select(population, k=k)))


class DeapOnePoint(Crossover):
    """DeapOnePoint

    Based on `DEAP <https://deap.readthedocs.io/>`_ cxOnePoint.
    :ref:`spadd` defining a crossover method.

    """

    def __init__(self):
        super(Crossover, self).__init__()

    @Crossover.target.setter
    def target(self, value):
        if value:
            self._target = value
        else:
            raise InitializationError(
                "Int DeapOneTournament, the Searchspace must be a MixedSearchspace, ContinuousSearchspace or DiscreteSearchspace."
            )

    def _build(self, toolbox):
        toolbox.register("mate", tools.cxOnePoint)
        self.toolbox = toolbox

    def __call__(self, children1, children2):
        self.toolbox.mate(children1, children2)
