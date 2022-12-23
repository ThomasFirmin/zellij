# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-06T12:07:46+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:36:06+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from zellij.core.addons import Mutator, Crossover, Selector
from zellij.core.variables import Constant
import numpy as np
from deap import tools


class NeighborMutation(Mutator):
    """NeighborMutation

    Based on `DEAP <https://deap.readthedocs.io/>`_.
    It is a :ref:`spadd` which defines a mutation.
    The mutation itself is based on the neighbor :ref:`addons` defined
    for each :ref:`var` of the :ref:`sp`. See :ref:`nbh`.

    Parameters
    ----------
    probability : float
        Probaility for an individual to be mutated.
    search_space : :ref:`sp`
        Targeted :ref:`sp`.

    Attributes
    ----------
    probability

    """

    def __init__(self, probability, search_space=None):
        assert (
            probability > 0 and probability <= 1
        ), f"Probability must be comprised in ]0,1], got {probability}"
        self.probability = probability

        super(Mutator, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):

        if search_space:
            # assert isinstance(
            #     search_space, Searchspace
            # ), f""" Target object must be a :ref:`sp`
            # for {self.__class__.__name__}, got {search_space}"""

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

    def __init__(self, size, search_space=None):
        assert size > 0, f"Size must be an int > 0, got {size}"
        self.size = size

        super(Selector, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):

        if search_space:
            # assert isinstance(
            #     search_space, Searchspace
            # ), f""" Target object must be a :ref:`sp`
            # for {self.__class__.__name__}, got {search_space}"""

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
    """DeapOnePoint

    Based on `DEAP <https://deap.readthedocs.io/>`_ cxOnePoint.
    :ref:`spadd` defining a crossover method.

    Parameters
    ----------
    search_space : :ref:`sp`
        Targeted :ref:`sp`.

    """

    def __init__(self, search_space=None):

        super(Crossover, self).__init__(search_space)

    @Mutator.target.setter
    def target(self, search_space):

        if search_space:
            # assert isinstance(
            #     search_space, Searchspace
            # ), f""" Target object must be a :ref:`sp`
            # for {self.__class__.__name__}, got {search_space}"""

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
