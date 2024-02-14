# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.metaheuristic import Metaheuristic

from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import Searchspace

import numpy as np

import logging

logger = logging.getLogger("zellij.RD")


class Random(Metaheuristic):

    """Random

    Samples random points from  the search space.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp`.
    size : int, default=1
        Number of points to sample at each :code:`forward`.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.

    Examples
    --------
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.mixed import Random

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
    >>> sp = ContinuousSearchspace(a)
    >>> opt = Random(sp)
    >>> stop = Calls(himmelblau, 4)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([3.0593050205475905, 0.7745988145845581])=11.910270230279236
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 4
    """

    def __init__(
        self,
        search_space: Searchspace,
        size: int = 1,
        verbose: bool = True,
    ):
        """__init__(search_space, size=1, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            :ref:`sp`.
        size : int, default=1
            Number of points to sample at each :code:`forward`.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############
        self.size = size

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int):
        if value > 0:
            self._size = value
        else:
            raise InitializationError(f"In Random, size must be > 0. Got {value}.")

    # Run Random
    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of CGS.

        X, Y, secondary, or constraints are not necessary.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        logger.info("GA Starting")

        solutions = self.search_space.random_point(self.size)

        return solutions, {"algorithm": "Random"}
