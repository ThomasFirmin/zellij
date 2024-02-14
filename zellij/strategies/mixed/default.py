# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.metaheuristic import Metaheuristic
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import Searchspace

import numpy as np

import logging

logger = logging.getLogger("zellij.DEFAULT")


class Default(Metaheuristic):

    """Default

    Evaluate a given list of solutions.

    Attributes
    ----------
    search_space : Searchspace
        Search space object containing bounds of the search space.
    solution : list
        List of lists. Each elements represents a single solution.
    batch : int, default=1
        Batch size of returned solution at each :code:`forward`.


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
    >>> from zellij.strategies.mixed import Default

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}
    >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
    >>> sp = ContinuousSearchspace(a)
    >>> points = [[-0.270844, -0.923038], [3, 2]]
    >>> opt = Default(sp, points, batch=2)
    >>> print(opt.solutions)
    [[-0.270844, -0.923038], [3, 2], [-0.270844, -0.923038], [3, 2]]
    >>> stop = Calls(himmelblau, 4)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([3, 2])=0
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 4

    """

    def __init__(
        self,
        search_space: Searchspace,
        solutions: List[list],
        batch: int = 1,
        verbose: bool = True,
    ):
        """__init__

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        solution : list[list]
            List of lists. Each elements represents a single solution.
        batch : int, default=1
            Batch size of returned solutions at each :code:`forward`.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############
        self.batch = batch
        self.solutions = solutions * self.batch

        #############
        # VARIABLES #
        #############
        # iterations
        self.i = 0

    def forward(
        self,
        X: Optional[List[list]] = None,
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

        logger.info("Default Starting")

        # batch = self.i % len(self.batch_sol)
        # solutions = self.batch_sol[f"batch_{batch}"]
        infos = {"algorithm": "Default", "iteration": self.i}
        self.i += 1

        return self.solutions, infos
