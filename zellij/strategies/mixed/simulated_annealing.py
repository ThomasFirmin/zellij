# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InputError
from zellij.core.metaheuristic import Metaheuristic

from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import Searchspace
    from zellij.strategies.tools.cooling import Cooling

import numpy as np
import os

import logging

logger = logging.getLogger("zellij.SA")


class SimulatedAnnealing(Metaheuristic):

    """SimulatedAnnealing

    SimulatedAnnealing (SA) is a hill climbing exploitation algorithm.
    Uses a :ref:`cooling` defining the probability of acceptance.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing bounds of the search space.
    cooling : Cooling
        :ref:`cooling` used to determine the probability of acceptance.
    max_iter : int
        Maximum iterations of the inner loop.
        Determines how long the algorithm should sample neighbors of a solution,\
        before decreasing the temperature.
    neighbors : int
        Number of neighbors to draw at each iteration.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij

    
    Examples
    --------
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.mixed import SimulatedAnnealing
    >>> from zellij.strategies.tools import AddLinear
    >>> from zellij.utils import ArrayDefaultN, FloatInterval

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    >>>     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    >>>     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, neighborhood=FloatInterval(0.5)),
    ...     FloatVar("i2", -5, 5, neighborhood=FloatInterval(0.5)),
    ...     neighborhood=ArrayDefaultN(),
    ... )
    >>> sp = ContinuousSearchspace(a)
    >>> first_point = sp.random_point(1)
    >>> cooling = AddLinear(40, 10, 1)
    >>> opt = SimulatedAnnealing(sp, cooling, 2, 5)
    >>> stop = Calls(himmelblau, 400)
    >>> exp = Experiment(opt, himmelblau, stop, )
    >>> exp.run(X=first_point)
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([2.996369117156215, 2.002698134812858])=0.00041521032082360716
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 400
    """

    # Initialize simulated annealing
    def __init__(
        self,
        search_space: Searchspace,
        cooling: Cooling,
        max_iter: int,
        neighbors: int,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        cooling : Cooling
            Cooling schedule used to determine the probability of acceptance.
        max_iter : int
            Maximum iterations of the inner loop.
            Determines how long the algorithm should sample neighbors of a solution,
            before decreasing the temperature.
        neighbors : int
            Number of neighbors to draw at each iteration.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)

        # Max iteration after each temperature decrease
        self.max_iter = max_iter

        # number of neighbors per inner iteration
        self.neighbors = neighbors

        # Cooling schedule
        self.cooling = cooling
        # inner iterations
        self.inner = 0

        self.initialized = False

    def reset(self):
        self.initialized = False

    def _one_inner(self, x_c: list, x_loss_c: np.ndarray) -> float:
        # Compute previous cost minus new cost
        delta = x_loss_c - self.x_loss_p

        logger.debug(f"New model score: {x_loss_c}")
        logger.debug(f"Old model score: {self.x_loss_p}")
        logger.debug(f"Best model score: {self.x_loss_b}")

        # If a better model is found, do...
        if delta < 0:
            self.x_p = x_c
            self.x_loss_p = x_loss_c

            if x_loss_c < self.x_loss_b:
                # Print if best model is found
                logger.debug("Better model found: YES ")
                self.x_b = x_c
                self.x_loss_b = x_loss_c
            else:
                logger.debug("Better model found: NO ")

            emdst = 0

        else:
            logger.debug("Better model found: NO ")

            p = np.random.uniform(0, 1)
            emdst = np.exp(-delta / self.cooling.Tcurrent)

            logger.debug(f"Escaping :  p<exp(-df/T) -->{p} < {emdst}")

            if p <= emdst:
                self.x_p = x_c
                self.x_loss_p = x_loss_c

        return float(emdst)

    # RUN SA
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of SA.
        SA is a local search and needs a starting point.
        X and Y must not be None.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        logger.info("Starting")

        # Determine the number of iterations according to the function parameters
        logger.debug("Determining number of iterations")

        if X is None:
            raise InputError(
                "Simulated annealing must be initialized by at least one solution, X."
            )
        elif Y is None:
            return X, {
                "algorithm": "InitSA",
                "temperature": -1,
                "probability": -1,
                "accepted": -1,
            }
        else:
            argmin = np.argmin(Y)
            # current point
            x_c = X[argmin]
            x_loss_c = Y[argmin]

        if self.initialized:
            # Simulated annealing starting
            if self.inner >= self.max_iter:
                self.inner = 0
                emdst = self._one_inner(x_c, x_loss_c)
            else:
                self.inner += 1
                emdst = self._one_inner(x_c, x_loss_c)

        else:
            logger.debug(f"Starting solution: {x_c}, {x_loss_c}")

            # best point
            self.x_b = x_c
            self.x_loss_b = x_loss_c
            # previous
            self.x_p = x_c
            self.x_loss_p = x_loss_c
            emdst = 0
            self.initialized = True

        logger.info("Ending")
        return self.search_space.neighborhood([x_c], size=self.neighbors), {
            "algorithm": "SA",
            "temperature": self.cooling.Tcurrent,
            "probability": emdst,
            "accepted": self.x_loss_p,
        }
