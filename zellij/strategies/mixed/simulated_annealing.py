# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-10T13:01:48+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.tools.cooling import Cooling
import zellij.utils.progress_bar as pb
import numpy as np
import os

import logging

logger = logging.getLogger("zellij.SA")


class Simulated_annealing(Metaheuristic):

    """Simulated_annealing

    Simulated_annealing (SA) is a hill climbing exploitation algorithm.

    It uses a :ref:`cooling` which partially drives the acceptance probability.

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

    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.neighborhoods import FloatInterval, ArrayInterval, Intervals
    >>> from zellij.strategies import Simulated_annealing
    >>> from zellij.strategies.tools import MulExponential
    >>> from zellij.utils.benchmarks import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(
    ...                           FloatVar("a",-5,5, neighbor=FloatInterval(0.5)),
    ...                           FloatVar("b",-5,5,neighbor=FloatInterval(0.5)),
    ...                           neighbor=ArrayInterval())
    ...                         ,lf, neighbor=Intervals())
    ...
    >>> cooling = MulExponential(0.85,100,2,3)
    >>> sa = Simulated_annealing(sp, cooling, 1, 5)
    ...
    >>> stop = Threshold(lf, 'calls', 100)
    >>> exp = Experiment(sa, stop)
    ...
    >>> x_start = [sp.random_point()]
    >>> _, y_start = lf(x_start)
    >>> sa.run(point, )
    >>> exp.run()
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")
    """

    # Initialize simulated annealing
    def __init__(
        self,
        search_space,
        cooling,
        max_iter,
        neighbors,
        verbose=True,
    ):
        """__init__(search_space, cooling, max_iter, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        cooling : Cooling
            Cooling schedule used to determine the probability of acceptance.

        max_iter : int
            Maximum iterations of the inner loop.
            Determines how long the algorithm should sample neighbors of a solution,\
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
        """reset()

        Reset SA variables to their initial values.

        """

        self.initialized = False

    def _one_inner(self, x_c, x_loss_c):
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

        return emdst

    # RUN SA
    def forward(self, X, Y):
        """forward(X, Y)
        Runs one step of Simulated_annealing.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        argmin = np.argmin(Y)
        # current point
        x_c = np.array(X[argmin])
        x_loss_c = Y[argmin]

        logger.info("Starting")

        # Determine the number of iterations according to the function parameters
        logger.debug("Determining number of iterations")

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

        return self.search_space.neighbor(x_c, size=self.neighbors), {
            "algorithm": "SA",
            "temperature": self.cooling.Tcurrent,
            "probability": emdst,
            "accepted": self.x_loss_p,
        }
