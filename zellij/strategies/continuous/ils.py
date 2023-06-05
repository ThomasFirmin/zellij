# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-19T19:21:45+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.core.metaheuristic import Metaheuristic

import logging

logger = logging.getLogger("zellij.ILS")

# Intensive local search
class ILS(Metaheuristic):

    """ILS

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.
    precision : flaot, default=1e-20
        When :code:`step`<:code:`precision`, stops the algorithm.
    verbose : boolean, default=True
        Activate or deactivate the progress bar.


    Methods
    -------

    forward(X, Y)
        Runs one step of ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        red_rate=0.5,
        precision=1e-20,
        verbose=True,
    ):

        """__init__(search_space, red_rate=0.5,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        red_rate : float, default=0.5
            Determines the step reduction rate each time an improvement happens.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)

        self.red_rate = red_rate
        self.precision = precision
        self.step = np.tile(
            self.search_space.radius * self.search_space.inflation, 2
        )
        self.step[1] = -self.step[1]

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    def search_space():
        doc = "The search_space property."

        def fget(self):
            return super().search_space.fget()

        def fset(self, value):
            super(ILS, self.__class__).search_space.fset(self, value)
            if value:
                self.step = np.tile(
                    self.search_space.radius * self.search_space.inflation, 2
                )
                self.step[1] = -self.step[1]

        def fdel(self):
            super().search_space.fdel()

        return locals()

    def reset(self):
        """reset()

        Reset ILS variables to their initial values.

        """

        self.step = np.tile(
            self.search_space.radius * self.search_space.inflation, 2
        )
        self.step[1] = -self.step[1]

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    def _one_step(self):
        if self.step[0] > self.precision:
            points = np.tile(self.current_point, (2, 1))
            points[:, self.current_dim] += self.step
            points[:, self.current_dim] = np.clip(
                points[:, self.current_dim],
                self.search_space._god.lo_bounds[self.current_dim],
                self.search_space._god.up_bounds[self.current_dim],
            )

            return points, {"algorithm": "ILS"}
        else:
            return [], "endILSprecision"

    def forward(self, X, Y):

        """forward(X, Y)
        Runs one step of ILS.

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

        # logging
        logger.info("Starting")

        if self.initialized:
            argmin = np.argmin(Y)

            if Y[argmin] < self.current_score:
                self.current_score = Y[argmin]
                self.current_point = X[argmin]
                self.improvement = True
        else:
            self.current_point = X[0]
            self.initialized = True

        if self.current_dim >= self.search_space.size:
            self.current_dim = 0
            if self.improvement:
                self.improvement = False
            else:
                self.step *= self.red_rate

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return self._one_step()

        else:
            to_return = self._one_step()
            self.current_dim += 1

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return to_return


# Intensive local search
class ILS_section(ILS):

    """ILS_section

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper. It evaluate a point in each dimension arround
    an initial solution. Distance of the computed point to the initial one is
    decreasing according to a reduction rate. At each iteration the algorithm
    moves to the best solution found.

    This variation works with sections

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.


    Methods
    -------

    forward(X, Y)
        Runs one step of ILS.


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        red_rate=0.5,
        verbose=True,
    ):

        """__init__(search_space, red_rate=0.5,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        red_rate : float, default=0.5
            Determines the step reduction rate each time an improvement happens.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, red_rate, verbose)

        self.step = np.tile(self.search_space.length, 2)
        self.step[1] = -self.step[1]

    def reset(self):
        """reset()

        Reset ILS variables to their initial values.

        """

        self.step = np.tile(self.search_space.length, 2)
        self.step[1] = -self.step[1]

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False
