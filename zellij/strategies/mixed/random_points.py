# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-06T17:28:46+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic
import numpy as np

import logging

logger = logging.getLogger("zellij.Rnd")


class Random(Metaheuristic):

    """Random

    Samples random points from  the search space.

    Attributes
    ----------

    search_space : Searchspace
        Search space object containing bounds of the search space.

    size : int, default=1
        Number of points to sample at each :code:`forward`.

    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.
    """

    def __init__(
        self,
        search_space,
        size=1,
        verbose=True,
    ):
        """__init__(search_space, size=1, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

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

    # Run Random
    def forward(self, X, Y, constraints):
        """forward(X, Y)
        Runs one step of Random.

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

        logger.info("GA Starting")

        if self.size > 2:
            solutions = self.search_space.random_point(self.size)
        else:
            solutions = [self.search_space.random_point(self.size)]

        return solutions, {"algorithm": "Random"}
