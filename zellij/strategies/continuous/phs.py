# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-19T19:21:57+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.core.metaheuristic import Metaheuristic

import logging

logger = logging.getLogger("zellij.PHS")

# Promising Hypersphere Search
class PHS(Metaheuristic):

    """PHS

    Promising Hypersphere Search  is an exploration algorithm comming from the original FDA paper.
    It is used to evaluate the center of an Hypersphere, and fixed points on each dimension arround this center.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    Methods
    -------

    forward(X, Y)
        Runs one step of PHS.


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, verbose=True):

        """__init__(search_space,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)

    def search_space():
        doc = "The search_space property."

        def fget(self):
            return super().search_space

        def fset(self, value):
            super(PHS, self.__class__).search_space.fset(self, value)
            if value:
                self.radius = np.tile(
                    self.search_space.inflation
                    * self.search_space.radius
                    / np.sqrt(self.search_space.size),
                    (2, 1),
                )
                self.radius[1] = -self.radius[1]

        def fdel(self):
            super().search_space.fdel()

        return locals()

    search_space = property(**search_space())

    def forward(self, X, Y):

        """forward(X, Y)
        Runs one step of PHS.

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

        points = np.tile(self.search_space.center, (3, 1))

        # logging
        logger.info("Starting")

        points[1:] += self.radius
        points[1:] = np.clip(
            points[1:],
            self.search_space._god.lo_bounds,
            self.search_space._god.up_bounds,
        )

        return points, {"algorithm": "PHS"}
