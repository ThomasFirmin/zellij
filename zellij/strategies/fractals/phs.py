# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-19T19:21:57+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.core.search_space import Fractal, ContinuousSearchspace

import logging

logger = logging.getLogger("zellij.PHS")


# Promising Hypersphere Search
class PHS(ContinuousMetaheuristic):

    """PHS

    Promising Hypersphere Search  is an exploration algorithm comming from the original FDA paper.
    It is used to evaluate the center of an Hypersphere, and fixed points on each dimension arround this center.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.
    inflation : float, default=1.75
        Inflation rate of the :code:`Hypersphere`
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

    def __init__(self, search_space, inflation=1.75, verbose=True):
        """__init__(search_space,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            :ref:`sp` object containing decision variables and the loss function.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`
        verbose : boolean, default=True
            Activate or deactivate the progress bar.

        """

        self.inflation = inflation
        super().__init__(search_space, verbose)

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value):
        if value:
            if (
                isinstance(value, ContinuousSearchspace)
                or isinstance(value, Fractal)
                or hasattr(value, "converter")
            ):
                self._search_space = value
            else:
                raise ValueError(
                    f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
                )

            if not (hasattr(value, "lower") and hasattr(value, "upper")):
                raise AttributeError(
                    "Search space must have lower and upper bounds attributes, got {value}."
                )

            self.radius = (
                np.tile(
                    self.inflation * self.search_space.radius,
                    (2, 1),
                )
                / self.search_space.size
            )
            self.radius[1] = -self.radius[1]

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
        points[1:] = np.clip(points[1:], 0.0, 1.0)

        return points, {"algorithm": "PHS"}
