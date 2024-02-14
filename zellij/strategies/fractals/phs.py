# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.strategies.tools import Hypersphere

from typing import List, Tuple, Optional

import numpy as np

import logging

logger = logging.getLogger("zellij.PHS")


# Promising Hypersphere Search
class PHS(ContinuousMetaheuristic):
    """PHS

    Promising Hypersphere Search  is an exploration algorithm comming from the original FDA paper.
    It is used to evaluate the center of an Hypersphere, and fixed points on each dimension arround this center.

    Attributes
    ----------
    search_space : Hypersphere
        A Hypersphere.
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

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypersphere
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import PHS

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = Hypersphere(a)
    >>> opt = PHS(sp, inflation=1)
    >>> stop = Calls(himmelblau, 3)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.5355339059327373, -3.5355339059327373])=8.002525316941673
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 3
    """

    def __init__(
        self, search_space: Hypersphere, inflation: float = 1.75, verbose: bool = True
    ):
        """__init__

        Parameters
        ----------
        search_space : Hypersphere
            :ref:`sp` object containing decision variables and the loss function.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`
        verbose : boolean, default=True
            Activate or deactivate the progress bar.

        """

        self.inflation = inflation
        super().__init__(search_space, verbose)
        self.computed = False

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value: Hypersphere):
        if value and isinstance(value, Hypersphere):
            self._search_space = value
            self.center = value.center
            self.radius = self.inflation * value.radius / np.sqrt(value.size)
        else:
            raise InitializationError(
                f"Search space must be a Hyperpshere, got {type(value)}."
            )

    # RUN PHS
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of PHS.
        PHS does not use secondary and constraint.

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

        if self.computed:
            return [], {"algorithm": "EndPHS"}
        else:
            points = np.tile(self.center, (3, 1))
            points[1] -= self.radius
            points[2] += self.radius
            points[1:] = np.clip(points[1:], 0.0, 1.0)

            self.computed = True
            return points.tolist(), {"algorithm": "PHS"}

    def reset(self):
        super().reset()
        self.computed = False
