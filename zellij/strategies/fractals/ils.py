# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.strategies.tools import Hypersphere, Section, Hypercube

from typing import List, Tuple, Union, Optional

import numpy as np
import logging

logger = logging.getLogger("zellij.ILS")


# Intensive local search
class ILS(ContinuousMetaheuristic):

    """ILS

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper.
    ILS is a local search, it uses the center of :ref:`sp` as
    a starting point.

    Attributes
    ----------
    search_space : Hypersphere
        Hypersphere, Section or Hypercube.
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

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypersphere
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import ILS

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
    >>> first_point = sp.random_point(1)
    >>> opt = ILS(sp)
    >>> stop = Calls(himmelblau, 1000)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run(X=first_point)
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.779310253377747, -3.283185991286169])=7.888609052210118e-31
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 1000

    """

    def __init__(
        self,
        search_space: Union[Hypersphere, Section, Hypercube],
        red_rate: float = 0.5,
        inflation: float = 1.75,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : {Hyperpshere, Section, Hypercube}
            An Hypersphere :ref:`sp`.
        red_rate : float, default=0.5
            Determines the step reduction rate at each solution improvement.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`. Not used for other Fractal.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        self.red_rate = red_rate
        self.inflation = inflation
        super().__init__(search_space, verbose)

        #############
        # VARIABLES #
        #############
        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    @property
    def red_rate(self) -> float:
        return self._red_rate

    @red_rate.setter
    def red_rate(self, value: float):
        if value > 0:
            self._red_rate = value
        else:
            raise InitializationError(f"red_rate must be >0. Got {value}.")

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value: Union[Hypersphere, Hypercube, Section]):
        if value:
            if isinstance(value, Hypersphere):
                self._search_space = value
                self.center = value.center
                self.radius = value.radius
                self.step = self.radius * self.inflation
            elif isinstance(value, (Hypercube, Section)):
                self._search_space = value
                self.center = (value.upper + value.lower) / 2
                self.radius = np.max(value.upper - value.lower)
                self.step = self.radius
            else:
                raise InitializationError(
                    f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
                )
        else:
            raise InitializationError(
                f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
            )

    def reset(self):
        super().reset()

        self.initialized = False
        self.improvement = False
        self.step = self.radius * self.inflation
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)

    def _one_step(self) -> Tuple[List[list], dict]:
        points = np.tile(self.current_point, (2, 1))
        points[0, self.current_dim] += self.step
        points[1, self.current_dim] -= self.step
        points[:, self.current_dim] = np.clip(points[:, self.current_dim], 0.0, 1.0)
        return points.tolist(), {"algorithm": "ILS"}

    # RUN ILS
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of ILS.
        ILS is a local search, it uses the center of :ref:`sp` as
        a starting point.

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

        # logging
        logger.info("Starting")

        if self.initialized:
            argmin = np.argmin(Y)

            if Y[argmin] < self.current_score:
                self.current_score = Y[argmin]
                self.current_point = X[argmin]
                self.improvement = True
        else:
            self.current_point = self.center
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


class ILSRandom(ILS):

    """ILSRandom

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper.
    ILS is a local search, it uses the center of :ref:`sp` as
    a starting point.

    Attributes
    ----------
    search_space : {Hypersphere, Section, Hypercube}
        Hypersphere, Section or Hypercube.
    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.
    precision : flaot, default=1e-20
        When :code:`step`<:code:`precision`, stops the algorithm.
    verbose : boolean, default=True
        Activate or deactivate the progress bar.


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
    >>> from zellij.strategies.fractals import ILSRandom

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
    >>> first_point = sp.random_point(1)
    >>> opt = ILSRandom(sp)
    >>> stop = Calls(himmelblau, 1000)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run(X=first_point)
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-2.805118086952745, 3.1313125182505726])=7.099748146989106e-30
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 1000

    """

    def __init__(
        self,
        search_space: Union[Hypersphere, Section, Hypercube],
        red_rate: float = 0.5,
        inflation: float = 1.75,
        verbose: bool = True,
    ):
        super().__init__(search_space, red_rate, inflation, verbose)
        self.direction = self._random_direction()

    def _random_direction(self) -> np.ndarray:
        newp = np.random.normal(size=self.search_space.size)
        norm = np.linalg.norm(newp)
        return newp / norm

    def _one_step(self) -> Tuple[List[list], dict]:
        points = np.empty((2, self.search_space.size))
        newp = self.step * self.direction + self.current_point
        points[0] = newp
        points[1] = newp - self.step
        points = np.clip(points, 0.0, 1.0)

        return points.tolist(), {"algorithm": "ILS_random"}

    # RUN ILS
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of ILSRandom.
        ILS is a local search, it uses the center of :ref:`sp` as
        a starting point.

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

        # logging
        logger.info("Starting")

        if self.initialized:
            argmin = np.argmin(Y)

            if Y[argmin] < self.current_score:
                self.current_score = Y[argmin]
                self.current_point = X[argmin]
                self.improvement = True
        else:
            self.current_point = self.center
            self.initialized = True

        if self.current_dim >= self.search_space.size:
            self.current_dim = 0
            if self.improvement:
                self.improvement = False
            else:
                self.direction = self._random_direction()
                self.step *= self.red_rate

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return self._one_step()

        else:
            if self.improvement:
                to_return = self._one_step()
            else:
                self.direction = self._random_direction()
                to_return = self._one_step()

            self.current_dim += 1

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return to_return
