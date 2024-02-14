# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import abstractmethod, ABC

from typing import Optional, Tuple, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from zellij.core.search_space import (
    ContinuousSearchspace,
    DiscreteSearchspace,
    Fractal,
    UnitSearchspace,
    Searchspace,
)

import numpy as np
import logging

logger = logging.getLogger("zellij.meta")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class Metaheuristic(ABC):

    """Metaheuristic

    :ref:`meta` is a core object which defines the structure
    of a metaheuristic in Zellij. It is an abtract class.
    The :code:`forward`, describes one iteration of the :ref:`meta`, and returns
    the solutions that must be computed by the loss function.
    :code:`forwards` takes as parameters, a list of computed solutions and their objective values.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space: Searchspace, verbose: bool = True):
        ##############
        # PARAMETERS #
        ##############
        self.search_space = search_space
        self.verbose = verbose

        #############
        # VARIABLES #
        #############
        # Saving file initialized by Experience
        self._save = ""

    @property
    def search_space(self) -> Searchspace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: Searchspace):
        self._search_space = value

    @abstractmethod
    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray],
        constraint: Optional[np.ndarray],
    ) -> Tuple[List[list], dict]:
        """forward

        Abstract method describing one step of the :ref:`meta`.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.
        secondary : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.
        constraint : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Dictionnary of additionnal information linked to :code:`points`.
        """
        pass

    def reset(self):
        """reset()

        reset :ref:`meta` to its initial value

        """
        pass


class ContinuousMetaheuristic(Metaheuristic):

    """ContinuousMetaheuristic

    ContinuousMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only with a continuous :ref:`sp`.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(
        self, search_space: Union[ContinuousSearchspace, Fractal], verbose: bool = True
    ):
        super().__init__(search_space=search_space, verbose=verbose)

    @property
    def search_space(self) -> Union[ContinuousSearchspace, Fractal]:
        return self._search_space

    @search_space.setter
    def search_space(self, value: Union[ContinuousSearchspace, Fractal]):
        if isinstance(value, (ContinuousSearchspace, Fractal)):
            self._search_space = value
        else:
            raise ValueError(
                f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
            )


class UnitMetaheuristic(Metaheuristic):

    """UnitMetaheuristic

    UnitMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only within the continuous unit hypercube.

    Attributes
    ----------
    search_space : UnitSearchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space: UnitSearchspace, verbose: bool = True):
        super().__init__(search_space=search_space, verbose=verbose)

    @property
    def search_space(self) -> UnitSearchspace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: UnitSearchspace):
        if isinstance(value, UnitSearchspace):
            self._search_space = value
        else:
            raise ValueError(f"Search space must be a UnitSearchspace. Got {value}")


class DiscreteMetaheuristic(Metaheuristic):

    """DiscreteMetaheuristic

    ContinuousMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only with a discrete :ref:`sp`.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space: DiscreteSearchspace, verbose: bool = True):
        super().__init__(search_space=search_space, verbose=verbose)

    @property
    def search_space(self) -> DiscreteSearchspace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: DiscreteSearchspace):
        if isinstance(value, DiscreteSearchspace):
            self._search_space = value
        else:
            raise ValueError(
                "Search space must be discrete or have a `converter` addon"
            )


class MockMixedMeta(Metaheuristic):
    """MockMeta

    Mock metaheuristc to test behaviors of Zellij.
    Return random points from :ref:`sp`.

    Parameters
    ----------
    points : int
        Number of random points to return at each iteration.
    iteration_error : int, optional
        At which iteration MockMeta should return None points.
    """

    def __init__(
        self,
        points: int,
        search_space: Searchspace,
        verbose: bool = True,
        iteration_error=None,
    ):
        super().__init__(search_space, verbose)
        self.points = points
        self.iteration_error = iteration_error if iteration_error else float("inf")

        self.iteration = 0

    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray],
        constraint: Optional[np.ndarray],
    ) -> Tuple[list, dict]:
        info = {"algorithm": "MockMeta", "iteration": self.iteration}
        if self.verbose:
            print(
                f"""
                MockMeta received:
                X: {X},
                Y: {Y},
                secondary: {secondary},
                constraint: {constraint},
                """
            )
        if self.iteration >= self.iteration_error:
            return [], info
        else:
            self.iteration += 1
            return self.search_space.random_point(self.points), info
