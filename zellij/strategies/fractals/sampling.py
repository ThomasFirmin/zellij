# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-12-23T02:22:10+01:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-04T11:08:44+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.strategies.tools.chaos_map import Henon, Chaos_map
from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.core.search_space import Fractal, ContinuousSearchspace
import logging

logger = logging.getLogger("zellij.sampling")


class Center(ContinuousMetaheuristic):

    """Center

    Samples the center of the targeted search space.
    The search space must have a :code:`center` attribute, or
    upper and lower bounds.

    Attributes
    ----------
    search_space : Searchspace
            :ref:`sp` containing bounds of the search space.
    verbose : boolean, default=True
        Algorithm verbosity


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, verbose=True):
        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        verbose : boolean, default=True
            Algorithm verbosity

        """

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

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.upper + value.lower) / 2

    def forward(self, X, Y):
        """run(X, Y)

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

        return [self.center], {"algorithm": "Center"}


class CenterSOO(ContinuousMetaheuristic):

    """Center

    Samples the center of the targeted search space.
    The search space must have a :code:`center` attribute, or
    upper and lower bounds. If fractal :code:`is_middle`,
    then return empty list.

    Attributes
    ----------
    search_space : Searchspace
            :ref:`sp` containing bounds of the search space.
    verbose : boolean, default=True
        Algorithm verbosity


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, verbose=True):
        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)
        self.computed = False

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

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.lower + value.upper) / 2

    def forward(self, X, Y):
        """run(X, Y)

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
        if np.isfinite(self.search_space.score) and self.search_space.is_middle:
            self.computed = True
            return [], {"algorithm": "Center"}
        else:
            self.computed = True
            return [self.center], {"algorithm": "Center"}

    def reset(self):
        self.computed = False


class Diagonal(ContinuousMetaheuristic):

    """Diagonal

    Sample the center of the :ref:`sp`, and two equidistant points
    on the diagonal.
    The search space must be a :code:`Hypercube` or a :code:`Section`.

    Attributes
    ----------
    search_space : Searchspace
            :ref:`sp` containing bounds of the search space.
    ratio : float, default=0.8
        0.0<:code:`ratio`<=1.0.
        Proportionnal distance of sampled points from the center.
    verbose : boolean, default=True
        Algorithm verbosity


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, ratio=0.8, verbose=True):
        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            :ref:`sp` containing bounds of the search space.
        ratio : float, default=0.8
            0.0<:code:`ratio`<=1.0.
            Proportionnal distance of sampled points from the center.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        self.ratio = ratio
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

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.lower + value.upper) / 2

    def forward(self, X, Y):
        """run(H=None, n_process=1)

        Parameters
        ----------
        H : Fractal, default=None
            When used by :ref:`dba`, a fractal corresponding to the current subspace is given
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        points = [
            self.center,
            np.array(self.center) + self.ratio * self.center,
            np.array(self.center) - self.ratio * self.center,
        ]

        # logging
        logger.info("Starting")

        return points, {"algorithm": "Diagonal"}


class Chaos(ContinuousMetaheuristic):

    """Chaos

     Sample points in a chaotic fashion.

    Attributes
     ----------
     search_space : Searchspace
             :ref:`sp` containing bounds of the search space.
     verbose : boolean, default=True
         Algorithm verbosity


     See Also
     --------
     Metaheuristic : Parent class defining what a Metaheuristic is.
     LossFunc : Describes what a loss function is in Zellij
     Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        samples,
        cmap=Henon,
        verbose=True,
        seed=None,
    ):
        """__init__(search_space, f_calls,verbose=True)

        Initialize Chaos class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        samples : int
            Number of sampled points.
        cmap : ChaosMap, default=Henon
            ChaosMap.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        self.seed = seed
        self.samples = samples
        super().__init__(search_space, verbose)
        self.map = cmap

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        np.random.seed(value)

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, value):
        if isinstance(value, Chaos_map):
            self._map = value.map
        else:
            self._map = value(self.samples, len(self.search_space)).map

    def forward(self, X, Y):
        """run(X, Y)

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

        logger.info(f"Evaluating points")
        points = (
            self.map * (self.search_space.upper - self.search_space.lower)
            + self.search_space.lower
        )

        logger.info("Ending")

        return points, {"algorithm": "Chaos"}


class Chaos_Hypersphere(ContinuousMetaheuristic):

    """Chaos_Hypersphere

    Sample points in a chaotic fashion. Adapted to :code:`Hypersphere`.

    Attributes
     ----------
     search_space : Searchspace
             :ref:`sp` containing bounds of the search space.
     verbose : boolean, default=True
         Algorithm verbosity


     See Also
     --------
     Metaheuristic : Parent class defining what a Metaheuristic is.
     LossFunc : Describes what a loss function is in Zellij
     Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        samples,
        cmap=Henon,
        verbose=True,
        seed=None,
    ):
        """__init__(search_space, f_calls,verbose=True)

        Initialize Chaos class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        samples : int
            Number of sampled points.
        cmap : ChaosMap, default=Henon
            ChaosMap.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        self.seed = seed
        self.samples = samples
        super().__init__(search_space, verbose)
        self.map = cmap

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        np.random.seed(value)

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, value):
        if isinstance(value, Chaos_map):
            self._map = value.map
        else:
            self._map = value(self.samples, len(self.search_space)).map

    def forward(self, X, Y):
        """run(X, Y)

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

        points = self.map + self.search_space.center
        points *= self.search_space.radius
        points = np.maximum(points, 0)
        points = np.minimum(points, 1)

        return points, {"algorithm": "ChaosH"}


class DirectSampling(ContinuousMetaheuristic):

    """DirectSampling

    Samples points as in DIRECT algorithms.

    Attributes
    ----------
    search_space : Searchspace
            :ref:`sp` containing bounds of the search space.
    verbose : boolean, default=True
        Algorithm verbosity


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, verbose=True):
        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)
        self.computed = False

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

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.upper + value.lower) / 2

    def forward(self, X, Y):
        """run(X, Y)

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

        center = (self.search_space.lower + self.search_space.upper) / 2

        if self.search_space.level == 0:
            n_points = 2 * len(self.search_space.set_i) + 1
            start = 1
        else:
            n_points = 2 * len(self.search_space.set_i)
            start = 0
        section_length = self.search_space.width / 3

        points = np.tile(center, (n_points, 1))

        for i, p1, p2 in zip(
            self.search_space.set_i,
            points[start:-1:2],
            points[start + 1 :: 2],
        ):
            p1[i] -= section_length
            p2[i] += section_length

        self.computed = True

        return points, {"algorithm": "DIRECT"}

    def reset(self):
        self.computed = False
