import numpy as np
from zellij.strategies.tools.chaos_map import Henon, Chaos_map
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.search_space import ContinuousSearchspace
import logging

logger = logging.getLogger("zellij.sampling")

# Promising Hypersphere Search
class Center(Metaheuristic):

    """Center

    Sample the center of the targeted search space.
    The search space must have a :code:`center` attribute.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    f_calls : int
        Maximum number of calls to :ref:`lf`.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    Methods
    -------

    run(self, n_process=1)
        Runs Center


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, f_calls=1, verbose=True):

        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of :ref:`lf` calls

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, f_calls, verbose)

    def run(self, H=None, n_process=1):
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

        points = [H.center]

        self.build_bar(self.f_calls)

        # logging
        logger.info("Starting")

        self.pending_pb(1)

        logger.info(f"Evaluating points")
        if (
            isinstance(self.search_space, ContinuousSearchspace)
            or not H.to_convert
        ):
            scores = self.search_space.loss(points, algorithm="Center")
        else:
            scores = self.search_space.loss(
                self.search_space.to_continuous.reverse(points, True),
                algorithm="Center",
            )

        self.update_main_pb(
            1, explor=True, best=self.search_space.loss.new_best
        )
        self.meta_pb.update(1)

        logger.info("Ending")

        self.close_bar()

        logger.info("Center ending")

        idx = np.argmin(scores)
        return points[idx], scores[idx]


class Diagonal(Metaheuristic):

    """Diagonal

    Sample the center of the targeted search space, and two equidistant points
    on the diagonal of an hypercube.
    The search space must be an hypercube or an hyperrectangle.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    f_calls : int
        Maximum number of calls to :ref:`lf`.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    Methods
    -------

    run(self, n_process=1)
        Runs Diagonal


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, f_calls=1, ratio=0.8, verbose=True):

        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of :ref:`lf` calls

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, f_calls, verbose)
        self.ratio = ratio

    def run(self, H=None, n_process=1):
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

        upmlo = (H.up_bounds - H.lo_bounds) / 2
        points = [
            H.center,
            np.array(H.center) + self.ratio * upmlo,
            np.array(H.center) - self.ratio * upmlo,
        ]

        self.build_bar(self.f_calls)

        # logging
        logger.info("Starting")

        self.pending_pb(3)

        logger.info(f"Evaluating points")
        if (
            isinstance(self.search_space, ContinuousSearchspace)
            or not H.to_convert
        ):
            scores = self.search_space.loss(points, algorithm="Diagonal")
        else:
            scores = self.search_space.loss(
                self.search_space.to_continuous.reverse(points, True),
                algorithm="Diagonal",
            )

        self.update_main_pb(
            1, explor=True, best=self.search_space.loss.new_best
        )
        self.meta_pb.update(1)

        logger.info("Ending")

        self.close_bar()

        logger.info("Diagonal ending")

        idx = np.argmin(scores)
        return points[idx], scores[idx]


class Chaos(Metaheuristic):

    """Chaos

    Sample points in a chaotic fashion.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    f_calls : int
        Maximum number of calls to :ref:`lf`.

    map : Chaos_map, default=Henon
        :ref:`cmap` used to generate points.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    Methods
    -------

    run(self, n_process=1)
        Runs Chaos


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
        f_calls=1,
        map=Henon,
        verbose=True,
        seed=None,
    ):

        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of :ref:`lf` calls

        verbose : boolean, default=True
            Algorithm verbosity

        """
        np.random.seed(seed)
        super().__init__(search_space, f_calls, verbose)
        if isinstance(map, Chaos_map):
            self.map = map.map
        else:
            self.map = map(samples, len(self.search_space)).map

    def run(self, H=None, n_process=1):
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

        self.build_bar(self.f_calls)

        # logging
        logger.info("Starting")

        self.pending_pb(len(self.map))

        logger.info(f"Evaluating points")
        if (
            isinstance(self.search_space, ContinuousSearchspace)
            or not H.to_convert
        ):
            points = (
                self.map
                * (self.search_space.up_bounds - self.search_space.lo_bounds)
                + self.search_space.lo_bounds
            )
            scores = self.search_space.loss(points, algorithm="Chaos")
        else:
            scores = self.search_space.loss(
                self.search_space.to_continuous.reverse(self.map, True),
                algorithm="Chaos",
            )

        self.update_main_pb(
            len(self.map), explor=True, best=self.search_space.loss.new_best
        )
        self.meta_pb.update(len(self.map))

        logger.info("Ending")

        self.close_bar()

        logger.info("Diagonal ending")

        idx = np.argmin(scores)
        return points[idx], scores[idx]


class Chaos_Hypersphere(Metaheuristic):

    """Chaos_Hypersphere

    Sample points in a chaotic fashion. For hypersphere

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    f_calls : int
        Maximum number of calls to :ref:`lf`.

    map : Chaos_map, default=Henon
        :ref:`cmap` used to generate points.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    Methods
    -------

    run(self, n_process=1)
        Runs Chaos


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
        f_calls=1,
        map=Henon,
        verbose=True,
        seed=None,
    ):

        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of :ref:`lf` calls

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, f_calls, verbose)
        if isinstance(map, Chaos_map):
            self.map = map[0].map
            self.map = self.map * 2 - 1
            inner = map[1].map

            d = np.linalg.norm(self.map, axis=1, keepdims=True)
            self.map = self.map * inner ** (1 / len(self.search_space)) / d
        else:
            np.random.seed(seed)
            self.map = map(samples, len(self.search_space)).map
            self.map = self.map * 2 - 1
            inner = map(samples, 1).map

            d = np.linalg.norm(self.map, axis=1, keepdims=True)
            self.map = self.map * inner ** (1 / len(self.search_space)) / d

    def run(self, H=None, n_process=1):
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

        self.build_bar(self.f_calls)

        # logging
        logger.info("Starting")

        points = self.map + self.search_space.center
        points *= self.search_space.radius
        points = np.maximum(points, self.search_space._god.lo_bounds)
        points = np.minimum(points, self.search_space._god.up_bounds)

        self.pending_pb(len(points))

        logger.info(f"Evaluating points")
        if (
            isinstance(self.search_space, ContinuousSearchspace)
            or not H.to_convert
        ):
            scores = self.search_space.loss(points, algorithm="ChaosH")
        else:
            scores = self.search_space.loss(
                self.search_space.to_continuous.reverse(points, True),
                algorithm="ChaosH",
            )

        self.update_main_pb(
            len(self.map), explor=True, best=self.search_space.loss.new_best
        )
        self.meta_pb.update(len(points))

        logger.info("Ending")

        self.close_bar()

        logger.info("Diagonal ending")

        idx = np.argmin(scores)
        return points[idx], scores[idx]
