# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:25+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.search_space import ContinuousSearchspace

import logging

logger = logging.getLogger("zellij.ILS")

# Intensive local search
class ILS(Metaheuristic):

    """ILS

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper. It evaluate a point in each dimension arround
    an initial solution. Distance of the computed point to the initial one is
    decreasing according to a reduction rate. At each iteration the algorithm
    moves to the best solution found.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    f_calls : int
        Maximum number of calls to search.space_space.loss.

    save : boolean, optional
        If True save results into a file

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    red_rate : float
        determine the step reduction rate ate each iteration.

    precision : float
        dtermine the stopping criterion.
        When the step is lower than the precision the algorithm stops.

    Methods
    -------

    run(self, n_process=1)
        Runs ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        f_calls,
        red_rate=0.5,
        precision=1e-20,
        verbose=True,
    ):

        """__init__(search_space, f_calls,save=False,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of :ref:`lf` calls

        red_rate : float, default=0.5
            determine the step reduction rate ate each iteration.

        precision : float, default=1e-20
            dtermine the stopping criterion. When the step is lower than <precision> the algorithm stops.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, f_calls, verbose)
        # assert hasattr(search_space, "to_continuous") or isinstance(
        #     search_space, ContinuousSearchspace
        # ), logger.error(
        #     f"""If the `search_space` is not a `ContinuousSearchspace`,
        #     the user must give a `Converter` to the :ref:`sp` object
        #     with the kwarg `to_continuous`"""
        # )

        self.red_rate = red_rate
        self.precision = precision

    def run(
        self, X0=None, Y0=None, H=None, radius=None, inflation=None, n_process=1
    ):

        """run(X0=None, Y0=None, H=None, n_process=1)

        Parameters
        ----------
        X0 : list[float], optional
            Initial solution. If None, a Fractal must be given (H!=None)
        Y0 : {int, float}, optional
            Score of the initial solution
            Determine the starting point of the chaotic map.
        H : :ref:`frac`, optional
            Instend of X0, user can give a :ref:`frac`.
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        # logging
        logger.info("Starting")

        self.build_bar(self.f_calls)

        scores = np.zeros(3, dtype=float)

        if X0:
            if self.search_space.to_convert:
                self.X0 = np.array(
                    self.search_space.to_continuous.convert([X0], True)[0]
                )
            else:
                self.X0 = np.array(X0)

            points = np.tile(self.X0, (3, 1))

        elif H:
            self.X0 = np.array(H.center)
            points = np.tile(H.center, (3, 1))
        else:
            raise ValueError(
                "No starting point given to Intensive Local Search"
            )

        if Y0:
            self.Y0 = Y0
        else:
            logger.info("ILS evaluating initial solution")
            if (
                isinstance(self.search_space, ContinuousSearchspace)
                or not H.to_convert
            ):
                self.Y0 = self.search_space.loss([self.X0], algorithm="ILS")[0]
            else:
                self.Y0 = self.search_space.loss(
                    self.search_space.to_continuous.reverse([self.X0], True),
                    algorithm="ILS",
                )[0]

        scores[0] = self.Y0
        if H:
            step = H.radius
        else:
            step = radius * inflation
        while (
            step > self.precision
            and self.search_space.loss.calls < self.f_calls
        ):
            i = 0
            improvement = False
            # logging
            logger.debug(f"ILS {step}>{self.precision}")

            while (
                i < self.search_space.size
                and self.search_space.loss.calls < self.f_calls
            ):

                # logging
                logger.debug(f"Evaluating dimension {i}")
                self.pending_pb(2)

                walk = points[0][i] + step
                points[1][i] = walk
                points[1][i] = max(
                    points[1][i], self.search_space._god.lo_bounds[i]
                )
                points[1][i] = min(
                    points[1][i], self.search_space._god.up_bounds[i]
                )

                walk = points[0][i] - step
                points[2][i] = walk
                points[2][i] = max(
                    points[2][i], self.search_space._god.lo_bounds[i]
                )
                points[2][i] = min(
                    points[2][i], self.search_space._god.up_bounds[i]
                )

                if self.search_space.to_convert:
                    scores[1:] = self.search_space.loss(
                        self.search_space.to_continuous.reverse(
                            points[1:], True
                        ),
                        algorithm="ILS",
                    )
                else:
                    scores[1:] = self.search_space.loss(
                        points[1:], algorithm="ILS"
                    )

                min_index = np.argmin(scores)

                if scores[min_index] < scores[0]:
                    points = np.tile(points[min_index], (3, 1))
                    scores[0] = scores[min_index]
                    improvement = True

                self.update_main_pb(
                    2, explor=False, best=self.search_space.loss.new_best
                )
                self.meta_pb.update(2)

                i += 1

            if not improvement:
                step = self.red_rate * step

        # logging
        logger.info("Ending")
        self.close_bar()

        return points[0], scores[0]


# Intensive local search
class ILS_section(Metaheuristic):

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

    f_calls : int
        Maximum number of calls to search.space_space.loss.

    save : boolean, optional
        If True save results into a file

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    red_rate : float
        determine the step reduction rate ate each iteration.

    precision : float
        dtermine the stopping criterion.
        When the step is lower than the precision the algorithm stops.

    Methods
    -------

    run(self, n_process=1)
        Runs ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        f_calls,
        red_rate=0.5,
        precision=1e-20,
        verbose=True,
    ):

        """__init__(search_space, f_calls,save=False,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of :ref:`lf` calls

        red_rate : float, default=0.5
            determine the step reduction rate ate each iteration.

        precision : float, default=1e-20
            dtermine the stopping criterion. When the step is lower than <precision> the algorithm stops.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, f_calls, verbose)

        self.red_rate = red_rate
        self.precision = precision

    def run(
        self, X0=None, Y0=None, H=None, radius=None, inflation=None, n_process=1
    ):

        """run(X0=None, Y0=None, H=None, n_process=1)

        Parameters
        ----------
        X0 : list[float], optional
            Initial solution. If None, a Fractal must be given (H!=None)
        Y0 : {int, float}, optional
            Score of the initial solution
            Determine the starting point of the chaotic map.
        H : :ref:`frac`, optional
            Instend of X0, user can give a :ref:`frac`.
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        # logging
        logger.info("Starting")

        self.build_bar(self.f_calls)

        scores = np.zeros(3, dtype=float)

        if X0:
            if self.search_space.to_convert:
                self.X0 = np.array(
                    self.search_space.to_continuous.convert([X0], True)[0]
                )
            else:
                self.X0 = np.array(X0)

            points = np.tile(self.X0, (3, 1))

        elif H:
            self.X0 = np.array(H.center)
            points = np.tile(H.center, (3, 1))
        else:
            raise ValueError(
                "No starting point given to Intensive Local Search"
            )

        if Y0:
            self.Y0 = Y0
        else:
            logger.info("ILS evaluating initial solution")
            if (
                isinstance(self.search_space, ContinuousSearchspace)
                or not H.to_convert
            ):
                self.Y0 = self.search_space.loss(
                    [self.X0], algorithm="ILS_section"
                )[0]
            else:
                self.Y0 = self.search_space.loss(
                    self.search_space.to_continuous.reverse([self.X0], True),
                    algorithm="ILS_section",
                )[0]

        scores[0] = self.Y0
        if H:
            step = H.length
        else:
            step = radius * inflation
        while (
            step > self.precision
            and self.search_space.loss.calls < self.f_calls
        ):
            i = 0
            improvement = False
            # logging
            logger.debug(f"ILS {step}>{self.precision}")

            while (
                i < self.search_space.size
                and self.search_space.loss.calls < self.f_calls
            ):

                # logging
                logger.debug(f"Evaluating dimension {i}")
                self.pending_pb(2)

                walk = points[0][i] + step
                points[1][i] = walk
                points[1][i] = max(
                    points[1][i], self.search_space._god.lo_bounds[i]
                )
                points[1][i] = min(
                    points[1][i], self.search_space._god.up_bounds[i]
                )

                walk = points[0][i] - step
                points[2][i] = walk
                points[2][i] = max(
                    points[2][i], self.search_space._god.lo_bounds[i]
                )
                points[2][i] = min(
                    points[2][i], self.search_space._god.up_bounds[i]
                )

                if self.search_space.to_convert:
                    scores[1:] = self.search_space.loss(
                        self.search_space.to_continuous.reverse(
                            points[1:], True
                        ),
                        algorithm="ILS_section",
                    )
                else:
                    scores[1:] = self.search_space.loss(
                        points[1:], algorithm="ILS_section"
                    )

                min_index = np.argmin(scores)

                if scores[min_index] < scores[0]:
                    points = np.tile(points[min_index], (3, 1))
                    scores[0] = scores[min_index]
                    improvement = True

                self.update_main_pb(
                    2, explor=False, best=self.search_space.loss.new_best
                )
                self.meta_pb.update(2)

                i += 1

            if not improvement:
                step = self.red_rate * step

        # logging
        logger.info("Ending")
        self.close_bar()

        return points[0], scores[0]
