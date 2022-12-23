# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:31+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.search_space import ContinuousSearchspace

import logging

logger = logging.getLogger("zellij.PHS")

# Promising Hypersphere Search
class PHS(Metaheuristic):

    """PHS

    Promising Hypersphere Search  is an exploration algorithm comming from original FDA paper.
    It used to evaluate the center of an Hypersphere, and fixed points on each dimension arround this center.

    It works on a continuous searh space.

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
        Runs PHS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space, f_calls, verbose=True):

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
        # assert hasattr(search_space, "to_continuous") or isinstance(
        #     search_space, ContinuousSearchspace
        # ) or , logger.error(
        #     f"""If the `search_space` is not a `ContinuousSearchspace`,
        #     the user must give a `Converter` to the :ref:`sp` object
        #     with the kwarg `to_continuous`"""
        # )

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
        points = np.tile(H.center, (3, 1))

        # logging
        logger.info("Starting")

        radius = H.inflation * H.radius / np.sqrt(self.search_space.size)
        points[1] += radius
        points[2] -= radius
        points = np.maximum(points, self.search_space._god.lo_bounds)
        points = np.minimum(points, self.search_space._god.up_bounds)

        self.pending_pb(3)

        logger.info(f"Evaluating points")
        if (
            isinstance(self.search_space, ContinuousSearchspace)
            or not H.to_convert
        ):
            scores = self.search_space.loss(points, algorithm="PHS")
        else:
            scores = self.search_space.loss(
                self.search_space.to_continuous.reverse(points, True),
                algorithm="PHS",
            )

        self.update_main_pb(
            3, explor=True, best=self.search_space.loss.new_best
        )
        self.meta_pb.update(3)

        logger.info("Ending")

        self.close_bar()

        logger.info("CGS ending")

        idx = np.argmin(scores)
        return points[idx], scores[idx]
