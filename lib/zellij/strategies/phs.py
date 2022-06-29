# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-05-31T10:56:18+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import numpy as np
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.fractals import Hypersphere

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

    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous.
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous.
    center : float
        List of floats containing the coordinates of the search space center converted to continuous.
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous.

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
        assert hasattr(search_space, "to_continuous") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `to_continuous`"""
        )

    def run(self, H, n_process=1):
        """run(H=None, n_process=1)

        Parameters
        ----------
        H : Fractal, default=None
            When used by FDA, a fractal corresponding to the current subspace is given
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        assert isinstance(H, Hypersphere), logger.error(
            f"PHS should use Hyperspheres, got {H.__class__.__name__}"
        )

        current_idx = len(self.search_space.loss.all_solutions)
        points = np.zeros((3, self.search_space.size), dtype=float)

        self.build_bar(self.f_calls)

        # logging
        logger.info("Starting")

        current_idx = len(self.search_space.loss.all_solutions)
        points = np.empty((3, self.search_space.size), dtype=float)

        # logging
        logger.info("Starting")

        self.pending_pb(3)

        if self.search_space.loss.calls < self.f_calls:

            radius_part = (
                H.inflation * H.radius / np.sqrt(self.search_space.size)
            )

            points[0] = H.center
            points[1] = H.center + radius_part
            points[2] = H.center - radius_part

            points[points > 1] = 1
            points[points < 0] = 0

        logger.info(f"Evaluating points")
        if isinstance(self.search_space, ContinuousSearchspace):
            scores = self.search_space.loss(train_x.numpy(), algorithm="PHS")
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

        return self.search_space.loss.get_best(n_process)

    def show(self, filepath="", save=False):

        """show(filename="")

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        super().show(filepath, save)
