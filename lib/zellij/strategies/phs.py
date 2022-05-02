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

    def __init__(self, loss_func, search_space, f_calls, verbose=True):

        """__init__(self, loss_func, search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func, search_space, f_calls, verbose)

    def run(self, H=None, n_process=1):
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

        if H:
            assert isinstance(H, Hypersphere), logger.error(f"PHS should use Hyperspheres, got {H.__class__.__name__}")

        current_idx = len(self.loss_func.all_solutions)
        points = np.zeros((3, self.search_space.n_variables), dtype=float)

        self.build_bar(self.f_calls)

        # logging
        logger.info("Starting")

        current_idx = len(self.loss_func.all_solutions)
        points = np.empty((3, self.search_space.n_variables), dtype=float)

        # logging
        logger.info("Starting")

        self.pending_pb(3)

        if self.loss_func.calls < self.f_calls:

            radius_part = H.inflation * H.radius / np.sqrt(self.search_space.n_variables)

            points[0] = H.center
            points[1] = H.center + radius_part
            points[2] = H.center - radius_part

            points[points > 1] = 1
            points[points < 0] = 0

        logger.info(f"Evaluating points")
        scores = self.loss_func(self.search_space.convert_to_continuous(points, True, True))

        self.update_main_pb(3, explor=True, best=self.loss_func.new_best)
        self.meta_pb.update(3)

        logger.info("Ending")

        scores = np.array(scores)
        idx = np.array(np.argsort(scores))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = scores[idx]

        self.close_bar()

        logger.info("CGS ending")

        return best_sol, best_scores

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
