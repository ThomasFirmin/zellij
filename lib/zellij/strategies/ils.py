import numpy as np
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.fractals import Hypersphere

import logging

logger = logging.getLogger("zellij.ILS")

# Intensive local search
class ILS(Metaheuristic):

    """ILS

    Intensive local search is an exploitation algorithm comming from original FDA paper.
    It evaluate a point in each dimension arround an initial solution.
    Distance of the computed point to the initial one is decreasing according to a reduction rate.
    At each iteration the algorithm moves to the best solution found.

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
    red_rate : float
        determine the step reduction rate ate each iteration.
    precision : float
        dtermine the stopping criterion. When the step is lower than <precision> the algorithm stops.

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

    def __init__(self, loss_func, search_space, f_calls, red_rate=0.5, precision=1e-5, verbose=True):

        """__init__(loss_func, search_space, f_calls,save=False,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        red_rate : float, default=0.5
            determine the step reduction rate ate each iteration.

        precision : float, default=1e-5
            dtermine the stopping criterion. When the step is lower than <precision> the algorithm stops.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func, search_space, f_calls, verbose)

        self.red_rate = red_rate
        self.precision = precision

    def run(self, X0=None, Y0=None, H=None, n_process=1):

        """run(X0=None, Y0=None, H=None, n_process=1)

        Parameters
        ----------
        X0 : list[float], optional
            Initial solution. If None, a Fractal must be given (H!=None)
        Y0 : {int, float}, optional
            Score of the initial solution
            Determine the starting point of the chaotic map.
        H : Fractal, optional
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
            assert isinstance(H, Hypersphere), logger.error(f"ILS should use Hyperspheres, got {H.__class__.__name__}")

        # logging
        logger.info("Starting")

        self.build_bar(self.f_calls)

        scores = np.zeros(3, dtype=float)

        if X0:
            points = np.tile(X0, (3, 1))
            points[points > 1] = 1
            points[points < 0] = 0

        elif H:
            points = np.tile(H.center, (3, 1))
            points[points > 1] = 1
            points[points < 0] = 0
        else:
            raise ValueError("No starting point given to Simulated Annealing")

        if Y0:
            scores[0] = Y0
        else:
            scores[0] = self.loss_func(self.search_space.convert_to_continuous([points[0]], True, True))[0]

        step = H.radius

        while step > self.precision and self.loss_func.calls < self.f_calls:
            i = 0
            improvement = False
            # logging
            logger.info(f"ILS {step}>{self.precision}")

            while i < self.search_space.n_variables and self.loss_func.calls < self.f_calls:

                # logging
                logger.info(f"Evaluating dimension {i}")

                self.pending_pb(2)

                walk = points[0][i] + step
                points[1][i] = walk
                points[1][points[1] > 1] = 1
                points[1][points[1] < 0] = 0

                walk = points[0][i] - step
                points[2][i] = walk
                points[2][points[2] > 1] = 1
                points[2][points[2] < 0] = 0

                scores[1:] = self.loss_func(self.search_space.convert_to_continuous(points[1:], True, True))

                min_index = np.argmin(scores)

                if min_index != 0:
                    points = np.tile(points[min_index], (3, 1))
                    scores[0] = scores[min_index]
                    improvement = True

                self.update_main_pb(2, explor=False, best=self.loss_func.new_best)
                self.meta_pb.update(2)

                i += 1

            if not improvement:
                step = self.red_rate * step

        # logging
        logger.info("Ending")
        self.close_bar()

        return points[0], scores[0]

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
