from zellij.core.metaheuristic import Metaheuristic

import numpy as np
import copy

import logging

logger = logging.getLogger("zellij.sampling")


class DirectCenter(Metaheuristic):

    """Uniform

    Attributes
    ----------

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        loss_func,
        search_space,
        f_calls,
        verbose=True,
    ):

        """__init__(self,loss_func, search_space, f_calls, pop_size = 10, generation = 1000, verbose=True)

        Initialize Uniform sampling class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func, search_space, f_calls, verbose)


class Center(Metaheuristic):

    """Uniform

    Attributes
    ----------

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        loss_func,
        search_space,
        f_calls,
        verbose=True,
    ):

        """__init__(self,loss_func, search_space, f_calls, pop_size = 10, generation = 1000, verbose=True)

        Initialize Uniform sampling class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func, search_space, f_calls, verbose)

    # Run GA
    def run(self, H=None, n_process=1):

        """run(self, n_process = 1,save=False)

        Runs Uniform Sampling

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """
        assert H is not None, logger.error(f"Center stratgey must use a fractal center, got {H}")
        logger.info("Center strategy starting")

        self.build_bar(1)

        # Update progress bar
        self.pending_pb(1)
        point = self.search_space.convert_to_continuous([H.center], True, True)
        logger.info("Evaluating center")

        ys = self.loss_func(
            point,
            algorithm="Center",
        )

        self.meta_pb.update()

        # Update progress bar
        self.update_main_pb(1, explor=True, best=self.loss_func.new_best)

        self.close_bar()

        logger.info("Center strategy ending")

        return point, ys

    def show(self, filepath="", save=False):

        """show(self, filepath="")

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        all_data, all_scores = super().show(filepath, save)

    # Run GA
    def run(self, H=None, n_process=1):

        """run(self, n_process = 1,save=False)

        Runs Uniform Sampling

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """
        assert H is not None, logger.error(f"Center stratgey must use a fractal center, got {H}")
        logger.info("Center strategy starting")

        self.build_bar(1)

        # Update progress bar
        self.pending_pb(1)
        point = self.search_space.convert_to_continuous([H.center], True, True)
        logger.info("Evaluating center")

        ys = self.loss_func(
            point,
            algorithm="Center",
        )

        self.meta_pb.update()

        # Update progress bar
        self.update_main_pb(1, explor=True, best=self.loss_func.new_best)

        self.close_bar()

        logger.info("Center strategy ending")

        return point, ys

    def show(self, filepath="", save=False):

        """show(self, filepath="")

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        all_data, all_scores = super().show(filepath, save)


class Uniform(Metaheuristic):

    """Uniform

    Attributes
    ----------

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        loss_func,
        search_space,
        f_calls,
        verbose=True,
    ):

        """__init__(self,loss_func, search_space, f_calls, pop_size = 10, generation = 1000, verbose=True)

        Initialize Uniform sampling class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func, search_space, f_calls, verbose)

    # Run GA
    def run(self, n_process=1):

        """run(self, n_process = 1,save=False)

        Runs Uniform Sampling

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        logger.info("CGS starting")

        self.build_bar(1)

        logger.info("Uniform samling")

        points = np.random.uniform([0] * self.search_space.n_variables, [1] * self.search_space.n_variables, (self.f_calls, self.search_space.n_variables))

        # Update progress bar
        self.pending_pb(len(points))

        logger.info("Evaluating samples from Uniform")
        ys = self.loss_func(
            self.search_space.convert_to_continuous(points, True, True),
            algorithm="Uniform",
        )

        self.meta_pb.update()

        # Update progress bar
        self.update_main_pb(len(points), explor=True, best=self.loss_func.new_best)

        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = ys[idx]

        self.close_bar()

        logger.info("Uniform sampling ending")

        return best_sol, best_scores

    def show(self, filepath="", save=False):

        """show(self, filepath="")

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        all_data, all_scores = super().show(filepath, save)


class BoxMuller(Metaheuristic):
    def Muller(c, r, d, n):
        u = np.random.normal(0, 1, (n, d))
        norm = np.linalg.norm(u, axis=1)
        radii = np.random.random(n) ** (1 / d)
        x = radii[:, None] * u / norm[:, None] * 2
        return x
