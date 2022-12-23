# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:35+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.tools.cooling import Cooling
import zellij.utils.progress_bar as pb
import numpy as np
import os

import logging

logger = logging.getLogger("zellij.SA")


class Simulated_annealing(Metaheuristic):

    """Simulated_annealing

    Simulated_annealing (SA) is a hill climbing exploitation algorithm.

    It uses a :ref:`cooling` which partially drives the acceptance probability.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing bounds of the search space.

    f_calls : int
        Maximum number of :ref:`lf` calls

    cooling : Cooling
        :ref:`cooling` used to determine the probability of acceptance.

    max_iter : int
        Maximum iterations of the inner loop.
        Determines how long the algorithm should sample neighbors of a solution,\
        before decreasing the temperature.

    save : boolean, optional
        if True save results into a file

    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij


    Examples
    --------

    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core import FloatVar, ArrayVar
    >>> from zellij.utils.neighborhoods import FloatInterval, ArrayInterval, Intervals
    >>> from zellij.strategies import Simulated_annealing
    >>> from zellij.strategies.tools import MulExponential
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(
    ...                           FloatVar("a",-5,5, neighbor=FloatInterval(0.5)),
    ...                           FloatVar("b",-5,5,neighbor=FloatInterval(0.5)),
    ...                           neighbor=ArrayInterval())
    ...                         ,lf, neighbor=Intervals())
    ...
    >>> cooling = MulExponential(0.85,100,2,3)
    >>> sa = Simulated_annealing(sp, 100, cooling, 1)
    ...
    >>> point = sp.random_point()
    >>> sa.run(point, lf([point])[0])
    """

    # Initialize simulated annealing
    def __init__(self, search_space, f_calls, cooling, max_iter, verbose=True):

        """__init__(search_space, f_calls, cooling, max_iter, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        cooling : Cooling
            Cooling schedule used to determine the probability of acceptance.

        max_iter : int
            Maximum iterations of the inner loop.
            Determines how long the algorithm should sample neighbors of a solution,\
            before decreasing the temperature.

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity


        """

        super().__init__(search_space, f_calls, verbose)

        # Max iteration after each temperature decrease
        self.max_iter = max_iter

        # Cooling schedule
        self.cooling = cooling

        self.n_scores = []
        self.n_best = []

        self.record_temp = [self.cooling.cool()]
        self.record_proba = [0]

        self.file_created = False

    # RUN SA
    def run(self, X0=None, Y0=None, H=None, n_process=1):

        """run(X0=None, Y0=None, H=None, n_process=1)

        Runs SA

        Parameters
        ----------
        X0 : list[float], optional
            Initial solution. If None, a Fractal must be given (H!=None)
        Y0 : {int, float}, optional
            Score of the initial solution
            Determine the starting point of the chaotic map.
        H : Fractal, optional
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

        self.search_space.loss.file_created = False

        if X0:
            self.X0 = X0
        elif H:
            self.X0 = H.center
        else:
            raise ValueError("No starting point given to Simulated Annealing")

        if Y0:
            self.Y0 = Y0
        else:
            logger.info("Simulated annealing evaluating initial solution")
            self.Y0 = self.search_space.loss(
                [self.X0],
                temperature=self.cooling.Tcurrent,
                probability=0.0,
                algorithm="SA",
            )[0]

        self.n_best.append(X0)
        self.n_scores.append(Y0)

        logger.info("Starting")
        logger.debug(f"Starting solution: {X0}, {Y0}")

        # Determine the number of iteration according to the function parameters
        logger.debug("Determining number of iterations")
        nb_iteration = self.cooling.iterations() * self.max_iter
        logger.info(f"Number of iterations: {nb_iteration}")

        self.build_bar(nb_iteration)

        # Initialize variable for simulated annealing
        # Best solution so far
        X = self.X0[:]

        # Best solution in the neighborhood
        X_p = X[:]

        # Current solution
        Y = X[:]

        # Initialize score
        cout_X = self.Y0
        cout_X_p = self.Y0

        T_actu = self.cooling.Tcurrent

        # Simulated annealing starting
        while T_actu and self.search_space.loss.calls < self.f_calls:
            iteration = 0
            while (
                iteration < self.max_iter
                and self.search_space.loss.calls < self.f_calls
            ):

                neighbors = self.search_space.neighbor(X, size=n_process)

                # Update progress bar
                self.pending_pb(len(neighbors))

                loss_values = self.search_space.loss(
                    neighbors,
                    temperature=self.record_temp[-1],
                    probability=self.record_proba[-1],
                )

                # Update progress bar
                self.update_main_pb(
                    len(neighbors),
                    explor=False,
                    best=self.search_space.loss.new_best,
                )

                index_min = np.argmin(loss_values)
                Y = neighbors[index_min][:]
                cout_Y = loss_values[index_min]

                # Compute previous cost minus new cost
                delta = cout_Y - cout_X

                logger.debug(f"New model score: {cout_Y}")
                logger.debug(f"Old model score: {cout_X}")
                logger.debug(f"Best model score: {cout_X_p}")

                # If a better model is found do...
                if delta < 0:
                    X = Y[:]
                    cout_X = cout_Y
                    if cout_Y < cout_X_p:

                        # Print if best model is found
                        logger.debug("Best model found: YES ")

                        X_p = X[:]
                        cout_X_p = cout_X

                    else:
                        logger.debug("Best model found: NO ")

                    self.record_proba.append(0)

                else:
                    logger.debug("Best model found: NO ")

                    p = np.random.uniform(0, 1)
                    emdst = np.exp(-delta / T_actu)

                    self.record_proba.append(emdst)

                    logger.debug(f"Escaping :  p<exp(-df/T) -->{p} < {emdst}")

                    if p <= emdst:
                        X = Y[:]
                        cout_X = cout_Y
                    else:
                        Y = X[:]

                iteration += 1
                self.meta_pb.update()

                logger.debug(
                    f"ITERATION: {self.search_space.loss.calls}/{self.f_calls}"
                )

                self.record_temp.append(T_actu)

                # Save file
                if self.search_space.loss.save:
                    if not self.file_created:
                        self.sa_save = os.path.join(
                            self.search_space.loss.outputs_path, "sa_best.csv"
                        )
                        with open(self.sa_save, "w") as f:
                            f.write(
                                ",".join(e for e in self.search_space.labels)
                                + ",loss,temperature,probability\n"
                            )
                            f.write(
                                ",".join(str(e) for e in self.X0)
                                + ","
                                + str(self.Y0)
                                + ","
                                + str(self.cooling.T0)
                                + ",0\n"
                            )
                            self.file_created = True

                    with open(self.sa_save, "a") as f:
                        f.write(
                            ",".join(str(e) for e in X)
                            + ","
                            + str(cout_X)
                            + ","
                            + str(self.record_temp[-1])
                            + ","
                            + str(self.record_proba[-1])
                            + "\n"
                        )

                self.n_scores.append(cout_X)
                self.n_best.append(X)

            T_actu = self.cooling.cool()

        # print the best solution from the simulated annealing
        logger.info(f"Best parameters: {X_p} score: {cout_X_p}")
        logger.info("Ending")

        best_idx = np.argpartition(self.search_space.loss.all_scores, n_process)
        best = [
            self.search_space.loss.all_solutions[i]
            for i in best_idx[:n_process]
        ]
        min = [
            self.search_space.loss.all_scores[i] for i in best_idx[:n_process]
        ]

        self.cooling.reset()
        self.close_bar()
        return best, min
