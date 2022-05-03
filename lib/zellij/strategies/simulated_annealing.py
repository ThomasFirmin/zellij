# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   ThomasFirmin
# @Last modified time: 2022-05-03T15:45:47+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.utils.cooling import Cooling
import zellij.utils.progress_bar as pb
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

import logging

logger = logging.getLogger("zellij.SA")


class Simulated_annealing(Metaheuristic):

    """Simulated_annealing

    Simulated_annealing (SA) is an exploitation strategy allowing to do hill climbing by starting from\
    an initial solution and iteratively moving to next one,\
     better than the previous one, or slightly worse to escape from local optima.

    It uses a :ref:`cooling` which partially drives the acceptance probability. This is the probability\
    to accept a worse solution according to the temperature, the best solution found so far and the actual solution.

    Attributes
    ----------

    loss_func : Loss
        :ref:`lf` to optimize. must be of type f(x)=y

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

    >>> from zellij.core.loss_func import Loss
    >>> from zellij.core.search_space import Searchspace
    >>> from zellij.strategies.simulated_annealing import Simulated_annealing
    >>> from zellij.strategies.utils.cooling import MulExponential
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> labels = ["a","b","c"]
    >>> types = ["R","R","R"]
    >>> values = [[-5, 5],[-5, 5],[-5, 5]]
    >>> sp = Searchspace(labels,types,values)
    >>> lf = Loss()(himmelblau)
    ...
    >>> cooling = MulExponential(0.85,100,2,3)
    >>> sa = Simulated_annealing(lf, sp, 100, cooling, 1)
    ...
    >>> point = sp.random_point()[0]
    >>> sa.run(point, lf([point])[0])
    >>> sa.show()

    .. image:: ../_static/sa_sp_ex.png
        :width: 924px
        :align: center
        :height: 487px
        :alt: alternate text
    .. image:: ../_static/sa_res_ex.png
        :width: 924px
        :align: center
        :height: 487px
        :alt: alternate text
    """

    # Initialize simulated annealing
    def __init__(
        self, loss_func, search_space, f_calls, cooling, max_iter, verbose=True
    ):

        """__init__(self,loss_func, search_space, f_calls, cooling, max_iter, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

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

        super().__init__(loss_func, search_space, f_calls, verbose)

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

        self.loss_func.file_created = False

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
            self.Y0 = self.loss_func(
                self.search_space.convert_to_continuous([self.X0], True)
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
        while T_actu and self.loss_func.calls < self.f_calls:
            iteration = 0
            while (
                iteration < self.max_iter
                and self.loss_func.calls < self.f_calls
            ):

                neighbors = self.search_space.get_neighbor(X, size=n_process)

                # Update progress bar
                self.pending_pb(len(neighbors))

                loss_values = self.loss_func(
                    neighbors,
                    temperature=self.record_temp[-1],
                    probability=self.record_proba[-1],
                )

                # Update progress bar
                self.update_main_pb(
                    len(neighbors), explor=False, best=self.loss_func.new_best
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
                    f"ITERATION: {self.loss_func.calls}/{self.f_calls}"
                )

                self.record_temp.append(T_actu)

                # Save file
                if self.loss_func.save:
                    if not self.file_created:
                        self.sa_save = os.path.join(
                            self.loss_func.outputs_path, "sa_best.csv"
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

        best_idx = np.argpartition(self.loss_func.all_scores, n_process)
        best = [self.loss_func.all_solutions[i] for i in best_idx[:n_process]]
        min = [self.loss_func.all_scores[i] for i in best_idx[:n_process]]

        self.cooling.reset()
        self.close_bar()
        return best, min

    def show(self, filepath="", save=False):

        """show(self, filename=None)

        Plots solutions and scores computed during the optimization

        Parameters
        ----------
        filepath : str, default=""
            If a filepath is given, the method will read files insidethe folder and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        data_all, all_scores = super().show(filepath, save)

        if filepath:

            path_sa = os.path.join(filepath, "outputs", "sa_best.csv")
            data_sa = pd.read_table(path_sa, sep=",", decimal=".")
            sa_scores = data_sa["loss"].to_numpy()

            temperature = data_sa["temperature"].to_numpy()
            probability = data_sa["probability"].to_numpy()

        else:
            data_sa = self.n_best
            sa_scores = np.array(self.n_scores)

            temperature = np.array(self.record_temp)
            probability = np.array(self.record_proba)

        argmin = np.argmin(sa_scores)

        f, (l1, l2) = plt.subplots(2, 2, figsize=(19.2, 14.4))

        ax1, ax2 = l1
        ax3, ax4 = l2

        ax1.plot(list(range(len(sa_scores))), sa_scores, "-")
        argmin = np.argmin(sa_scores)
        all_argmin = np.argmin(all_scores)

        ax1.scatter(
            argmin,
            sa_scores[argmin],
            color="red",
            label="Best score: " + str(sa_scores[argmin]),
        )
        ax1.scatter(
            0,
            sa_scores[0],
            color="green",
            label="Initial score: " + str(sa_scores[0]),
        )

        ax1.set_title("Simulated annealing")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score")
        ax1.legend(loc="upper right")

        if len(all_scores) < 100:
            s = 5
        else:
            s = 2500 / len(all_scores)

        ax2.scatter(list(range(len(all_scores))), all_scores, s=s)
        ax2.scatter(
            all_argmin,
            all_scores[all_argmin],
            color="red",
            label="Best score: " + str(sa_scores[argmin]),
        )
        ax2.scatter(
            0,
            sa_scores[0],
            color="green",
            label="Initial score: " + str(sa_scores[0]),
        )

        ax2.set_title("All evaluated solutions")
        ax2.set_xlabel("Solution ID")
        ax2.set_ylabel("Score")
        ax2.legend(loc="upper right")

        ax3.plot(list(range(len(sa_scores))), temperature, "-")
        argmin = np.argmin(sa_scores)
        ax3.scatter(
            argmin,
            temperature[argmin],
            color="red",
            label="Best score: " + str(temperature[argmin]),
        )

        ax3.set_title("Temperature decrease")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Temperature")
        ax3.legend(loc="upper right")

        if len(sa_scores) < 100:
            s = 5
        else:
            s = 2500 / len(all_scores)

        ax4.scatter(list(range(len(sa_scores))), probability, s=s)
        argmin = np.argmin(sa_scores)
        ax4.scatter(
            argmin,
            probability[argmin],
            color="red",
            label="Best score: " + str(probability[argmin]),
        )

        ax4.set_title("Escaping probability")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Probability")
        ax4.legend(loc="upper right")

        if save:
            save_path = os.path.join(
                self.loss_func.plots_path, f"sa_summary.png"
            )

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
