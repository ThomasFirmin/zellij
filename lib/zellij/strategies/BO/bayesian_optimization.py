from zellij.core.metaheuristic import Metaheuristic

import numpy as np
import GPy
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os


class Bayesian_optimization(Metaheuristic):
    """Bayesian_optimization

    Bayesian optimization (BO) is a surrogate based optimization method which interpolates the actual loss function
    with a surrogate model, here it is a gaussian process. By sampling into this surrogate using a metaheuristic,
    BO determines promising points, which are worth to evaluate with the actual loss function. Once done, the
    gaussian process is updated using results obtained by evaluating those encouraging solutions with the loss function.

    Be carefull, BO struggles to scale up with high dimensional search space (d~20).

    Parameters
    ----------
    loss_func : Loss
        Loss function to optimize. must be of type f(x)=y
    search_space : Searchspace
        Search space object containing bounds of the search space
    f_calls : int
        Maximum number of loss_func calls
    iterations : int
        Number of BO iterations
    acquisition : Acquisition
        Acquisition function allows to
    kernel : type
        Description of parameter `kernel`.
    optimizer : type
        Description of parameter `optimizer`.
    verbose : type
        Description of parameter `verbose`.

    Attributes
    ----------
    X : type
        Description of attribute `X`.
    Y : type
        Description of attribute `Y`.
    Y_min : type
        Description of attribute `Y_min`.
    surrogate : type
        Description of attribute `surrogate`.
    optimizer
    acquisition
    kernel
    iterations

    """

    def __init__(self, loss_func, search_space, f_calls, iterations, acquisition, optimizer, verbose=False):

        super().__init__(loss_func, search_space, f_calls, verbose)

        ##############
        # PARAMETERS #
        ##############

        self.optimizer = optimizer
        self.acquisition = acquisition
        self.kernel = kernel

        self.iterations = iterations

        #############
        # VARIABLES #
        #############

    def optimize_acqf(self, n_process=1):

        points, scores = self.optimization_func.run(n_process=n_process)

        return points, scores

    def write_points(self, x, y, comments, scores, new=False):

        if new:
            self.f = open("results_bo.txt", "w")
            self.f.write(str(self.search_space.label)[1:-1].replace(" ", "").replace("'", "") + ",loss_value,aquisition,comments\n")
        else:
            self.f = open("results_bo.txt", "a")

        for i, j, k, c in zip(x, y, scores, comments):
            self.f.write(str(i)[1:-1].replace(" ", "") + "," + str(j) + "," + str(k) + "," + str(c) + "\n")
        self.f.close()

    def run(self, n_process=1, save=False):

        # Initialize comm Size architectures
        X = np.random.random((n_process, self.search_space.n_variables)).tolist()

        # Compute architectures
        converted = self.search_space.convert_to_continuous(X, reverse=True)
        Y, comments = self.loss_func(converted)
        Y = np.array(Y)

        scores = [-1] * len(Y)

        # Construct new GP by adding new computed points
        self.add_points(X, Y)

        # Master worker save points in a file
        if save:
            self.write_points(converted, Y, comments, scores, new=True)

        # BO starting
        i = 0
        calls = 0

        while i < self.iterations and calls < self.f_calls:

            # Save points in a file
            converted, scores = self.optimize_acqf(n_process)

            Y, comments = self.loss_func(converted)
            Y = np.array(Y)
            X = self.search_space.convert_to_continuous(converted)

            # Construct new GP by adding new computed points

            self.add_points(X, Y)

            i += 1
            calls += len(X)

            # Save points in a file
            if save:
                print("New archi: ", X)
                print("New value: ", Y)
                self.write_points(converted, Y, comments, scores, new=False)

        ind_min = np.argsort(self.Y)[0:n_process]
        min = np.array(self.Y)[ind_min].tolist()
        best = np.array(self.X)[ind_min].tolist()

        print(best)
        return self.search_space.convert_to_continuous(best[0], reverse=True)[0], min

    def show(self, filename=None):

        if filename == None:
            scores = np.array(self.Y)
        else:
            data = pd.read_table(filename, sep=",", decimal=".")
            scores = data["loss_value"].to_numpy()

        min = np.argmin(scores)
        plt.scatter(np.arange(len(scores)), scores, c=scores, cmap="plasma_r")
        plt.title("Scores evolution during bayesian optimization")
        plt.scatter(min, scores[min], color="red", label="Best score")
        plt.annotate(str(scores[min]), (min, scores[min]))
        plt.xlabel("Iterations")
        plt.ylabel("Scores")
        plt.legend()
        plt.show()

        if filename != None:
            self.search_space.show(data.iloc[:, 0 : self.search_space.n_variables], scores)
