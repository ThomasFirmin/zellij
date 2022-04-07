from abc import abstractmethod
from zellij.core.loss_func import LossFunc

from scipy.stats import norm
import numpy as np


class Acquisition:
    def __init__(self, kernel):

        ##############
        # PARAMETERS #
        ##############

        self.kernel = kernel

        #############
        # VARIABLES #
        #############

        self.X = []
        self.Y = []
        self.Y_min = float("inf")
        self.surrogate = None

    def train_surrogate(self):
        self.surrogate = GPy.models.GPRegression(
            np.array(self.X), np.array(self.Y)
        )
        self.surrogate.optimize(optimizer="lbfgs")

    def add_points(self, X, Y):
        """add_points(X, Y)

        Add points to the surrogate.

        Parameters
        ----------
        X : list[{int, float, str}]
            A point to the continuous format.
        Y : type
            Value associated to `X`.

        """

        self.X += X[:]
        self.Y += Y[:]

        min_index = np.argmin(Y)

        if Y[min_index] < self.Y_min:
            self.Y_min = Y[min_index]

        self.train_surrogate()

    def run_surrogate(self, X):

        mean, cov = self.surrogate.predict(np.array(X))

        return mean, cov

    # Acquisition function
    def run_acqf(self, X):

        converted = self.search_space.convert_to_continuous(X)

        mean, cov = self.run_surrogate(converted)
        score = -self.af(self.Y_min, mean, cov).flatten()

        return score.tolist()


def LCB(ymin, mean, diagCov, beta=1):  # Grand beta permet d'explorer plus
    return -mean + beta * np.sqrt(diagCov)


def UCB(ymin, mean, diagCov, beta=1):  # Grand beta permet d'explorer plus
    return -mean - beta * np.sqrt(diagCov)


def PI(ymin, mean, diagCov):
    delta = ymin - mean
    normalized = delta / np.sqrt(diagCov)
    return norm.cdf(normalized)


def EI(ymin, mean, diagCov, psi=0.05):
    delta = ymin - mean + psi
    normalized = delta / np.sqrt(diagCov)
    return delta * norm.cdf(normalized) + np.sqrt(diagCov) * norm.pdf(
        normalized
    )


def acqui_fun(acqui_name):
    acqui_switcher = {"LCB": LCB, "UCB": UCB, "PI": PI, "EI": EI}
    try:
        return acqui_switcher[acqui_name]
    except KeyError:
        print("Error in acquisition function name")
        raise
