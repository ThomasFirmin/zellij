# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-09T14:32:42+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger("zellij.scoring")


class Scoring(ABC):
    """Scoring

    Scoring is an abstract class defining the scoring method of DAC.
    It is similar to an acquisition function in BO. According to sampled points,
    it gives a score to a :ref:`frac`, which determines how promising it is.

    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, fractal):
        """__call__(fractal)

        Abstract method

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Minimal score found.

        """
        pass


class Min(Scoring):
    """Min

    Returns
    -------
    out : float
        Minimal score found inside the fractal
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Minimal score found.

        """
        if len(fractal.losses) > 0:
            return np.min(fractal.losses)
        else:
            return fractal.score


class Nothing(Scoring):
    """Nothing

    Does not modify current score.

    Returns
    -------
    out : float
        Return score of the current fractal.
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Score of the fractal

        """
        return fractal.score


class Median(Scoring):
    """Median

    Returns
    -------
    out : float
        Median score found inside the fractal
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Median score found.

        """
        if len(fractal.losses) > 0:
            return np.median(fractal.losses)
        else:
            return fractal.score


class Mean(Scoring):
    """Mean

    Returns
    -------
    out : float
        Mean score found inside the fractal
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Mean score found.

        """
        if len(fractal.losses) > 0:
            return np.mean(fractal.losses)
        else:
            return fractal.score


class Std(Scoring):
    """Std

    Standard deviation

    Returns
    -------
    out : float
        Std score found inside the fractal
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Standard deviation.

        """
        if len(fractal.losses) > 0:
            return np.std(fractal.losses)
        else:
            return fractal.score


class Distance_to_the_best(Scoring):
    """Distance_to_the_best

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Distance to the best solution found so far.

        """
        if len(fractal.losses) > 0:
            best_ind = fractal.loss.best_point
            distances = [
                fractal.distance(s, best_ind) + 1e-20 for s in fractal.solutions
            ]
            res = -np.max(np.array(fractal.losses) / distances)
            return res
        else:
            return fractal.score


class Distance_to_the_best_centered(Scoring):
    """Distance_to_the_best_centered

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal
    """

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Distance to the best solution found so far.

        """
        if len(fractal.losses) > 0:
            best_ind = fractal.loss.best_point
            distances = [
                fractal.distance(s, best_ind) + 1e-20 for s in fractal.solutions
            ]
            res = np.min(
                (np.array(fractal.losses) - fractal.loss.best_score) / distances
            )
            return res
        else:
            return fractal.score


class Belief(Scoring):
    """Belief

    Returns
    -------
    out : float
        Belief score found inside the fractal
    """

    def __init__(self, gamma=0.5):
        super(Belief, self).__init__()
        self.gamma = gamma

    def __call__(self, fractal):
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Belief from FRACTOP.

        """
        best_sc = fractal.loss.best_score

        if len(fractal.losses) > 0:
            if np.isfinite(fractal.score):
                father_score = fractal.score
            else:
                father_score = 0

            ratio = np.array(fractal.losses) / best_sc
            # Negate because minimization problem and maximize Belief
            return -(
                self.gamma * father_score
                + (1 - self.gamma) * np.mean(ratio * np.exp(1 - ratio))
            )
        else:
            return fractal.score
