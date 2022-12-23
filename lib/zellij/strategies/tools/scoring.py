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
    def __call__(self, search_space, indexes):
        pass


class Min(Scoring):
    """Min

    Returns
    -------
    out : float
        Minimal score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """__call__(loss, indexes)

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in :code:`loss.all_scores`
            used when computing score.

        Returns
        -------
        out : float
            Minimal score found.

        """
        return np.min(search_space.loss.all_scores[indexes])


class Median(Scoring):
    """Median

    Returns
    -------
    out : float
        Median score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """__call__(loss, indexes)

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in :code:`loss.all_scores`
            used when computing score.

        Returns
        -------
        out : float
            Median score found.

        """
        return np.median(search_space.loss.all_scores[indexes])


class Mean(Scoring):
    """Mean

    Returns
    -------
    out : float
        Mean score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in :code:`loss.all_scores`
            used when computing score.

        Returns
        -------
        out : float
            Mean score found.

        """
        return np.mean(search_space.loss.all_scores[indexes])


class Std(Scoring):
    """Std

    Standard deviation

    Returns
    -------
    out : float
        Std score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """__call__(loss, indexes)

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in :code:`loss.all_scores`
            used when computing score.

        Returns
        -------
        out : float
            Std score found.

        """
        return np.std(search_space.loss.all_scores[indexes])


class Distance_to_the_best(Scoring):
    """Distance_to_the_best

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        def __call__(self, search_space, indexes):
            """__call__(loss, indexes)

            Parameters
            ----------
            search_space : Searchspace
                Search space object containing bounds of the search space.
            indexes : {int,slice}
                Indexes of the scores, saved in :code:`loss.all_scores`
                used when computing score.

            Returns
            -------
            out : float
                Distance_to_the_best score found.

            """

        if search_space.to_convert:
            best_ind = search_space.convert.to_continuous(
                search_space.loss.best_point, sub_values=True
            )

            return -np.max(
                np.array(search_space.loss.all_scores[indexes])
                / (
                    np.linalg.norm(
                        np.array(
                            search_space.convert.to_continuous(
                                search_space.loss.all_solutions[indexes],
                                sub_values=True,
                            )
                        )
                        - np.array(best_ind),
                        axis=1,
                    )
                    + 1e-20
                )
            )
        else:
            best_ind = search_space.loss.best_point
            res = -np.max(
                np.array(search_space.loss.all_scores[indexes])
                / (
                    np.linalg.norm(
                        np.array(search_space.loss.all_solutions[indexes])
                        - np.array(best_ind),
                        axis=1,
                    )
                    + 1e-20
                )
            )
            return res


class Distance_to_the_best_corrected(Scoring):
    """Distance_to_the_best_corrected

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """__call__(loss, indexes)

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in :code:`loss.all_scores`
            used when computing score.

        Returns
        -------
        out : float
            Distance_to_the_best score found.

        """
        if search_space.to_convert:
            best_ind = search_space.convert.to_continuous(
                search_space.loss.best_point, sub_values=True
            )

            return np.min(
                (
                    np.array(search_space.loss.all_scores[indexes])
                    - search_space.loss.best_score
                )
                / (
                    np.linalg.norm(
                        np.array(
                            search_space.convert.to_continuous(
                                search_space.loss.all_solutions[indexes],
                                sub_values=True,
                            )
                        )
                        - np.array(best_ind),
                        axis=1,
                    )
                    + 1e-20
                )
            )
        else:
            best_ind = search_space.loss.best_point
            res = np.min(
                (
                    np.array(search_space.loss.all_scores[indexes])
                    - search_space.loss.best_score
                )
                / (
                    np.linalg.norm(
                        np.array(search_space.loss.all_solutions[indexes])
                        - np.array(best_ind),
                        axis=1,
                    )
                    + 1e-20
                )
            )
            return res


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

    def __call__(self, search_space, indexes):
        """__call__(loss, indexes)


        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in :code:`loss.all_scores`
            used when computing score.

        Returns
        -------
        out : float
            Belief score found.

        """
        best_sc = search_space.loss.best_score

        if type(search_space.father.father) == str:
            search_space.father.score = 0

        ratio = np.array(search_space.loss.all_scores[indexes]) / best_sc
        # Negate because minimization problem and maximize Belief
        return -(
            self.gamma * search_space.father.score
            + (1 - self.gamma) * np.mean(ratio * np.exp(1 - ratio))
        )
