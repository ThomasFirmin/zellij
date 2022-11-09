# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-09T14:32:42+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from abc import ABC, abstractmethod


class Heuristic(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, search_space, indexes):
        pass


class Min(Heuristic):
    """Min

    Returns
    -------
    out : float
        Minimal score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

        Returns
        -------
        out : float
            Minimal score found.

        """
        return np.min(search_space.loss.all_scores[indexes])


class Median(Heuristic):
    """Median

    Returns
    -------
    out : float
        Median score found inside the fractal
    """

    def __call__(self, loss, indexes):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

        Returns
        -------
        out : float
            Median score found.

        """
        return np.median(search_space.loss.all_scores[indexes])


class Mean(Heuristic):
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
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

        Returns
        -------
        out : float
            Mean score found.

        """
        return np.mean(loss.all_scores[indexes])


class Std(Heuristic):
    """Std

    Returns
    -------
    out : float
        Std score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

        Returns
        -------
        out : float
            Std score found.

        """
        return np.std(search_space.loss.all_scores[indexes])


class Distance_to_the_best(Heuristic):
    """Distance_to_the_best

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

        Returns
        -------
        out : float
            Distance_to_the_best score found.

        """
        if search_space.to_convert:
            best_ind = search_space.convert.to_continuous(
                search_space.loss.best_sol, sub_values=True
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
            best_ind = search_space.loss.best_sol
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


class Distance_to_the_best_corrected(Heuristic):
    """Distance_to_the_best

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal
    """

    def __call__(self, search_space, indexes):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

        Returns
        -------
        out : float
            Distance_to_the_best score found.

        """
        if search_space.to_convert:
            best_ind = search_space.convert.to_continuous(
                search_space.loss.best_sol, sub_values=True
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
            best_ind = search_space.loss.best_sol
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


class Belief(Heuristic):
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
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        indexes : {int,slice}
            Indexes of the scores, saved in `loss.all_scores`
            used when computing heuristic.

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
