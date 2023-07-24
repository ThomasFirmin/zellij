# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-12T16:24:24+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import numpy as np
import abc
import copy

from collections import defaultdict
from itertools import groupby

import logging

logger = logging.getLogger("zellij.tree_search")


class Tree_search(object):

    """Tree_search

    Tree_search is an abstract class which determines how to explore a
    partition tree defined by :ref:`dba`.
    It is based on the OPEN/CLOSED lists algorithm.

    Attributes
    ----------

    open : list[Fractal]
        Open list containing not explored nodes from the partition tree.

    close : list[Fractal]
        Close list containing explored nodes from the partition tree.

    max_depth : int
        Maximum depth of the partition tree.

    Methods
    -------
    add(c)
        Add a node c to the fractal tree

    get_next()
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    """

    def __init__(self, open, max_depth, save_close=True):
        """__init__(self,open,max_depth)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing unexplored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        close : boolean, default=False
            If True, store expanded, explored and scored fractals within a :code:`close` list.

        """

        ##############
        # PARAMETERS #
        ##############

        if isinstance(open, list):
            self.open = open
        else:
            self.open = [open]

        self.save_close = save_close
        self.close = []

        assert max_depth > 0, f"Level must be > 0, got {max_depth}"
        self.max_depth = max_depth

    @abc.abstractmethod
    def add(self, c):
        """__init__(open,max_depth)

        Parameters
        ----------
        c : Fractal
            Add a new fractal to the tree

        """
        pass

    @abc.abstractmethod
    def get_next(self):
        """__init__(open, max_depth)

        Returns
        -------

        continue : boolean
            If True determine if the open list has been fully explored or not

        nodes : {list[Fractal], -1}
            if -1 no more nodes to explore, else return a list of the next node to explore
        """
        pass


class Breadth_first_search(Tree_search):
    """Breadth_first_search

    Breadth First Search algorithm (BFS).
    Fractal are first sorted by lowest level, then by lowest score.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Breadth_first_search, at each get_next, tries to return Q nodes.

    Methods
    -------

    add(c)
        Add a node c to the fractal tree

    get_next()
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Depth_first_search : Tree search Depth based startegy
    """

    def __init__(self, open, max_depth, save_close=True, Q=1):
        """__init__(open, max_depth, save_close=True, Q=1)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Breadth_first_search, at each get_next, tries to return Q nodes.

        """

        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.next_frontier + self.open,
                reverse=False,
                key=lambda x: (x.level, x.score),
            )[:]
            self.next_frontier = []

        if len(self.open) > 0:
            idx_min = np.min([len(self.open), self.Q])

            for _ in range(idx_min):
                self.close.append(self.open.pop(0))

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Depth_first_search(Tree_search):
    """Depth_first_search

    Depth First Search algorithm (DFS).
    Fractal are first sorted by highest level, then by lowest score.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
            maximum depth of the partition tree.

    save_close : boolean, default=True
        If True save expanded, explored and scored fractal within a :code:`close` list. tree.

    Q : int, default=1
        Q-Depth_first_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Breadth_first_search : Tree search Breadth based startegy
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth, save_close=True, Q=1):
        """__init__(open, max_depth, save_close=True,Q=1)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Breadth_first_search, at each get_next, tries to return Q nodes.

        """

        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.next_frontier + self.open,
                reverse=True,
                key=lambda x: (x.level, -x.score),
            )[:]
            self.next_frontier = []

        if len(self.open) > 0:
            idx_min = np.min([len(self.open), self.Q])

            for _ in range(idx_min):
                self.close.append(self.open.pop(0))

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Best_first_search(Tree_search):

    """Best_first_search

    Best First Search algorithm (BestFS).
    At each iteration, it selects the Q-best fractals.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing non-expanded nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Best_first_search, at each :code:`get_next`, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Beam_search : Memory efficient tree search algorithm based on BestFS
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth, save_close=True, Q=1, reverse=False):
        """__init__(self, open, max_depth, save_close=True, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.open
                + sorted(
                    self.next_frontier,
                    reverse=self.reverse,
                    key=lambda x: x.score,
                )[:],
                reverse=self.reverse,
                key=lambda x: x.score,
            )
            self.next_frontier = []

        if len(self.open) > 0:
            idx_min = np.min([len(self.open), self.Q])
            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.close.append(self.open.pop(0))

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Beam_search(Tree_search):

    """Beam_search

    Beam Search algorithm (BS). BS is an improvement of BestFS.
    It includes a beam length which allows to prune the worst nodes.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Beam_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(c)
        Add a node c to the fractal tree

    get_next()
        Get the next node to evaluate

    beam_length : int, default=10
        Determines the length of the open list for memory and prunning issues.

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Best_first_search : Tree search algorithm based on the best node from the open list
    Cyclic_best_first_search : Hybrid between DFS and BestFS, which can also perform pruning.
    """

    def __init__(
        self, open, max_depth, save_close=True, Q=1, reverse=False, beam_length=10
    ):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Beam_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        beam_length : int, default=10
            Determines the length of the open list for memory and prunning issues.

        """

        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.Q = Q
        self.beam_length = beam_length

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.next_frontier
                + sorted(self.open, reverse=self.reverse, key=lambda x: x.score),
                reverse=self.reverse,
                key=lambda x: x.score,
            )[: self.beam_length]
            self.next_frontier = []

        if len(self.open) > 0:
            idx_min = np.min([len(self.open), self.Q])
            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.close.append(self.open.pop(0))

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Epsilon_greedy_search(Tree_search):

    """Epsilon_greedy_search

    Epsilon Greedy Search (EGS).
    EGS is an improvement of BestFS. At each iteration, nodes are selected
    randomly or according to their best score.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Epsilon_greedy_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    epsilon : float, default=0.1
        Probability to select a random node from the open list.
        Determine how random the selection must be.
        The higher it is, the more exploration EGS does.

    Methods
    -------
    add(c)
        Add a node c to the fractal tree

    get_next()
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Best_first_search : Tree search algorithm based on the best node from the open list
    Diverse_best_first_search : Tree search strategy based on an adaptative probability to select random nodes.
    """

    def __init__(
        self, open, max_depth, save_close=True, Q=1, reverse=False, epsilon=0.1
    ):
        """__init__(self, open, max_depth, save_close=True, Q=1, reverse=False, epsilon=0.1):

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Epsilon_greedy_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        epsilon : float, default=0.1
            Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration EGS does.

        """

        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q
        self.reverse = reverse
        self.epsilon = epsilon

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.open
                + sorted(
                    self.next_frontier,
                    reverse=self.reverse,
                    key=lambda x: x.score,
                )[:],
                reverse=self.reverse,
                key=lambda x: x.score,
            )
            self.next_frontier = []

        if len(self.open) > 0:
            idx_min = np.min([len(self.open), self.Q])
            for _ in range(idx_min):
                if np.random.random() > self.epsilon:
                    self.close.append(self.open.pop(0))

                else:
                    if len(self.open) > 1:
                        idx = np.random.randint(1, len(self.open))
                        self.close.append(self.open.pop(idx))
                    else:
                        self.close.append(self.open.pop(0))

            return True, self.close[-idx_min:]

        else:
            return False, -1


##########
# DIRECT #
##########


class Potentially_Optimal_Rectangle(Tree_search):

    """Potentially_Optimal_Rectangle

    Potentially Optimal Rectangle algorithm (POR),
    is a the selection strategy comming from DIRECT.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Beam_search : Memory efficient tree search algorithm based on BestFS
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth=600, save_close=True, error=1e-4, maxdiv=3000):
        """__init__

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.
        maxdiv : int, default=3000
            Prunning parameter. Maximum number of fractals to store.
        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.error = error
        self.maxdiv = maxdiv

        #############
        # VARIABLES #
        #############
        self.max_i1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.min_i2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        self.best_score = np.min([c.score for c in self.open])

    def add(self, c):
        self.next_frontier.append(c)
        if c.score < self.best_score:
            self.best_score = c.score

    def get_next(self):
        if len(self.next_frontier) > 0:
            # sort potentially optimal rectangle by length (increasing)
            # then by score
            self.open += sorted(self.next_frontier, key=lambda x: (x.measure, x.score))
            # clip open list to maxdiv
            self.open = sorted(self.open, key=lambda x: (x.measure, x.score))[
                : self.maxdiv
            ]
            self.next_frontier = []

        if len(self.open):
            self.max_i1.fill(-float("inf"))
            self.min_i2.fill(float("inf"))

            groups = groupby(self.open, lambda x: x.measure)
            idx = self.optimal(groups)
            if idx:
                for i in reversed(idx):
                    self.close.append(self.open.pop(i))

                return True, self.close[-len(idx) :]
            else:
                self.close.append(self.open.pop(0))
                return True, self.close[-1:]
        else:
            return False, -1

    def optimal(self, groups):
        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # Potentially optimal index
        potoptidx = []

        group_size = 0
        for key, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            while (
                idx < len(subgroup)
                and np.abs(subgroup[idx].score - current_score) <= 1e-13
            ):
                current_score = subgroup[idx].score
                selected = subgroup[idx]
                current_idx = group_size + idx
                for jdx in range(current_idx + 1, len(self.open)):
                    c = self.open[jdx]

                    if c.measure < selected.measure:
                        denom = selected.measure - c.measure
                        num = selected.score - c.score
                        if denom != 0:
                            low_k = (num) / (denom)
                        else:
                            low_k = -float("inf")

                        if low_k > self.max_i1[current_idx]:
                            self.max_i1[current_idx] = low_k
                        elif low_k < self.min_i2[jdx]:
                            self.min_i2[jdx] = low_k

                    elif c.measure > selected.measure:
                        denom = c.measure - selected.measure
                        num = c.score - selected.score
                        if denom != 0:
                            up_k = (num) / (denom)
                        else:
                            up_k = float("inf")

                        if up_k < self.min_i2[current_idx]:
                            self.min_i2[current_idx] = up_k
                        elif up_k > self.max_i1[jdx]:
                            self.max_i1[jdx] = up_k

                if self.min_i2[current_idx] > 0 and (
                    self.max_i1[current_idx] <= self.min_i2[current_idx]
                ):
                    if self.best_score != 0:
                        num = self.best_score - selected.score
                        denum = np.abs(self.best_score)
                        scnd_part = selected.measure / denum * self.min_i2[current_idx]

                        if self.error <= num / denum + scnd_part:
                            potoptidx.append(current_idx)
                    else:
                        scnd_part = selected.measure * self.min_i2[current_idx]

                        if selected.score <= scnd_part:
                            potoptidx.append(current_idx)

                idx += 1
            group_size += len(subgroup)
        return potoptidx


class Locally_biased_POR(Tree_search):
    """Locally_biased_POR

    Locally_biased_POR, is a the selection strategy comming from Locally Biased DIRECT.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

    maxdiv : int, default=3000
        Prunning parameter. Maximum number of fractals to store.

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Beam_search : Memory efficient tree search algorithm based on BestFS
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth=600, save_close=True, error=1e-4, maxdiv=3000):
        """__init__(self, open, max_depth, Q=1, reverse=False, error=1e-4)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.error = error
        self.maxdiv = maxdiv

        #############
        # VARIABLES #
        #############
        self.max_i1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.min_i2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        min = [c.score for c in self.open]
        self.best_score = np.min(min)

    def add(self, c):
        self.next_frontier.append(c)
        if c.score < self.best_score:
            self.best_score = c.score

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open += sorted(
                self.next_frontier, key=lambda x: (-x.lengmeasureth, x.score)
            )
            self.open = sorted(self.open, key=lambda x: (-x.measure, x.score))[
                : self.maxdiv
            ]
            self.next_frontier = []

        if len(self.open):
            self.max_i1.fill(-float("inf"))
            self.min_i2.fill(float("inf"))

            groups = groupby(self.open, lambda x: x.measure)
            idx = self.optimal(groups)
            if idx:
                for i in reversed(idx):
                    self.close.append(self.open.pop(i))

                return True, self.close[-len(idx) :]
            else:
                self.close.append(self.open.pop(0))
                return True, self.close[-1:]
        else:
            return False, -1

    def optimal(self, groups):
        # Potentially optimal index
        potoptidx = defaultdict(None)

        group_size = 0
        for key, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            while (
                idx < len(subgroup)
                and np.abs(subgroup[idx].score - current_score) <= 1e-13
            ):
                current_score = subgroup[idx].score
                selected = subgroup[idx]
                current_idx = group_size + idx
                for jdx in range(current_idx + 1, len(self.open)):
                    c = self.open[jdx]

                    if c.measure < selected.measure:
                        denom = selected.measure - c.measure
                        num = selected.score - c.score
                        if denom != 0:
                            low_k = (num) / (denom)
                        else:
                            low_k = -float("inf")

                        if low_k > self.max_i1[current_idx]:
                            self.max_i1[current_idx] = low_k
                        elif low_k < self.min_i2[jdx]:
                            self.min_i2[jdx] = low_k

                    elif c.measure > selected.measure:
                        denom = c.measure - selected.measure
                        num = c.score - selected.score
                        if denom != 0:
                            up_k = (num) / (denom)
                        else:
                            up_k = float("inf")

                        if up_k < self.min_i2[current_idx]:
                            self.min_i2[current_idx] = up_k
                        elif up_k > self.max_i1[jdx]:
                            self.max_i1[jdx] = up_k

                if not (selected.measure in potoptidx):
                    if self.min_i2[current_idx] > 0 and (
                        self.max_i1[current_idx] <= self.min_i2[current_idx]
                    ):
                        if self.best_score != 0:
                            num = self.best_score - selected.score
                            denum = np.abs(self.best_score)
                            scnd_part = (
                                selected.measure / denum * self.min_i2[current_idx]
                            )

                            if self.error <= num / denum + scnd_part:
                                potoptidx[selected.measure] = current_idx
                        else:
                            scnd_part = selected.measure * self.min_i2[current_idx]

                            if selected.score <= scnd_part:
                                potoptidx[selected.measure] = current_idx

                idx += 1
            group_size += len(subgroup)
        return list(potoptidx.values())


class Adaptive_POR(Tree_search):

    """Adaptive_POR

    Adaptive_POR, is a the selection strategy
    comming from DIRECT-Restart.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

    maxdiv : int, default=3000
        Prunning parameter. Maximum number of fractals to store.

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Beam_search : Memory efficient tree search algorithm based on BestFS
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(
        self, open, max_depth=600, save_close=True, error=1e-2, maxdiv=3000, patience=5
    ):
        """__init__(self, open, max_depth, Q=1, reverse=False, error=1e-4)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.max_error = error
        self.maxdiv = maxdiv
        self.patience = patience
        #############
        # VARIABLES #
        #############
        self.max_i1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.min_i2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        min = [c.score for c in self.open]
        self.best_score = np.min(min)
        self.new_best_score = float("inf")

        self.stagnation = 0
        self.error = self.max_error

    def add(self, c):
        self.next_frontier.append(c)
        if c.score < self.new_best_score:
            self.new_best_score = c.score

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open += sorted(self.next_frontier, key=lambda x: (-x.measure, x.score))
            self.open = sorted(self.open, key=lambda x: (-x.measure, x.score))[
                : self.maxdiv
            ]
            self.next_frontier = []

        if self.best_score - self.new_best_score >= 1e-4 * np.abs(
            np.median(self.open[0].scores) - self.best_score
        ):
            self.best_score = self.new_best_score
            self.new_best_score = float("inf")
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.stagnation == self.patience:
            if self.error == 0.0:
                self.error = self.max_error
            else:
                self.error = 0.0

        if len(self.open) > 1:
            self.max_i1.fill(-float("inf"))
            self.min_i2.fill(float("inf"))

            groups = groupby(self.open, lambda x: x.measure)
            idx = self.optimal(groups)
            if idx:
                for i in reversed(idx):
                    self.close.append(self.open.pop(i))

                return True, self.close[-len(idx) :]

        elif len(self.open) == 1:
            self.close.append(self.open.pop(0))
            return True, self.close[-1:]

        else:
            return False, -1

    def optimal(self, groups):
        # Potentially optimal index
        potoptidx = []

        group_size = 0
        for key, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            while (
                idx < len(subgroup)
                and np.abs(subgroup[idx].score - current_score) <= 1e-13
            ):
                current_score = subgroup[idx].score
                selected = subgroup[idx]
                current_idx = group_size + idx
                for jdx in range(current_idx + 1, len(self.open)):
                    c = self.open[jdx]

                    if c.measure < selected.measure:
                        denom = selected.measure - c.measure
                        num = selected.score - c.score
                        if denom != 0:
                            low_k = (num) / (denom)
                        else:
                            low_k = -float("inf")

                        if low_k > self.max_i1[current_idx]:
                            self.max_i1[current_idx] = low_k
                        elif low_k < self.min_i2[jdx]:
                            self.min_i2[jdx] = low_k

                    elif c.measure > selected.measure:
                        denom = c.measure - selected.measure
                        num = c.score - selected.score
                        if denom != 0:
                            up_k = (num) / (denom)
                        else:
                            up_k = float("inf")

                        if up_k < self.min_i2[current_idx]:
                            self.min_i2[current_idx] = up_k
                        elif up_k > self.max_i1[jdx]:
                            self.max_i1[jdx] = up_k

                if self.min_i2[current_idx] > 0 and (
                    self.max_i1[current_idx] <= self.min_i2[current_idx]
                ):
                    if self.best_score != 0:
                        num = self.best_score - selected.score
                        denum = np.abs(self.best_score)
                        scnd_part = selected.measure / denum * self.min_i2[current_idx]

                        if self.error <= num / denum + scnd_part:
                            potoptidx.append(current_idx)
                    else:
                        scnd_part = selected.measure * self.min_i2[current_idx]

                        if selected.score <= scnd_part:
                            potoptidx.append(current_idx)

                idx += 1
            group_size += len(subgroup)
        return potoptidx


class Potentially_Optimal_Hypersphere(Tree_search):
    """Potentially_Optimal_Hypersphere

    Potentially Optimal Hypersphere algorithm (POH),
    is a the selection strategy comming from DIRECT adapted for Hyperspheres.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Best_first_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Beam_search : Memory efficient tree search algorithm based on BestFS
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth=600, save_close=True, error=1e-4, maxdiv=3000):
        """__init__(self, open, max_depth, Q=1, reverse=False, error=1e-4)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.error = error
        self.maxdiv = maxdiv

        #############
        # VARIABLES #
        #############
        self.max_i1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.min_i2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        min = [c.score for c in self.open]
        self.best_score = np.min(min)

    def add(self, c):
        self.next_frontier.append(c)
        if c.score < self.best_score:
            self.best_score = c.score

    def get_next(self):
        if len(self.next_frontier) > 0:
            # sort potentially optimal rectangle by radius (incresing)
            # then by score
            self.open += sorted(self.next_frontier, key=lambda x: (-x.radius, x.score))
            # clip open list to maxdiv
            self.open = sorted(self.open, key=lambda x: (-x.radius, x.score))[
                : self.maxdiv
            ]
            self.next_frontier = []

        if len(self.open):
            self.max_i1.fill(-float("inf"))
            self.min_i2.fill(float("inf"))

            groups = groupby(self.open, lambda x: x.radius)
            idx = self.optimal(groups)
            if idx:
                for i in reversed(idx):
                    self.close.append(self.open.pop(i))

                return True, self.close[-len(idx) :]
            else:
                self.close.append(self.open.pop(0))
                return True, self.close[-1:]
        else:
            return False, -1

    def optimal(self, groups):
        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # Potentially optimal index
        potoptidx = []

        group_size = 0
        for key, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            while (
                idx < len(subgroup)
                and np.abs(subgroup[idx].score - current_score) <= 1e-13
            ):
                current_score = subgroup[idx].score
                selected = subgroup[idx]
                current_idx = group_size + idx
                for jdx in range(current_idx + 1, len(self.open)):
                    c = self.open[jdx]

                    if c.radius < selected.radius:
                        denom = selected.radius - c.radius
                        num = selected.score - c.score
                        if denom != 0:
                            low_k = (num) / (denom)
                        else:
                            low_k = -float("inf")

                        if low_k > self.max_i1[current_idx]:
                            self.max_i1[current_idx] = low_k
                        elif low_k < self.min_i2[jdx]:
                            self.min_i2[jdx] = low_k

                    elif c.radius > selected.radius:
                        denom = c.radius - selected.radius
                        num = c.score - selected.score
                        if denom != 0:
                            up_k = (num) / (denom)
                        else:
                            up_k = float("inf")

                        if up_k < self.min_i2[current_idx]:
                            self.min_i2[current_idx] = up_k
                        elif up_k > self.max_i1[jdx]:
                            self.max_i1[jdx] = up_k

                if self.min_i2[current_idx] > 0 and (
                    self.max_i1[current_idx] <= self.min_i2[current_idx]
                ):
                    if self.best_score != 0:
                        num = self.best_score - selected.score
                        denum = np.abs(self.best_score)
                        scnd_part = selected.radius / denum * self.min_i2[current_idx]

                        if self.error <= num / denum + scnd_part:
                            potoptidx.append(current_idx)
                    else:
                        scnd_part = selected.radius * self.min_i2[current_idx]

                        if selected.score <= scnd_part:
                            potoptidx.append(current_idx)

                idx += 1
            group_size += len(subgroup)
        return potoptidx


#######
# SOO #
#######


class Soo_tree_search(Tree_search):
    """Soo_tree_search

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Depth_first_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Breadth_first_search : Tree search Breadth based startegy
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth, save_close=True, Q=1, reverse=False):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Depth_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            # sort leaves according to level and score ascending
            self.open = sorted(
                self.next_frontier
                + sorted(
                    self.open,
                    reverse=self.reverse,
                    key=lambda x: (x.level, x.score),
                ),
                reverse=self.reverse,
                key=lambda x: (x.level, x.score),
            )[:]
            self.next_frontier = []

        if len(self.open) > 0:
            current_level = self.open[0].level
            self.close.append(self.open.pop(0))
            idx_min = 1

            idx = 0
            size = len(self.open)

            # select the lowest score among all leaves at the current level
            while idx < size:
                node = self.open[idx]
                # If level change, then select the first node of this level.
                # (with the lowest score)
                if node.level != current_level:
                    current_level = node.level
                    self.close.append(self.open.pop(idx))
                    idx -= 1
                    size -= 1
                    idx_min += 1

                idx += 1

            return True, self.close[-idx_min:]

        else:
            return False, -1


#######
# FDA #
#######


class Move_up(Tree_search):
    """Move_up

    FDA tree search.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Depth_first_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(self,c)
        Add a node c to the fractal tree

    get_next(self)
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Breadth_first_search : Tree search Breadth based startegy
    Cyclic_best_first_search : Hybrid between DFS and BestFS
    """

    def __init__(self, open, max_depth, save_close=True, Q=1, reverse=False):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list. tree.

        Q : int, default=1
            Q-Depth_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(open, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.next_frontier
                + sorted(
                    self.open,
                    reverse=self.reverse,
                    key=lambda x: (-x.level, x.score),
                ),
                reverse=self.reverse,
                key=lambda x: (-x.level, x.score),
            )[:]
            self.next_frontier = []

        if len(self.open) > 0:
            for _ in range(self.Q):
                self.close.append(self.open.pop(0))

            return True, self.close[-self.Q :]

        else:
            return False, -1


class Cyclic_best_first_search(Tree_search):

    """Cyclic_best_first_search

    Cyclic Best First Search (CBFS). CBFS is an hybridation between DFS and
    BestFS. First, CBFS tries to reach a leaf of the fractal tree to quickly
    determine a base score. Then CBFS will do pruning according to this value,
    and it will decompose the problem into subproblems by inserting nodes into
    contours (collection of unexplored subproblems). At each iteration CBFS
    selects the best subproblem according to an heuristic value.
    Then the child will be inserted into their respective contours
    according to a labelling function.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the partition tree.

    max_depth : int
        maximum depth of the partition tree.

    Q : int, default=1
        Q-Cyclic_best_first_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    Methods
    -------
    add(c)
        Add a node c to the fractal tree

    get_next()
        Get the next node to evaluate

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm
    Tree_search : Base class
    Best_first_search : Tree search algorithm based on the best node from the open list
    Depth_first_search : Tree search Depth based startegy
    """

    def __init__(self, open, max_depth, Q=1, reverse=False):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the partition tree.

        max_depth : int
            maximum depth of the partition tree.

        Q : int, default=1
            Q-Cyclic_best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        """

        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

        self.L = [False] * (self.max_depth + 1)
        self.L[0] = True
        self.i = 0
        self.contour = [[] for i in range(self.max_depth + 1)]
        self.contour[0] = self.open

        self.best_scores = float("inf")
        self.first_complete = False

    def add(self, c):
        # Verify if a node must be pruned or not.
        # A node can be pruned only if at least one exploitation has been made
        if not self.first_complete:
            self.next_frontier.append(c)

            if c.level == self.max_depth - 1:
                self.first_complete = True
                self.best_score = c.score
        else:
            if c.score < self.best_score:
                self.best_score = c.score
                self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            modified_levels = []
            for h in self.next_frontier:
                self.contour[h.level].append(h)
                modified_levels.append(h.level)

                if not self.L[h.level]:
                    self.L[h.level] = True

            modified_levels = np.unique(modified_levels)
            for l in modified_levels:
                self.contour[l] = sorted(
                    self.contour[l], reverse=self.reverse, key=lambda x: x.score
                )

            self.next_frontier = []

        if np.any(self.L):
            search = True
            found = True

            l = 0
            i = -1

            while l < len(self.L) and search:
                if self.L[l]:
                    if found:
                        i = l
                        found = False

                    if l > self.i:
                        self.i = l
                        search = False

                l += 1

            if search and not found:
                self.i = i

            idx_min = np.min([len(self.contour[self.i]), self.Q])

            for _ in range(idx_min):
                self.close.append(self.contour[self.i].pop(0))

            if len(self.contour[self.i]) == 0:
                self.L[self.i] = False

            return True, self.close[-idx_min:]

        else:
            return False, -1
