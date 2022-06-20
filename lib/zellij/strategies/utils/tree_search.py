# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-20T12:55:44+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import numpy as np
import abc
import copy


class Tree_search(object):

    """Tree_search

    Tree_search is an abstract class which determines how to explore the fractal rooted tree, builded during Fractal Decomposition.
    It is based on the OPEN/CLOSED lists algorithm.

    Attributes
    ----------

    open : list[Fractal]
        Open list containing not explored nodes from the fractal rooted tree.

    close : list[Fractal]
        Close list containing explored nodes from the fractal rooted tree.

    max_depth : int
        Maximum depth of the fractal rooted tree.

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

    def __init__(self, open, max_depth):

        """__init__(self,open,max_depth)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        """

        ##############
        # PARAMETERS #
        ##############

        if isinstance(open, list):
            self.open = open
        else:
            self.open = [open]

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

    Breadth First Search algorithm (BFS), computationally inefficient with fractal decomposition algorithm, because it is a greedy algorithm doing only exploration of the fractal tree,\
     exploring entirely each fractal level.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

    Q : int, default=1
        Q-Breadth_first_search, at each get_next, tries to return Q nodes.

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
    Depth_first_search : Tree search Depth based startegy
    """

    def __init__(self, open, max_depth, Q=1, reverse=False):

        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Breadth_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        """

        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q
        self.reverse = reverse

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
                reverse=self.reverse,
                key=lambda x: x.level,
            )[:]
            self.next_frontier = []

        if len(self.open) > 0:

            idx_min = np.min([len(self.open), self.Q])

            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.open.pop(0)

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Depth_first_search(Tree_search):
    """Depth_first_search

    Depth First Search algorithm (DFS), computationally inefficient with fractal decomposition algorithm, because it is a greedy algorithm doing only exploitation of the fractal tree,\
    by trying to go has deep as possible.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, max_depth, Q=1, reverse=False):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Depth_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

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

    def add(self, c):

        self.next_frontier.append(c)

    def get_next(self):

        if len(self.next_frontier) > 0:
            self.open = (
                sorted(
                    self.next_frontier,
                    reverse=self.reverse,
                    key=lambda x: x.score,
                )[:]
                + self.open
            )
            self.next_frontier = []

        if len(self.open) > 0:

            idx_min = np.min([len(self.open), self.Q])

            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.open.pop(0)

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Best_first_search(Tree_search):

    """Best_first_search

    Best First Search algorithm (BestFS), BestFS is better than BFS and DFS, because it tries to explore and exploit only the best current node in the tree.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, max_depth, Q=1, reverse=False):
        """__init__(self, open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

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
                self.open.pop(0)

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Beam_search(Tree_search):

    """Beam_search

    Beam Search algorithm (BS). BS is an improvement of BestFS. It includes a beam length (resp. open list length),\
    which allows to prune the worst nodes and only keep in memory a certain number of the best found nodes.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, max_depth, Q=1, reverse=False, beam_length=10):

        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Beam_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        beam_length : int, default=10
            Determines the length of the open list for memory and prunning issues.

        """

        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.open = open
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
                + sorted(
                    self.open, reverse=self.reverse, key=lambda x: x.score
                ),
                reverse=self.reverse,
                key=lambda x: x.score,
            )[: self.beam_length]
            self.next_frontier = []

        if len(self.open) > 0:

            idx_min = np.min([len(self.open), self.Q])
            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.open.pop(0)

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Diverse_best_first_search(Tree_search):

    """Diverse_best_first_search

    Diverse Best First Search (DBFS). DBFS is an improvement of BestFS. When a node is badly evaluated, this one has no more chance to be explored.\
    DBFS tries to tackle this problem by randomly selecting nodes according to a probability computed with its heuristic value (score) and its parents scores,\
    or according to a probability P.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

    Q : int, default=1
        Q-Diverse_best_first_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    P : float, default=0.1
        Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration DBFS does.

    T : float, default=0.5
        Influences the probability of a node to be selected according to its score compared to the best score from the open list.

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
    Epsilon_greedy_search : Based on BestFS, allows to randomly select a node.
    """

    def __init__(self, open, max_depth, Q=1, reverse=False, P=0.1, T=0.5):

        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Diverse_best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        P : float, default=0.1
            Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration DBFS does.

        T : float, default=0.5
            Influences the probability of a node to be selected according to its score compared to the best score from the open list.
        """

        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

        self.open = open
        self.reverse = reverse
        self.Q = Q
        self.P = P
        self.T = T

        #############
        # VARIABLES #
        #############
        self.next_frontier = []

        for i in self.open:
            i.g_value = i.min_score

    def add(self, c):

        c.g_value = c.min_score
        start = c.father

        while type(start.father) != str:
            c.g_value += start.father.min_score
            start = start.father

        self.next_frontier.append(c)

    def fetch_node(self):

        if len(self.open) > 1:
            p_total = 0

            h_values = [i.min_score for i in self.open]
            g_values = [i.g_value for i in self.open]

            p = []

            combination = []

            hmin, hmax = np.min(h_values), np.max(h_values)
            gmin, gmax = np.min(g_values), np.max(g_values)

            if np.random.random() < self.P:
                G = np.random.choice(g_values)
            else:
                G = gmax

            for h, g in zip(h_values, g_values):

                if g > G:
                    p.append(0)
                else:
                    p.append(self.T ** (h - hmin))

                p_total += p[-1]

            idx = np.random.choice(len(self.open), p=p / p_total)

            return idx

        else:
            return 0

    def get_next(self):

        if len(self.next_frontier) > 0:
            self.open = sorted(
                self.next_frontier
                + sorted(
                    self.open, reverse=self.reverse, key=lambda x: x.score
                ),
                reverse=self.reverse,
                key=lambda x: x.score,
            )
            self.next_frontier = []

        if len(self.open) > 0:

            idx = self.fetch_node()
            self.close += [self.open[idx]]
            self.open.pop(idx)

            return True, [self.close[-1]]

        else:
            return False, -1


class Cyclic_best_first_search(Tree_search):

    """Cyclic_best_first_search

    Cyclic Best First Search (CBFS). CBFS is an hybrid of DFS and BestFS. First, CBFS tries to reach a leaf of the fractal tree to quickly determine a base score.
    Then CBFS will do pruning according to this value, and it will decompose the problem into subproblems by inserting nodes into contours (collection of unexplored subproblems).
    At each iteration CBFS select the best subproblem according to an heuristic value. Then the child subproblems will be inserted into their respective contours, according to a labelling function.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

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
        self.contour[0] = open

        self.best_scores = float("inf")
        self.first_complete = False

    def add(self, c):

        # Verify if a node must be pruned or not.
        # A node can be pruned only if at least one exploitation has been made
        if not self.first_complete:
            self.next_frontier.append(c)

            if c.level == self.max_depth:
                self.first_complete = True
                self.best_score = c.min_score
        else:
            if c.min_score < self.best_score:
                self.best_score = c.min_score
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

            self.close += self.contour[self.i][0:idx_min]

            for _ in range(idx_min):
                self.contour[self.i].pop(0)

            if len(self.contour[self.i]) == 0:
                self.L[self.i] = False

            return True, self.close[-idx_min:]

        else:
            return False, -1


class Epsilon_greedy_search(Tree_search):

    """Epsilon_greedy_search

    Epsilon Greedy Search (EGS). EGS is an improvement of BestFS. At each iteration nodes are selected randomly or according to their best score.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

    Q : int, default=1
        Q-Epsilon_greedy_search, at each get_next, tries to return Q nodes.

    reverse : boolean, default=False
        if False do a descending sort the open list, else do an ascending sort

    epsilon : float, default=0.1
        Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration EGS does.

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

    def __init__(self, open, max_depth, reverse=False, epsilon=0.1):

        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Epsilon_greedy_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        epsilon : float, default=0.1
            Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration EGS does.

        """

        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

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

            if np.random.random() > self.epsilon:

                self.close += [self.open[0]]
                self.open.pop(0)

            else:

                if len(self.open) > 1:
                    idx = np.random.randint(1, len(self.open))
                    self.close += [self.open[idx]]
                    self.open.pop(idx)
                else:
                    self.close += [self.open[0]]
                    self.open.pop(0)

            return True, [self.close[-1]]

        else:
            return False, -1


# class Potentially_Optimal_Rectangle(Tree_search):
#
#     """Potentially_Optimal_Rectangle
#
#     Potentially Optimal Rectangle algorithm (POR), is a the selection strategy comming from DIRECT.
#
#     Attributes
#     ----------
#
#     open : list[Fractal]
#         Initial Open list containing not explored nodes from the fractal rooted tree.
#
#     max_depth : int
#         maximum depth of the fractal rooted tree.
#
#     Q : int, default=1
#         Q-Best_first_search, at each get_next, tries to return Q nodes.
#
#     reverse : boolean, default=False
#         if False do a descending sort the open list, else do an ascending sort
#
#     Methods
#     -------
#     add(self,c)
#         Add a node c to the fractal tree
#
#     get_next(self)
#         Get the next node to evaluate
#
#     See Also
#     --------
#     Fractal : Abstract class defining what a fractal is.
#     FDA : Fractal Decomposition Algorithm
#     Tree_search : Base class
#     Beam_search : Memory efficient tree search algorithm based on BestFS
#     Cyclic_best_first_search : Hybrid between DFS and BestFS
#     """
#
#     def __init__(self, open, max_depth, error=1e-4):
#         """__init__(self, open, max_depth, Q=1, reverse=False, error=1e-4)
#
#         Parameters
#         ----------
#         open : list[Fractal]
#             Initial Open list containing not explored nodes from the fractal rooted tree.
#
#         max_depth : int
#             maximum depth of the fractal rooted tree.
#
#         Q : int, default=1
#             Q-Best_first_search, at each get_next, tries to return Q nodes.
#
#         reverse : boolean, default=False
#             if False do a descending sort the open list, else do an ascending sort
#
#         error : float, default=1e-4
#             Small value which determines when an evaluation should be considered as good as the best solution found so far.
#
#         """
#         super().__init__(open, max_depth)
#
#         ##############
#         # PARAMETERS #
#         ##############
#
#         self.error = error
#
#         #############
#         # VARIABLES #
#         #############
#
#         self.next_frontier = []
#
#         self.best_score = float("inf")
#
#         self.build = np.vectorize(
#             lambda x: (x.score, np.linalg.norm(x.center - x.lo_bounds))
#         )
#
#     def add(self, c):
#
#         self.next_frontier.append(c)
#         self.best_score = c.loss.best_score
#
#     def get_next(self):
#
#         if len(self.next_frontier) > 0:
#             idx = self.optimal()
#
#             for i, j in enumerate(idx):
#                 self.open.append(self.next_frontier.pop(j - i))
#
#         if len(self.open) > 0:
#             idx_min = len(self.open)
#
#             for _ in range(idx_min):
#                 self.close.append(self.open.pop(0))
#
#             return True, self.close[-idx_min:]
#
#         else:
#             return False, -1
#
#     def optimal(self):
#
#         res = []
#
#         a, d = self.build(self.next_frontier)
#
#         p1 = a.reshape((len(a), 1)) - a
#         p2 = d.reshape((len(d), 1)) - d
#         K = np.divide(
#             p1, p2, out=np.full((len(p1), len(p2)), float("inf")), where=p2 != 0
#         )
#
#         K2 = (a - self.best_score + self.error * np.abs(self.best_score)) / d
#
#         for i in range(len(a)):
#             U = K[i][p2[i] < 0]
#             if len(U) == 0:
#                 res.append(i)
#             else:
#                 minU = np.nanmin(U)
#                 if minU > 0:
#                     L = K[i][p2[i] >= 0]
#                     maxL = np.maximum(np.nanmax(L), K2[i])
#                     if maxL > 0 and maxL <= minU:
#                         res.append(i)
#
#         return res


class Potentially_Optimal_Rectangle(Tree_search):

    """Potentially_Optimal_Rectangle

    Potentially Optimal Rectangle algorithm (POR), is a the selection strategy comming from DIRECT.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, measure, max_depth=600, error=1e-4, maxdiv=3000):
        """__init__(self, open, max_depth, Q=1, reverse=False, error=1e-4)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        error : float, default=1e-4
            Small value which determines when an evaluation should be considered as good as the best solution found so far.

        """
        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

        self.error = error
        self.maxdiv = maxdiv

        #############
        # VARIABLES #
        #############
        self.maxi1 = [-float("inf")] * self.maxdiv
        self.mini2 = [float("inf")] * self.maxdiv
        self.initialized = [False] * self.maxdiv
        self.i1 = [[]] * self.maxdiv
        self.i2 = [[]] * self.maxdiv
        self.i3 = [[]] * self.maxdiv

        self.next_frontier = []
        min = [c.score for c in self.open]
        self.best_score = np.min(min)

    def add(self, c):

        self.next_frontier.append(c)
        self.best_score = c.loss.best_score

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.initialized += [False] * self.maxdiv
            self.open += self.next_frontier

            self.open, self.initialized = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(self.open, self.initialized),
                        key=lambda x: x[0].score,
                    )
                )
            )
            self.open = self.open[: self.maxdiv]
            self.initialized = self.initialized[: self.maxdiv]

            self.next_frontier = []

        if len(self.open) > 0:
            idx = self.optimal()
            print(len(self.open), len(idx))
            for i, j in enumerate(idx):
                self.close.append(self.open.pop(j - i))

            idx_min = len(idx) if len(idx) < self.maxdiv else self.maxdiv

            return True, self.close[-idx_min:]

        else:
            return False, -1

    def optimal(self):
        # Potentially optimal index
        potoptidx = []

        for idx in range(len(self.open)):
            selected = self.open[idx]
            if not self.initialized[idx]:
                self.maxi1[idx], self.mini2[idx] = -float("inf"), float("inf")
                self.initialized[idx] = True
                self.i1[idx], self.i2[idx], self.i3[idx] = [], [], []

                for jdx in range(idx + 1, len(self.open)):
                    c = self.open[jdx]

                    if c.length < selected.length:
                        self.i1[idx].append(c)
                        self.i2[jdx].append(selected)

                        denom = selected.length - c.length
                        num = selected.score - c.score
                        if denom != 0:
                            low_k = (num) / (denom)
                        else:
                            low_k = -float("inf")

                        if low_k > self.maxi1[idx]:
                            self.maxi1[idx] = low_k
                        elif low_k < self.mini2[jdx]:
                            self.mini2[jdx] = low_k

                    elif c.length > selected.length:

                        self.i2[idx].append(c)
                        self.i1[jdx].append(selected)

                        denom = c.length - selected.length
                        num = c.score - selected.score
                        if denom != 0:
                            up_k = (num) / (denom)
                        else:
                            up_k = float("inf")

                        if up_k < self.mini2[idx]:
                            self.mini2[idx] = up_k
                        elif up_k > self.maxi1[jdx]:
                            self.maxi1[jdx] = up_k
                    else:
                        # self.i3[idx].append(c)
                        # self.i3[jdx].append(selected)
                        pass

            if self.mini2[idx] > 0 and (self.maxi1[idx] <= self.mini2[idx]):

                if self.best_score != 0:

                    num = self.best_score - selected.score
                    denum = np.abs(self.best_score)
                    scnd_part = selected.length / denum * self.mini2[idx]

                    if self.error <= num / denum + scnd_part:
                        potoptidx.append(idx)
                else:
                    scnd_part = selected.length * self.mini2[idx]

                    if selected.score <= scnd_part:
                        potoptidx.append(idx)

        return potoptidx


class Locally_biased_POR(Tree_search):

    """Locally biased Potentially_Optimal_Rectangle

    Potentially Optimal Rectangle algorithm (POR), is a the selection strategy comming from DIRECT.

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, measure, max_depth=600, error=1e-4, maxdiv=3000):
        """__init__(self, open, max_depth, Q=1, reverse=False, error=1e-4)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Best_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

        error : float, default=1e-4
            Small value which determines when an evaluation should be considered as good as the best solution found so far.

        """
        super().__init__(open, max_depth)

        ##############
        # PARAMETERS #
        ##############

        self.error = error
        self.maxdiv = maxdiv

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
                    key=lambda x: (x.level, x.score, -x.length),
                ),
                key=lambda x: (x.level, x.score, -x.length),
            )[:]
            self.next_frontier = []

        if len(self.open) > 0:
            idx = self.optimal()
            print(len(self.open), len(idx))
            for i, j in enumerate(idx):
                self.close.append(self.open.pop(j - i))

            idx_min = len(idx) if len(idx) < self.maxdiv else self.maxdiv

            return True, self.close[-idx_min:]

        else:
            return False, -1

    def optimal(self):
        # Potentially optimal index
        potoptidx = []

        for idx in range(len(self.open)):
            selected = self.open[idx]
            if not self.initialized[idx]:
                self.maxi1[idx], self.mini2[idx] = -float("inf"), float("inf")
                self.initialized[idx] = True
                self.i1[idx], self.i2[idx], self.i3[idx] = [], [], []

                for jdx in range(idx + 1, len(self.open)):
                    c = self.open[jdx]

                    if c.length < selected.length:
                        self.i1[idx].append(c)
                        self.i2[jdx].append(selected)

                        denom = selected.length - c.length
                        num = selected.score - c.score
                        if denom != 0:
                            low_k = (num) / (denom)
                        else:
                            low_k = -float("inf")

                        if low_k > self.maxi1[idx]:
                            self.maxi1[idx] = low_k
                        elif low_k < self.mini2[jdx]:
                            self.mini2[jdx] = low_k

                    elif c.length > selected.length:

                        self.i2[idx].append(c)
                        self.i1[jdx].append(selected)

                        denom = c.length - selected.length
                        num = c.score - selected.score
                        if denom != 0:
                            up_k = (num) / (denom)
                        else:
                            up_k = float("inf")

                        if up_k < self.mini2[idx]:
                            self.mini2[idx] = up_k
                        elif up_k > self.maxi1[jdx]:
                            self.maxi1[jdx] = up_k
                    else:
                        # self.i3[idx].append(c)
                        # self.i3[jdx].append(selected)
                        pass

            if self.mini2[idx] > 0 and (self.maxi1[idx] <= self.mini2[idx]):

                if self.best_score != 0:

                    num = self.best_score - selected.score
                    denum = np.abs(self.best_score)
                    scnd_part = selected.length / denum * self.mini2[idx]

                    if self.error <= num / denum + scnd_part:
                        potoptidx.append(idx)
                else:
                    scnd_part = selected.length * self.mini2[idx]

                    if selected.score <= scnd_part:
                        potoptidx.append(idx)

        return potoptidx


class Soo_tree_search(Tree_search):
    """Soo_tree_search

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, max_depth, Q=1, reverse=False):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Depth_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

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

    def add(self, c):

        self.next_frontier.append(c)

    def get_next(self):

        if len(self.next_frontier) > 0:
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

            while idx < size:

                node = self.open[idx]
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


class Move_up(Tree_search):
    """Move_up

    Attributes
    ----------

    open : list[Fractal]
        Initial Open list containing not explored nodes from the fractal rooted tree.

    max_depth : int
        maximum depth of the fractal rooted tree.

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

    def __init__(self, open, max_depth, Q=1, reverse=False):
        """__init__(open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        open : list[Fractal]
            Initial Open list containing not explored nodes from the fractal rooted tree.

        max_depth : int
            maximum depth of the fractal rooted tree.

        Q : int, default=1
            Q-Depth_first_search, at each get_next, tries to return Q nodes.

        reverse : boolean, default=False
            if False do a descending sort the open list, else do an ascending sort

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

    def add(self, c):

        self.next_frontier.append(c)

    def get_next(self):

        if len(self.next_frontier) > 0:
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

            for _ in range(self.Q):
                self.close.append(self.open.pop(0))

            return True, self.close[-self.Q :]

        else:
            return False, -1
