# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Iterable

from zellij.core.search_space import BaseFractal
from zellij.core.errors import InitializationError

import numpy as np
from itertools import groupby

import logging

logger = logging.getLogger("zellij.tree_search")


class TreeSearch(ABC):
    """TreeSearch

    TreeSearch is an abstract class which determines how to explore a
    partition tree defined by :ref:`dba`.
    Based on the OPEN/CLOSED lists algorithm.

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

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
    ):
        """__init__

        Parameters
        ----------
        openl : list[Fractal]
            Initial Open list containing unexplored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        close : boolean, default=False
            If True, store expanded, explored and scored fractals within a :code:`close` list.

        """

        ##############
        # PARAMETERS #
        ##############
        self.openl = openl
        self.save_close = save_close
        self.close = []
        self.max_depth = max_depth

    @property
    def openl(self) -> List[BaseFractal]:
        return self._openl

    @openl.setter
    def openl(self, value: Union[BaseFractal, List[BaseFractal]]):
        if isinstance(value, list):
            self._openl = value
        elif isinstance(value, BaseFractal):
            self._openl = [value]
        else:
            raise InitializationError(
                "lopen list must be a BaseFractal or a list of BaseFractal."
            )

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int):
        if value <= 0:
            raise InitializationError(f"max_depth must be > 0, got {value}")
        else:
            self._max_depth = value

    def add_close(self, nodes: List[BaseFractal]):
        if self.save_close:
            self.close.extend(nodes)

    @abstractmethod
    def add(self, c: BaseFractal):
        """__init__(open,max_depth)

        Parameters
        ----------
        c : Fractal
            Add a new fractal to the tree

        """
        pass

    @abstractmethod
    def get_next(self) -> List[BaseFractal]:
        """__init__(open, max_depth)

        Returns
        -------

        continue : boolean
            If True determine if the open list has been fully explored or not

        nodes : {list[Fractal], -1}
            if -1 no more nodes to explore, else return a list of the next node to explore
        """
        pass


class BreadthFirstSearch(TreeSearch):
    """BreadthFirstSearch

    Breadth First Search algorithm (BFS).
    Fractal are first sorted by lowest level, then by lowest score.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-Breadth_first_search, at each get_next, tries to return Q nodes.

    Methods
    -------
    add(c)
        Add a node c to the fractal tree.
    get_next()
        Get the next node to evaluate.

    See Also
    --------
    Fractal : Abstract class defining what a fractal is.
    FDA : Fractal Decomposition Algorithm.
    TreeSearch : Base class.
    DepthFirstSearch : Tree search Depth based startegy.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, BreadthFirstSearch

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> ts = BreadthFirstSearch(sp, 3)
    >>> children = sp.create_children()
    >>> for idx, c in enumerate(children):
    ...     c.score = idx
    ...     ts.add(c)
    >>> print(ts.openl)
    [Hypercube(0,-1,0)]
    >>> new = ts.get_next()
    >>> print(new)
    [Hypercube(0,-1,0)]
    >>> print(ts.openl)
    [Hypercube(1,0,0), Hypercube(1,0,1), Hypercube(1,0,2), Hypercube(1,0,3)]
    >>> print(ts.close)
    [Hypercube(0,-1,0)]
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-Breadth_first_search, at each get_next, tries to return Q nodes.

        """

        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            self.openl.sort(key=lambda x: (x.level, x.score), reverse=False)

        sel_nodes = []
        if len(self.openl) > 0:
            idx_min = min(len(self.openl), self.Q)
            sel_nodes = self.openl[:idx_min]
            self.openl = self.openl[idx_min:]

            self.add_close(sel_nodes)

        return sel_nodes


class DepthFirstSearch(TreeSearch):
    """DepthFirstSearch

    Depth First Search algorithm (DFS).
    Fractal are first sorted by highest level, then by lowest score.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    save_close : boolean, default=True
        If True save expanded, explored and scored fractal within a :code:`close` list.
    Q : int, default=1
        Q-DepthFirstSearch, at each get_next, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort.

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
    TreeSearch : Base class
    BreadthFirstSearch : Tree search Breadth based startegy
    CyclicBestFirstSearch : Hybrid between DFS and BestFS

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, DepthFirstSearch

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> ts = DepthFirstSearch(sp, 3)
    >>> children = sp.create_children()
    >>> for idx, c in enumerate(children):
    ...     c.score = idx
    ...     ts.add(c)
    >>> print(ts.openl)
    [Hypercube(0,-1,0)]
    >>> new = ts.get_next()
    >>> print(new)
    [Hypercube(1,0,0)]
    >>> print(ts.openl)
    [Hypercube(1,0,1), Hypercube(1,0,2), Hypercube(1,0,3), Hypercube(0,-1,0)]
    >>> print(ts.close)
    [Hypercube(1,0,0)]
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-Breadth_first_search, at each get_next, tries to return Q nodes.
        """

        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.openl.sort(key=lambda x: (x.level, -x.score), reverse=True)
            self.next_frontier = []

        sel_nodes = []
        if len(self.openl) > 0:
            idx_min = min(len(self.openl), self.Q)
            sel_nodes = self.openl[:idx_min]
            self.openl = self.openl[idx_min:]
            self.add_close(sel_nodes)

        return sel_nodes


class BestFirstSearch(TreeSearch):
    """BestFirstSearch

    Best First Search algorithm (BestFS).
    At each iteration, it selects the Q-best fractals.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing non-expanded nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-BestFirstSearch, at each :code:`get_next`, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort

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
    TreeSearch : Base class
    BeamSearch : Memory efficient tree search algorithm based on BestFS
    CyclicBestFirstSearch : Hybrid between DFS and BestFS

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, DepthFirstSearch

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> ts = DepthFirstSearch(sp, 3)
    >>> children = sp.create_children()
    >>> for idx, c in enumerate(children):
    ...     c.score = idx
    ...     ts.add(c)
    >>> print(ts.openl)
    [Hypercube(0,-1,0)]
    >>> new = ts.get_next()
    >>> print(new, new[0].score)
    [Hypercube(1,0,0)] 0
    >>> print(ts.openl)
    [Hypercube(1,0,1), Hypercube(1,0,2), Hypercube(1,0,3), Hypercube(0,-1,0)]
    >>> print(ts.close)
    [Hypercube(1,0,0)]

    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
        reverse: bool = False,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-BestFirstSearch, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############

        self.next_frontier = []

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self):
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            self.openl.sort(key=lambda x: x.score, reverse=self.reverse)

        sel_nodes = []
        if len(self.openl) > 0:
            idx_min = min(len(self.openl), self.Q)
            sel_nodes = self.openl[:idx_min]
            self.openl = self.openl[idx_min:]
            self.add_close(sel_nodes)

        return sel_nodes


class BeamSearch(TreeSearch):
    """BeamSearch

    Beam Search algorithm (BS). BS is an improvement of BestFS.
    It includes a beam length which allows to prune the worst nodes.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-BeamSearch, at each get_next, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort

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
    TreeSearch : Base class
    BestFirstSearch : Tree search algorithm based on the best node from the open list
    CyclicBestFirstSearch : Hybrid between DFS and BestFS, which can also perform pruning.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, BeamSearch

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> ts = BeamSearch(sp, 3, beam_length=3)
    >>> children = sp.create_children()
    >>> for idx, c in enumerate(children):
    ...     c.score = idx
    ...     ts.add(c)
    >>> print(ts.openl)
    [Hypercube(0,-1,0)]
    >>> new = ts.get_next()
    >>> print(new, new[0].score)
    [Hypercube(1,0,0)] 0
    >>> print(ts.openl)
    [Hypercube(1,0,1), Hypercube(1,0,2), Hypercube(1,0,3)]
    >>> print(ts.close)
    [Hypercube(1,0,0)]

    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
        reverse: bool = False,
        beam_length: int = 10,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-BeamSearch, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort
        beam_length : int, default=10
            Determines the length of the open list for memory and prunning issues.

        """

        super().__init__(openl, max_depth, save_close)

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

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            self.openl.sort(key=lambda x: x.score, reverse=self.reverse)
            self.next_frontier = []

        sel_nodes = []
        if len(self.openl) > 0:
            idx_min = min(len(self.openl), self.Q)
            sel_nodes = self.openl[:idx_min]
            self.openl = self.openl[idx_min:][: self.beam_length]
            self.add_close(sel_nodes)

        return sel_nodes


class EpsilonGreedySearch(TreeSearch):
    """Epsilon_greedy_search

    Epsilon Greedy Search (EGS).
    EGS is an improvement of BestFS. At each iteration, nodes are selected
    randomly or according to their best score.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-Epsilon_greedy_search, at each get_next, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort.
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
    FDA : Fractal Decomposition Algorithm.
    TreeSearch : Base class.
    BestFirstSearch : Tree search algorithm based on the best node from the open list.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, EpsilonGreedySearch

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> ts = EpsilonGreedySearch(sp, 3, epsilon=0.5)
    >>> children = sp.create_children()
    >>> for idx, c in enumerate(children):
    ...     c.score = idx
    ...     ts.add(c)
    >>> print(ts.openl)
    [Hypercube(0,-1,0)]
    >>> new = ts.get_next()
    >>> print(new, new[0].score)
    [Hypercube(1,0,3)] 3
    >>> print(ts.openl)
    [Hypercube(1,0,0), Hypercube(1,0,1), Hypercube(1,0,2), Hypercube(0,-1,0)]
    >>> print(ts.close)
    [Hypercube(1,0,3)]

    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
        reverse: bool = False,
        epsilon: float = 0.1,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal,list[Fractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-Epsilon_greedy_search, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort
        epsilon : float, default=0.1
            Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration EGS does.

        """

        super().__init__(openl, max_depth, save_close)

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

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            self.openl.sort(key=lambda x: x.score, reverse=self.reverse)

        sel_nodes = []
        if len(self.openl) > 0:
            idx_min = min(len(self.openl), self.Q)
            for _ in range(idx_min):
                if np.random.random() > self.epsilon:
                    sel_nodes.append(self.openl.pop(0))
                else:
                    if len(self.openl) > 1:
                        idx = np.random.randint(1, len(self.openl))
                        sel_nodes.append(self.openl.pop(idx))
                    else:
                        sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)

        return sel_nodes


class CyclicBestFirstSearch(TreeSearch):
    """CyclicBestFirstSearch

    Cyclic Best First Search (CBFS). CBFS is an hybridation between DFS and
    BestFS. First, CBFS tries to reach a leaf of the fractal tree to quickly
    determine a base score. Then CBFS will do pruning according to this value,
    and will decompose the problem into subproblems by inserting nodes into
    contours (collection of unexplored subproblems). At each iteration CBFS
    selects the best subproblem according to an heuristic value.
    Then the child will be inserted into their respective contours
    according to a labelling function.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-CyclicBestFirstSearch, at each get_next, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort

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
    TreeSearch : Base class
    BestFirstSearch : Tree search algorithm based on the best node from the open list
    DepthFirstSearch : Tree search Depth based startegy

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, CyclicBestFirstSearch

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> ts = CyclicBestFirstSearch(sp, 3)
    >>> children = sp.create_children()
    >>> for idx, c in enumerate(children):
    ...     c.score = idx
    ...     ts.add(c)
    >>> print(ts.contour)
    [[Hypercube(0,-1,0)], [], [], []]
    >>> new = ts.get_next()
    >>> print(new, new[0].score)
    [Hypercube(0,-1,0)] inf
    >>> print(ts.contour)
    [[Hypercube(0,-1,0)], [Hypercube(1,0,0), Hypercube(1,0,1), Hypercube(1,0,2), Hypercube(1,0,3)], [], []]
    >>> print(ts.close)
    [Hypercube(0,-1,0)]
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
        reverse: bool = False,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-DepthFirstSearch, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(openl, max_depth, save_close)

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
        self.contour[0] = self.openl

        self.best_scores = float("inf")
        self.first_complete = False

    def add(self, c: BaseFractal):
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

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            modified_levels = []
            for h in self.next_frontier:
                self.contour[h.level].append(h)
                modified_levels.append(h.level)

                if not self.L[h.level]:
                    self.L[h.level] = True

            modified_levels = np.unique(modified_levels)
            for l in modified_levels:
                self.contour[l].sort(key=lambda x: x.score, reverse=self.reverse)

            self.next_frontier = []

        sel_nodes = []

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

            idx_min = min(len(self.contour[self.i]), self.Q)
            sel_nodes = self.openl[:idx_min]
            self.openl = self.openl[idx_min:]
            self.add_close(sel_nodes)

            if len(self.contour[self.i]) == 0:
                self.L[self.i] = False

        return sel_nodes


##########
# DIRECT #
##########


class PotentiallyOptimalRectangle(TreeSearch):
    """Potentially_Optimal_Rectangle

    Potentially Optimal Rectangle algorithm (POR),
    is a the selection strategy comming from DIRECT.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
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
    TreeSearch : Base class
    BeamSearch : Memory efficient tree search algorithm based on BestFS
    CyclicBestFirstSearch : Hybrid between DFS and BestFS
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int = 600,
        save_close: bool = True,
        error: float = 1e-4,
        maxdiv: int = 3000,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.
        maxdiv : int, default=3000
            Prunning parameter. Maximum number of fractals to store.
        """
        super().__init__(openl, max_depth, save_close)
        ##############
        # PARAMETERS #
        ##############
        self.error = error
        self.maxdiv = maxdiv

        #############
        # VARIABLES #
        #############
        self.maxi1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.mini2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        self.best_score = float("inf")

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)
        if c.score < self.best_score:
            self.best_score = c.score

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            # sort potentially optimal rectangle by length (increasing)
            # then by score
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            # clip open list to maxdiv (oldest subspace are prunned)
            self.openl = self.openl[-self.maxdiv :]
            self.openl.sort(key=lambda x: (x.measure, x.score))

        sel_nodes = []
        self.maxi1.fill(-float("inf"))
        self.mini2.fill(float("inf"))
        if len(self.openl) > 1:
            # I3 groups
            groups = groupby(self.openl, lambda x: x.measure)
            idx = self.optimal(groups)
            if idx:
                for i in reversed(idx):
                    sel_nodes.append(self.openl.pop(i))
                self.add_close(sel_nodes)
        elif len(self.openl) == 1:
            sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)
        return sel_nodes

    def optimal(
        self, groups: Iterable[Tuple[float, Iterable[BaseFractal]]]
    ) -> List[int]:
        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # Potentially optimal index
        potoptidx = []

        group_size = 0
        for _, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            selected = subgroup[idx]
            current_idx = group_size + idx

            is_potopt = False

            for jdx in range(current_idx + len(subgroup), len(self.openl)):
                c = self.openl[jdx]
                # I1 group
                if c.measure < selected.measure:
                    num = selected.score - c.score
                    denom = selected.measure - c.measure
                    if denom != 0:
                        low_k = num / denom
                    else:
                        low_k = -float("inf")

                    if low_k > self.maxi1[current_idx]:
                        self.maxi1[current_idx] = low_k
                    elif low_k < self.mini2[jdx]:
                        self.mini2[jdx] = low_k
                # I2 group
                elif c.measure > selected.measure:
                    denom = c.measure - selected.measure
                    num = c.score - selected.score
                    if denom != 0:
                        up_k = (num) / (denom)
                    else:
                        up_k = float("inf")

                    if up_k < self.mini2[current_idx]:
                        self.mini2[current_idx] = up_k
                    elif up_k > self.maxi1[jdx]:
                        self.maxi1[jdx] = up_k

                if self.mini2[current_idx] > 0 and (
                    self.maxi1[current_idx] <= self.mini2[current_idx]
                ):
                    if self.best_score != 0:
                        num = self.best_score - selected.score
                        denum = np.abs(self.best_score)
                        scnd_part = selected.measure / denum * self.mini2[current_idx]

                        left = num / denum + scnd_part
                        if np.isnan(left) or (self.error <= left):
                            is_potopt = True
                    else:
                        scnd_part = selected.measure * self.mini2[current_idx]

                        if selected.score <= scnd_part:
                            is_potopt = True

                if is_potopt:
                    potoptidx.append(current_idx)
                    idx += 1
                    while (
                        idx < len(subgroup)
                        and np.abs(subgroup[idx].score - current_score) <= 1e-13
                    ):
                        current_idx = group_size + idx
                        potoptidx.append(current_idx)
                        idx += 1

            group_size += len(subgroup)
        return potoptidx


class LocallyBiasedPOR(TreeSearch):
    """Locally_biased_POR

    Locally_biased_POR, is a the selection strategy comming from Locally Biased DIRECT.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
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
    TreeSearch : Base class
    BeamSearch : Memory efficient tree search algorithm based on BestFS
    CyclicBestFirstSearch : Hybrid between DFS and BestFS
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int = 600,
        save_close: bool = True,
        error: float = 1e-4,
        maxdiv: int = 3000,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.
        maxdiv : int, default=3000
            Prunning parameter. Maximum number of fractals to store.

        """
        super().__init__(openl, max_depth, save_close)
        ##############
        # PARAMETERS #
        ##############
        self.error = error
        self.maxdiv = maxdiv

        #############
        # VARIABLES #
        #############
        self.maxi1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.mini2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        self.best_score = float("inf")

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)
        if c.score < self.best_score:
            self.best_score = c.score

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            # sort potentially optimal rectangle by length (increasing)
            # then by score
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            # clip open list to maxdiv (oldest subspace are prunned)
            self.openl = self.openl[-self.maxdiv :]
            self.openl.sort(key=lambda x: (x.measure, x.score))

        sel_nodes = []
        self.maxi1.fill(-float("inf"))
        self.mini2.fill(float("inf"))
        if len(self.openl) > 1:
            # I3 groups
            groups = groupby(self.openl, lambda x: x.measure)
            idx = self.optimal(groups)

            if idx:
                for i in reversed(idx):
                    sel_nodes.append(self.openl.pop(i))
                self.add_close(sel_nodes)

        elif len(self.openl) == 1:
            sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)

        return sel_nodes

    def optimal(
        self, groups: Iterable[Tuple[float, Iterable[BaseFractal]]]
    ) -> List[int]:
        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # found potopt rectangle at level x
        found_levels = [False] * self.max_depth

        # Potentially optimal index
        potoptidx = []

        group_size = 0
        for _, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            selected = subgroup[idx]
            current_idx = group_size + idx

            is_potopt = False

            for jdx in range(current_idx + len(subgroup), len(self.openl)):
                c = self.openl[jdx]
                # I1 group
                if c.measure < selected.measure:
                    num = selected.score - c.score
                    denom = selected.measure - c.measure
                    if denom != 0:
                        low_k = num / denom
                    else:
                        low_k = -float("inf")

                    if low_k > self.maxi1[current_idx]:
                        self.maxi1[current_idx] = low_k
                    elif low_k < self.mini2[jdx]:
                        self.mini2[jdx] = low_k
                # I2 group
                elif c.measure > selected.measure:
                    denom = c.measure - selected.measure
                    num = c.score - selected.score
                    if denom != 0:
                        up_k = (num) / (denom)
                    else:
                        up_k = float("inf")

                    if up_k < self.mini2[current_idx]:
                        self.mini2[current_idx] = up_k
                    elif up_k > self.maxi1[jdx]:
                        self.maxi1[jdx] = up_k

            if self.mini2[current_idx] > 0 and (
                self.maxi1[current_idx] <= self.mini2[current_idx]
            ):
                if self.best_score != 0:
                    num = self.best_score - selected.score
                    denum = np.abs(self.best_score)
                    scnd_part = selected.measure / denum * self.mini2[current_idx]

                    left = num / denum + scnd_part
                    if np.isnan(left) or (self.error <= left):
                        is_potopt = True
                else:
                    scnd_part = selected.measure * self.mini2[current_idx]

                    if selected.score <= scnd_part:
                        is_potopt = True

            if is_potopt and not found_levels[selected.level]:
                potoptidx.append(current_idx)
                found_levels[selected.level] = True
                idx += 1
                while (
                    idx < len(subgroup)
                    and np.abs(subgroup[idx].score - current_score) <= 1e-13
                ):
                    current_idx = group_size + idx
                    selected = subgroup[idx]
                    if not found_levels[selected.level]:
                        potoptidx.append(current_idx)
                        found_levels[selected.level] = True
                    idx += 1

            group_size += len(subgroup)
        return potoptidx


class AdaptivePOR(TreeSearch):
    """Adaptive_POR

    Adaptive_POR, is a the selection strategy
    comming from DIRECT-Restart.

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
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
    TreeSearch : Base class
    BeamSearch : Memory efficient tree search algorithm based on BestFS
    CyclicSestSirstSearch : Hybrid between DFS and BestFS
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int = 600,
        save_close: bool = True,
        error: float = 1e-2,
        maxdiv: int = 3000,
        patience: int = 5,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-BestFirstSearch, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort
        error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.

        """
        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.max_error = error
        self.maxdiv = maxdiv
        self.patience = patience
        #############
        # VARIABLES #
        #############
        self.maxi1 = np.full(self.maxdiv, -float("inf"), dtype=float)
        self.mini2 = np.full(self.maxdiv, float("inf"), dtype=float)

        self.next_frontier = []
        self.best_score = float("inf")
        self.new_best_score = float("inf")

        self.stagnation = 0
        self.error = self.max_error

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)
        if c.score < self.new_best_score:
            self.new_best_score = c.score

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            # clip open list to maxdiv (oldest subspace are prunned)
            self.openl = self.openl[-self.maxdiv :]
            self.openl.sort(key=lambda x: (-x.measure, x.score))

        if np.abs(self.best_score - self.new_best_score) < 1e-4:
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

        sel_nodes = []
        self.maxi1.fill(-float("inf"))
        self.mini2.fill(float("inf"))
        if len(self.openl) > 1:
            groups = groupby(self.openl, lambda x: x.measure)
            idx = self.optimal(groups)

            if idx:
                for i in reversed(idx):
                    sel_nodes.append(self.openl.pop(i))
                self.add_close(sel_nodes)

        elif len(self.openl) == 1:
            sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)

        return sel_nodes

    def optimal(
        self, groups: Iterable[Tuple[float, Iterable[BaseFractal]]]
    ) -> List[int]:
        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # Potentially optimal index
        potoptidx = []

        group_size = 0
        for _, value in groups:
            subgroup = list(value)
            current_score = subgroup[0].score
            idx = 0
            selected = subgroup[idx]
            current_idx = group_size + idx

            is_potopt = False

            for jdx in range(current_idx + len(subgroup), len(self.openl)):
                c = self.openl[jdx]

                # I1 group
                if c.measure < selected.measure:
                    num = selected.score - c.score
                    denom = selected.measure - c.measure
                    if denom != 0:
                        low_k = num / denom
                    else:
                        low_k = -float("inf")

                    if low_k > self.maxi1[current_idx]:
                        self.maxi1[current_idx] = low_k
                    elif low_k < self.mini2[jdx]:
                        self.mini2[jdx] = low_k
                # I2 group
                elif c.measure > selected.measure:
                    denom = c.measure - selected.measure
                    num = c.score - selected.score
                    if denom != 0:
                        up_k = (num) / (denom)
                    else:
                        up_k = float("inf")

                    if up_k < self.mini2[current_idx]:
                        self.mini2[current_idx] = up_k
                    elif up_k > self.maxi1[jdx]:
                        self.maxi1[jdx] = up_k

            if self.mini2[current_idx] > 0 and (
                self.maxi1[current_idx] <= self.mini2[current_idx]
            ):
                if self.best_score != 0:
                    num = self.best_score - selected.score
                    denum = np.abs(self.best_score)
                    scnd_part = selected.measure / denum * self.mini2[current_idx]

                    left = num / denum + scnd_part
                    if np.isnan(left) or (self.error <= left):
                        is_potopt = True
                else:
                    scnd_part = selected.measure * self.mini2[current_idx]

                    if selected.score <= scnd_part:
                        is_potopt = True

            if is_potopt:
                potoptidx.append(current_idx)
                idx += 1
                while (
                    idx < len(subgroup)
                    and np.abs(subgroup[idx].score - current_score) <= 1e-13
                ):
                    current_idx = group_size + idx
                    potoptidx.append(current_idx)
                    idx += 1

            group_size += len(subgroup)
        return potoptidx


#######
# SOO #
#######


class SooTreeSearch(TreeSearch):
    """SooTreeSearch

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-DepthFirstSearch, at each get_next, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort

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
    TreeSearch : Base class
    Breadth_first_search : Tree search Breadth based startegy
    CyclicBestFirstSearch : Hybrid between DFS and BestFS
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
        reverse: bool = False,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-DepthFirstSearch, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############
        self.next_frontier = []

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            # sort leaves according to level and score ascending
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            self.openl.sort(reverse=self.reverse, key=lambda x: (x.level, x.score))

        sel_nodes = []
        if len(self.openl) > 0:
            current_level = self.openl[0].level
            sel_nodes.append(self.openl.pop(0))
            idx_min = 1

            idx = 0
            size = len(self.openl)

            # select the lowest score among all leaves at the current level
            while idx < size:
                node = self.openl[idx]
                # If level change, then select the first node of this level.
                # (with the lowest score)
                if node.level != current_level:
                    current_level = node.level
                    sel_nodes.append(self.openl.pop(idx))
                    idx -= 1
                    size -= 1
                    idx_min += 1

                idx += 1

            self.add_close(sel_nodes)

        return sel_nodes


#######
# FDA #
#######


class MoveUp(TreeSearch):
    """Move_up

    FDA tree search.

    Attributes
    ----------
    openl : {BaseFractal, list[BaseFractal]}
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.
    Q : int, default=1
        Q-DepthFirstSearch, at each get_next, tries to return Q nodes.
    reverse : boolean, default=False
        If False do a descending sort the open list, else do an ascending sort

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
    TreeSearch : Base class
    Breadth_first_search : Tree search Breadth based startegy
    CyclicBestFirstSearch : Hybrid between DFS and BestFS
    """

    def __init__(
        self,
        openl: Union[BaseFractal, List[BaseFractal]],
        max_depth: int,
        save_close: bool = True,
        Q: int = 1,
        reverse: bool = False,
    ):
        """__init__

        Parameters
        ----------
        openl : {BaseFractal, list[BaseFractal]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        Q : int, default=1
            Q-DepthFirstSearch, at each get_next, tries to return Q nodes.
        reverse : boolean, default=False
            If False do a descending sort the open list, else do an ascending sort

        """
        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.reverse = reverse
        self.Q = Q

        #############
        # VARIABLES #
        #############
        self.next_frontier = []

    def add(self, c: BaseFractal):
        self.next_frontier.append(c)

    def get_next(self) -> List[BaseFractal]:
        if len(self.next_frontier) > 0:
            self.openl.extend(self.next_frontier)
            self.next_frontier = []
            self.openl.sort(reverse=self.reverse, key=lambda x: (-x.level, x.score))

        sel_nodes = []
        if len(self.openl) > 0:
            sel_nodes = self.openl[: self.Q]
            self.openl = self.openl[self.Q :]
            self.add_close(sel_nodes)

        return sel_nodes
