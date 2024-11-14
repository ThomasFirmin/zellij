# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union, Callable, TYPE_CHECKING, Tuple
from collections import defaultdict

from zellij.core.search_space import BaseFractal
from zellij.core.errors import InitializationError
from zellij.strategies.tools.geometry import (
    NMSOSection,
    LatinHypercubeUCB,
)

import numpy as np
from itertools import groupby
import logging
from bisect import insort_left
import gc

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
        save_close: bool = False,
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
        save_close: bool = False,
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

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (x.level, x.score))

    def get_next(self) -> List[BaseFractal]:
        sel_nodes = []
        if len(self.openl) > 0:
            sel_nodes = self.openl[: self.Q]
            self.openl = self.openl[self.Q :]
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
        save_close: bool = False,
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

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (-x.level, x.score))

    def get_next(self) -> List[BaseFractal]:
        sel_nodes = []
        if len(self.openl) > 0:
            sel_nodes = self.openl[: self.Q]
            self.openl = self.openl[self.Q :]
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
        save_close: bool = False,
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
            Q-BestFirstSearch, at each get_next, tries to return Q nodes.

        """
        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.Q = Q

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: x.score)

    def get_next(self):
        gc.collect()
        sel_nodes = []
        if len(self.openl) > 0:
            sel_nodes = self.openl[: self.Q]
            self.openl = self.openl[self.Q :]
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
        save_close: bool = False,
        Q: int = 1,
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
        beam_length : int, default=10
            Determines the length of the open list for memory and prunning issues.

        """

        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.Q = Q
        self.beam_length = beam_length

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: x.score)
        if len(self.openl) > self.beam_length:
            self.openl.pop()

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
        sel_nodes = []
        if len(self.openl) > 0:
            sel_nodes = self.openl[: self.Q]
            self.openl = self.openl[self.Q :]
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
        save_close: bool = False,
        Q: int = 1,
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
        epsilon : float, default=0.1
            Probability to select a random node from the open list. Determine how random the selection must be. The higher it is, the more exploration EGS does.

        """

        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############

        self.Q = Q
        self.epsilon = epsilon

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: x.score)

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
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
        save_close: bool = False,
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
            Q-DepthFirstSearch, at each get_next, tries to return Q nodes.

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
        gc.collect()
        if len(self.next_frontier) > 0:
            modified_levels = []
            for h in self.next_frontier:
                self.contour[h.level].append(h)
                modified_levels.append(h.level)

                if not self.L[h.level]:
                    self.L[h.level] = True

            modified_levels = np.unique(modified_levels)
            for l in modified_levels:
                self.contour[l].sort(key=lambda x: x.score)

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
    error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.
    maxopen : int, default=3000
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
        save_close: bool = False,
        error: float = 1e-4,
        maxopen: int = 3000,
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
        maxopen : int, default=3000
            Prunning parameter. Maximum number of fractals to store.
        """
        super().__init__(openl, max_depth, save_close)
        ##############
        # PARAMETERS #
        ##############
        self.error = error
        self.maxopen = maxopen

        #############
        # VARIABLES #
        #############
        self.best_score = float("inf")

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (x.measure, x.score))
        if c.score < self.best_score:
            self.best_score = c.score
        if len(self.openl) > self.maxopen:
            self.openl.pop(0)

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
        sel_nodes = []
        if len(self.openl) > 1:
            # I3 groups
            groups = groupby(self.openl, lambda x: x.measure)
            max_groups = []
            min_groups = []
            size_groups = []
            len_groups = []
            for size, g in groups:
                lg = list(g)
                max_groups.append(lg[-1].score)
                min_groups.append(lg[0].score)
                size_groups.append(size)
                len_groups.append(len(lg))
            max_groups = np.array(max_groups, dtype=float)
            min_groups = np.array(min_groups, dtype=float)
            size_groups = np.array(size_groups, dtype=float)

            idx = self.optimal(max_groups, min_groups, size_groups, len_groups)
            if idx:
                for i in reversed(idx):
                    sel_nodes.append(self.openl.pop(i))
                self.add_close(sel_nodes)
        elif len(self.openl) == 1:
            sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)
        return sel_nodes

    def optimal(
        self,
        max_groups: np.ndarray,
        min_groups: np.ndarray,
        size_groups: np.ndarray,
        len_groups: list,
    ) -> List[int]:

        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # Potentially optimal index
        potoptidx = []
        group_size = 0

        for gidx, (jvalue, jsize, grp_len) in enumerate(
            zip(min_groups, size_groups, len_groups)
        ):
            current_idx = group_size
            if gidx > 0:
                max_i1 = np.max(
                    (jvalue - min_groups[:gidx]) / (jsize - size_groups[:gidx])
                )
            else:
                max_i1 = 0

            gidxpo = gidx + 1
            if gidxpo < len(size_groups):
                min_i2 = np.min(
                    (min_groups[gidxpo:] - jvalue) / (size_groups[gidxpo:] - jsize)
                )
            else:
                min_i2 = float("inf")

            if not np.isfinite(min_i2):
                comp = True
            elif self.best_score != 0:
                right = (1 / np.abs(self.best_score)) * (
                    self.best_score - jvalue + jsize * min_i2
                )
                comp = self.error <= right
            else:
                right = jsize * min_i2
                comp = jvalue <= right

            # Potentially optimal
            if (max_i1 <= min_i2) and comp:
                potoptidx.append(current_idx)
                idx = 1
                current_idx = group_size + idx
                while (
                    idx < grp_len
                    and np.abs(self.openl[current_idx].score - jvalue) <= self.error
                ):
                    potoptidx.append(current_idx)
                    idx += 1
                    current_idx = group_size + idx

            group_size += grp_len
        # print(min_groups, len_groups, size_groups, potoptidx)
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
    error : float, default=1e-4
            Small value which determines when an evaluation should be considered
            as good as the best solution found so far.
    maxopen : int, default=3000
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
        save_close: bool = False,
        error: float = 1e-4,
        maxopen: int = 3000,
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
        maxopen : int, default=3000
            Prunning parameter. Maximum number of fractals to store.

        """
        super().__init__(openl, max_depth, save_close)
        ##############
        # PARAMETERS #
        ##############
        self.error = error
        self.maxopen = maxopen

        #############
        # VARIABLES #
        #############
        self.best_score = float("inf")

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (x.measure, x.score))
        if c.score < self.best_score:
            self.best_score = c.score
        if len(self.openl) > self.maxopen:
            self.openl.pop(0)

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
        sel_nodes = []
        if len(self.openl) > 1:
            # I3 groups
            groups = groupby(self.openl, lambda x: x.measure)
            max_groups = []
            min_groups = []
            size_groups = []
            len_groups = []
            for size, g in groups:
                lg = list(g)
                max_groups.append(lg[-1].score)
                min_groups.append(lg[0].score)
                size_groups.append(size)
                len_groups.append(len(lg))
            max_groups = np.array(max_groups, dtype=float)
            min_groups = np.array(min_groups, dtype=float)
            size_groups = np.array(size_groups, dtype=float)

            idx = self.optimal(max_groups, min_groups, size_groups, len_groups)
            if idx:
                for i in reversed(idx):
                    sel_nodes.append(self.openl.pop(i))
                self.add_close(sel_nodes)
        elif len(self.openl) == 1:
            sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)
        return sel_nodes

    def optimal(
        self,
        max_groups: np.ndarray,
        min_groups: np.ndarray,
        size_groups: np.ndarray,
        len_groups: list,
    ) -> List[int]:

        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        sel_levels = [True] * self.max_depth

        # Potentially optimal index
        potoptidx = []
        group_size = 0

        for gidx, (jvalue, jsize, grp_len) in enumerate(
            zip(min_groups, size_groups, len_groups)
        ):
            current_idx = group_size
            if gidx > 0:
                max_i1 = np.max(
                    (jvalue - min_groups[:gidx]) / (jsize - size_groups[:gidx])
                )
            else:
                max_i1 = 0

            gidxpo = gidx + 1
            if gidxpo < len(size_groups):
                min_i2 = np.min(
                    (min_groups[gidxpo:] - jvalue) / (size_groups[gidxpo:] - jsize)
                )
            else:
                min_i2 = float("inf")

            if not np.isfinite(min_i2):
                comp = True
            elif self.best_score != 0:
                right = (1 / np.abs(self.best_score)) * (
                    self.best_score - jvalue + jsize * min_i2
                )
                comp = self.error <= right
            else:
                right = jsize * min_i2
                comp = jvalue <= right

            # Potentially optimal
            if (
                (max_i1 <= min_i2)
                and comp
                and sel_levels[self.openl[current_idx].level]
            ):
                potoptidx.append(current_idx)
                sel_levels[self.openl[current_idx].level] = False
                idx = 1
                current_idx = group_size + idx
                while (
                    idx < grp_len
                    and np.abs(self.openl[current_idx].score - jvalue) <= self.error
                    and sel_levels[self.openl[current_idx].level]
                ):
                    potoptidx.append(current_idx)
                    sel_levels[self.openl[current_idx].level] = False
                    idx += 1
                    current_idx = group_size + idx

            group_size += grp_len
        # print(min_groups, len_groups, size_groups, potoptidx)
        return potoptidx


class AdaptivePOR(TreeSearch):
    """Adaptive_POR

    Adaptive_POR, is a the selection strategy from DIRECT-Restart.

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
        save_close: bool = False,
        error: float = 1e-2,
        maxopen: int = 3000,
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
        self.maxopen = maxopen
        self.patience = patience
        #############
        # VARIABLES #
        #############
        self.best_score = float("inf")
        self.new_best_score = float("inf")

        self.stagnation = 0
        self.error = self.max_error

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (x.measure, x.score))
        if c.score < self.best_score:
            self.best_score = c.score
        if len(self.openl) > self.maxopen:
            self.openl.pop(0)

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
        if np.abs(self.best_score - self.new_best_score) < self.max_error:
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

        if len(self.openl) > 1:
            groups = groupby(self.openl, lambda x: x.measure)
            max_groups = []
            min_groups = []
            size_groups = []
            len_groups = []
            for size, g in groups:
                lg = list(g)
                max_groups.append(lg[-1].score)
                min_groups.append(lg[0].score)
                size_groups.append(size)
                len_groups.append(len(lg))
            max_groups = np.array(max_groups, dtype=float)
            min_groups = np.array(min_groups, dtype=float)
            size_groups = np.array(size_groups, dtype=float)

            idx = self.optimal(max_groups, min_groups, size_groups, len_groups)

            if idx:
                for i in reversed(idx):
                    sel_nodes.append(self.openl.pop(i))
                self.add_close(sel_nodes)

        elif len(self.openl) == 1:
            sel_nodes.append(self.openl.pop(0))
            self.add_close(sel_nodes)

        return sel_nodes

    def optimal(
        self,
        max_groups: np.ndarray,
        min_groups: np.ndarray,
        size_groups: np.ndarray,
        len_groups: list,
    ) -> List[int]:

        # see DIRECT Optimization Algorithm User Guide Daniel E. Finkel
        # for explanation

        # Potentially optimal index
        potoptidx = []
        group_size = 0

        for gidx, (jvalue, jsize, grp_len) in enumerate(
            zip(min_groups, size_groups, len_groups)
        ):
            current_idx = group_size
            if gidx > 0:
                max_i1 = np.max(
                    (jvalue - min_groups[:gidx]) / (jsize - size_groups[:gidx])
                )
            else:
                max_i1 = 0

            gidxpo = gidx + 1
            if gidxpo < len(size_groups):
                min_i2 = np.min(
                    (min_groups[gidxpo:] - jvalue) / (size_groups[gidxpo:] - jsize)
                )
            else:
                min_i2 = float("inf")

            if not np.isfinite(min_i2):
                comp = True
            elif self.best_score != 0:
                right = (1 / np.abs(self.best_score)) * (
                    self.best_score - jvalue + jsize * min_i2
                )
                comp = self.error <= right
            else:
                right = jsize * min_i2
                comp = jvalue <= right

            # Potentially optimal
            if (max_i1 <= min_i2) and comp:
                potoptidx.append(current_idx)
                idx = 1
                current_idx = group_size + idx
                while (
                    idx < grp_len
                    and np.abs(self.openl[current_idx].score - jvalue) <= self.error
                ):
                    potoptidx.append(current_idx)
                    idx += 1
                    current_idx = group_size + idx

            group_size += grp_len
        # print(min_groups, len_groups, size_groups, potoptidx)
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
        save_close: bool = False,
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

        """
        super().__init__(openl, max_depth, save_close)

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (x.level, x.score))

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
        sel_nodes = []
        node = self.openl.pop(0)
        sel_nodes.append(node)
        current_level = node.level
        vmax = node.score

        if len(self.openl) > 0:
            idx = 0
            size_ol = len(self.openl)
            # select the lowest score among all leaves at the current level
            while idx < size_ol:
                # If level change, then select the first node of this level.
                # (with the lowest score)
                if self.openl[idx].level > current_level:
                    current_level = self.openl[idx].level
                    if node.score <= vmax:
                        node = self.openl.pop(idx)
                        sel_nodes.append(node)
                        vmax = node.score
                        idx -= 1
                        size_ol -= 1

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
        save_close: bool = False,
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
            Q-DepthFirstSearch, at each get_next, tries to return Q nodes.

        """
        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.Q = Q

    def add(self, c: BaseFractal):
        insort_left(self.openl, c, key=lambda x: (-x.level, x.score))

    def get_next(self) -> List[BaseFractal]:
        gc.collect()
        sel_nodes = []
        if len(self.openl) > 0:
            sel_nodes = self.openl[: self.Q]
            self.openl = self.openl[self.Q :]
            self.add_close(sel_nodes)

        return sel_nodes


########
# NMSO #
########


class NMSOTreeSearch(TreeSearch):
    """NMSOTreeSearch

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.

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
        openl: Union[NMSOSection, List[NMSOSection]],
        max_depth: int,
        V: int,
        alpha: float,
        beta: float,
        save_close: bool = False,
    ):
        """__init__

        Parameters
        ----------
        openl : {NMSOSection, list[NMSOSection]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.

        V : int
            How many evaluation before visiting the basket.
        alpha : float, default=1e-8
        beta : float, default=1e-8
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.

        """
        self.size = -1
        super().__init__(openl, max_depth, save_close)  # type: ignore

        ##############
        # PARAMETERS #
        ##############
        self.V = V
        self.alpha = alpha
        self.beta = beta

        #############
        # VARIABLES #
        #############
        self.current_depth = 0
        self._new_child = []
        self._newleft = None
        self._newmiddle = None
        self._newright = None

    @property
    def openl(self) -> defaultdict:
        return self._openl

    @openl.setter
    def openl(self, value: Union[NMSOSection, List[NMSOSection]]):
        self._openl = defaultdict(lambda: [])
        if isinstance(value, list):
            for node in value:
                self._openl[node.level].append(node)
            self.size = value[0].size
        elif isinstance(value, NMSOSection):
            self._openl[value.level].append(value)
            self.size = value.size
        else:
            raise InitializationError(
                "lopen list must be a BaseFractal or a list of BaseFractal."
            )

    def add(self, c: NMSOSection):
        # ALL NEW FRACTALS ARE FROM THE SAME PARENT AS get_next RETURNS ONLY 1 FRACTAL
        insort_left(self.openl[c.level], c, key=lambda x: x.score)
        if c.left:
            self._newleft = c
        elif c.middle:
            self._newmiddle = c
        elif c.right:
            self._newright = c
        self._new_child.append(c)

    def get_next(self) -> List[NMSOSection]:
        gc.collect()
        if self.current_depth > self.max_depth:
            self.current_depth = 1

        if self._newmiddle and self._newright and self._newleft:
            # COMPUTE DF (<= alpha)
            new_df = np.abs(self._newright.score - self._newleft.score)
            if (self._newright.level - 1) % self.size == 0:
                df = new_df
            else:
                df = max(self._newright.df, new_df)

            for child in self._new_child:
                child.df = df

            if (
                (self._newmiddle.level % self.size == 0)
                and (df <= self.alpha)
                and (self._newmiddle.dx <= self.beta)
            ):
                self.current_depth = 1
                for child in self._new_child:
                    child.visited = 0

            self._new_child = []
            self._newleft = None
            self._newmiddle = None
            self._newright = None

        leaf_at_l = len(self.openl[self.current_depth])

        update_basket = True
        true_idx = 0
        while update_basket and true_idx < leaf_at_l:
            best_leaf = self.openl[self.current_depth][true_idx]
            if best_leaf.visited < self.V:
                best_leaf.visited += 1
                true_idx += 1
            else:
                update_basket = False

        if true_idx >= leaf_at_l:
            self.current_depth += 1
            return self.get_next()
        else:
            sel_node = [self.openl[self.current_depth].pop(true_idx)]
            self.add_close(sel_node)
            self.current_depth += 1
            return sel_node


class SOOUCB(TreeSearch):
    """SOOUCB

    Attributes
    ----------
    openl : list[BaseFractal]
        Initial Open list containing not explored nodes from the partition tree.
    max_depth : int
        Maximum depth of the partition tree.

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
    """

    def __init__(
        self,
        openl: BaseFractal,
        max_depth: int,
        nu: float,
        save_close: bool = False,
    ):
        """__init__

        Parameters
        ----------
        openl : {LatinHypercube, list[LatinHypercube]}
            Initial Open list containing not explored nodes from the partition tree.
        max_depth : int
            Maximum depth of the partition tree.
        sp: LatinHypercube
            Initial search space.
        save_close : boolean, default=True
            If True save expanded, explored and scored fractal within a :code:`close` list.
        """

        super().__init__(openl, max_depth, save_close)

        ##############
        # PARAMETERS #
        ##############
        self.size = self.openl[0].size

        self.nu = nu

        maxd = self.max_depth
        self.layers = [[] for _ in range(maxd)]
        self.layers[0].append(self.openl[0])
        self.openl = []

        self.layers_avg = [[] for _ in range(maxd)]
        self.layers_var = [[] for _ in range(maxd)]
        self.layers_scr = [[] for _ in range(maxd)]

        self.levels_weights = np.ones(maxd, dtype=float)

        self.levels_bests = np.full(maxd, float("inf"), dtype=float)
        self.levels_bf = [None for _ in range(maxd)]
        self.best_score = float("inf")
        self.best_f = self.layers[0][0]

        self.initialized = False
        self.current_level = 0
        self.N = 1

    def add(self, c: LatinHypercubeUCB):

        self.N += len(c.losses)

        self.layers[c.level].append(c)
        self.layers_avg[c.level].append(c.mean)
        self.layers_var[c.level].append(np.sqrt(c.var))
        self.layers_scr[c.level].append(c.best_loss)

        if c.level > self.current_level:
            self.current_level = c.level
            self.levels_weights[c.level] = c.length

    def get_next(self) -> List[BaseFractal]:
        beta = np.sqrt(4 * (np.log(np.pi) + np.log(self.N)) - np.log(6 * self.nu))
        crlevel = self.current_level + 1
        vmax = float("inf")

        sel_node = [self.layers[0][0]]

        for l in range(1, crlevel):
            if len(self.layers[l]) > 0:
                mean = np.array(self.layers_avg[l], dtype=float)
                var = np.array(self.layers_var[l], dtype=float)
                scr = np.array(self.layers_scr[l], dtype=float)

                ucb = mean - beta * var

                mask = ucb > self.best_score
                ucb[mask] = mean[mask] + beta * var[mask]
                ucb[~mask] = scr[~mask]

                minidx = np.argmin(ucb)
                gscore = ucb[minidx]

                if gscore < vmax:
                    sf = self.layers[l].pop(minidx)

                    if mask[minidx] and l < (self.max_depth - 1):
                        sf.descending = True

                    sel_node.append(sf)
                    self.layers_avg[l].pop(minidx)
                    self.layers_var[l].pop(minidx)
                    self.layers_scr[l].pop(minidx)
                    vmax = gscore

                    if sf.best_loss < self.levels_bests[l]:
                        self.levels_bests[l] = sf.best_loss
                        self.levels_bf[l] = sf
                        if sf.best_loss < self.best_score:
                            self.best_score = sf.best_loss
                            self.best_f = sf

        return sel_node
