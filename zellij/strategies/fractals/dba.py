# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.stop import Stopping
from zellij.core.search_space import BaseFractal

from typing import Sequence, Optional, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.strategies.tools.tree_search import TreeSearch
    from zellij.strategies.tools.scoring import Scoring
    from zellij.strategies.tools.geometry import Direct

import numpy as np
from collections import deque

import logging

logger = logging.getLogger("zellij.DBA")


class DBA(Metaheuristic):

    """DBA

    DBA works in the unit hypercube.

    Decomposition-Based-Algorithm (DBA) is made of 5 part:

        * **Geometry** : DBA uses hyper-spheres or hyper-cubes to decompose the search-space into smaller sub-spaces in a fractal way.
        * **Tree search**: Fractals are stored in a *k-ary rooted tree*. The tree search determines how to move inside this tree.
        * **Exploration** : To explore a fractal, DBA requires an exploration algorithm.
        * **Exploitation** : At the final fractal level (e.g. a leaf of the rooted tree) DBA performs an exploitation.
        * **Scoring method**: To score a fractal, DBA can use the best score found, the median, ...

    Attributes
    ----------
    search_space : BaseFractal
        :ref:`sp` defined as a  :ref:`frac`. Contains decision
        variables of the search space.
    exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}, optional
        Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.
        If None, then the exploration phase is ignored.
    exploitation : (Metaheuristic, Stopping), optional
        Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
        of the partition tree.
        If None, then the exploitation phase is ignored.
    tree_search : TreeSearch
        Tree search algorithm applied on the partition tree.
    scoring : Scoring
        Scoring component used to compute a score of a given fractal.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a search space is in Zellij
    Tree_search : Tree search algorithm to explore and exploit the fractal tree.
    Fractal : Base class which defines what a fractal is.
    """

    def __init__(
        self,
        search_space: BaseFractal,
        tree_search: TreeSearch,
        exploration: Tuple[Metaheuristic, Union[Stopping, List[Stopping]]],
        exploitation: Optional[Tuple[Metaheuristic, Stopping]],
        scoring: Scoring,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : BaseFractal
            BaseFractal :ref:`sp`.
        tree_search : Tree_search
            Tree search algorithm applied on the partition tree.
        exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}
            Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.
        exploitation : (Metaheuristic, Stopping), optional
            Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
            of the partition tree.
        scoring : Scoring
            Function that defines how promising a space is according to sampled
            points.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super(DBA, self).__init__(search_space, verbose)
        ##############
        # PARAMETERS #
        ##############
        self.exploration = exploration
        self.exploitation = exploitation
        self.tree_search = tree_search
        self.scoring = scoring

        #############
        # VARIABLES #
        #############
        # Number of explored hypersphere
        self.n_h = 0
        self.initialized = False  # DBA status
        self.initialized_explor = False  # Exploration status
        self.initialized_exploi = False  # Exploitation status
        # Number of current computed points for exploration or exploitation
        self.current_calls = 0
        self.current_subspace = None
        # Queue to manage built subspaces
        self.subspaces_queue = deque()
        # If true do exploration, elso do exploitation
        self.do_explor = True

    @property
    def search_space(self) -> BaseFractal:
        return self._search_space

    @search_space.setter
    def search_space(self, value: BaseFractal):
        if isinstance(value, BaseFractal):
            self._search_space = value
        else:
            raise InitializationError(
                f"Searchspace in DBA must be of type BaseFractal. Got {type(value).__name__}."
            )

    @property
    def exploration(self) -> Metaheuristic:
        return self._exploration

    @exploration.setter
    def exploration(self, value: Tuple[Metaheuristic, Union[Stopping, List[Stopping]]]):
        if value:  # If there is exploration
            self._exploration = value[0]
            self.stop_explor = value[1]
        else:
            raise InitializationError(f"DBA must implement at least an exploration.")

    @property
    def stop_explor(self) -> List[Stopping]:
        return self._stop_explor  # type: ignore

    @stop_explor.setter
    def stop_explor(self, value: Union[Stopping, List[Stopping]]):
        if isinstance(value, list):
            self._stop_explor = value
        elif isinstance(value, Stopping):
            self._stop_explor = [value]
        else:  # if only 1 Stopping
            raise InitializationError(
                f"Wrong type for exploration Stopping. None, Stoppping or list of Stopping is required. Got {value}"
            )
        # If no target for stop -> self
        for s in self._stop_explor:
            if s and not s.target:
                s.target = self

    @property
    def exploitation(self) -> Optional[Metaheuristic]:
        return self._exploitation

    @exploitation.setter
    def exploitation(self, value: Optional[Tuple[Metaheuristic, Stopping]]):
        if value:
            self._exploitation = value[0]
            self.stop_exploi = value[1]
        else:
            self._exploitation = None
            self.stop_exploi = None

    @property
    def stop_exploi(self) -> Optional[Stopping]:
        return self._stop_exploi

    @stop_exploi.setter
    def stop_exploi(self, value: Optional[Stopping]):
        if value:
            self._stop_exploi = value
            self._stop_exploi.target = self
        else:  # if only 1 Stopping
            self._stop_exploi = None

    def reset(self):
        """reset

        Reset SA variables to their initial values.

        """
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False
        self.current_calls = 0
        self.current_subspace = None
        self.do_explor = True
        self.subspaces_queue = deque()

    # Add more info to ouputs
    def _add_info(self, info: dict) -> dict:
        info["level"] = self.current_subspace.level  # type: ignore
        info["father"] = self.current_subspace.father  # type: ignore
        info["f_id"] = self.current_subspace.f_id  # type: ignore
        info["c_id"] = self.current_subspace.f_id  # type: ignore

        return info

    def _explor(
        self,
        stop: Stopping,
        exploration: Metaheuristic,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        if stop():
            return [], {"algorithm": "EndExplor"}  # Exploration ending
        else:
            points, info = exploration.forward(X, Y, secondary, constraint)
            if len(points) > 0:
                info = self._add_info(info)
                self.current_calls += len(points)  # Add new computed points to counter
                return points, info  # Continue exploration
            else:
                return [], {"algorithm": "EndExplor"}  # Exploration ending

    def _exploi(
        self,
        stop: Stopping,
        exploitation: Metaheuristic,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        if stop():
            return [], {"algorithm": "EndExploi"}  # Exploitation ending
        else:
            points, info = exploitation.forward(X, Y, secondary, constraint)
            if len(points) > 0:
                info = self._add_info(info)  # add DBA information
                self.current_calls += len(points)  # Add new computed points to counter
                return points, info
            else:
                return [], {"algorithm": "EndExploi"}  # Exploitation ending

    def _new_children(self, subspace: BaseFractal) -> Sequence[BaseFractal]:
        children = subspace.create_children()
        subspace.losses = []
        subspace.solutions = []
        return children

    def _next_tree(self) -> bool:
        # continue, selected fractals
        subspaces = self.tree_search.get_next()  # Get next leaves to decompose
        for s in subspaces:
            # creates children and add them to the queue
            children = self._new_children(s)
            self.subspaces_queue.extendleft(children)
        return len(subspaces) > 0

    def _next_subspace(self) -> Optional[BaseFractal]:
        if len(self.subspaces_queue) < 1:  # if no more subspace in queue
            # if there is leaves, create children and add to queue
            if self._next_tree():
                return self._next_subspace()
            else:  # else end algorithm
                return None
        elif (
            self.current_subspace
            and self.current_subspace.level < self.tree_search.max_depth
        ):
            # add subspace to OPEN list
            self.tree_search.add(self.current_subspace)

        return self.subspaces_queue.pop()

    def _switch(self, subspace: BaseFractal) -> bool:
        # If not max level do exploration else exploitation
        if subspace.level < self.tree_search.max_depth:
            do_explor = True
        elif self.exploitation:
            do_explor = False
        else:
            do_explor = True
        return do_explor

    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of ILS.
        ILS is a local search and needs a starting point.
        X and Y must not be None.

        Parameters
        ----------
        X : list, optional
            List of points.
        Y : numpy.ndarray[float], optional
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if not self.initialized:
            self.initialized = True
            self.current_calls = 0
            # Select initial hypervolume (root) from the search tree
            subspace = self._next_subspace()
            if subspace:
                self.do_explor = self._switch(subspace)
                self.current_subspace = subspace
            else:  # Early stopping
                return [], {"algorithm": "EndDBA"}

        if not self.current_subspace:
            raise InitializationError(
                "An error occured in the initialization of current_searchspace."
            )
        elif self.do_explor:
            # select the stoping criterion
            index = min(self.current_subspace.level, len(self.stop_explor)) - 1
            stop = self.stop_explor[index]
            # continue, points, info
            if self.initialized_explor:
                self.exploration.search_space.add_solutions(X, Y)  # type: ignore
                points, info = self._explor(stop, self.exploration, X, Y)
            else:
                self.initialized_explor = True
                self.exploration.search_space = self.current_subspace
                points, info = self._explor(stop, self.exploration, None, None)

            if len(points) > 0:
                return points, info
            else:
                self.n_h += 1
                self.current_calls = 0
                self.initialized_explor = False
                self.current_subspace.score = self.scoring(self.current_subspace)
                self.exploration.reset()
                subspace = self._next_subspace()

                if subspace:
                    self.do_explor = self._switch(subspace)
                    self.current_subspace = subspace
                    return self.forward(X, Y)
                else:
                    return [], {"algorithm": "EndExplor"}

        else:
            stop = self.stop_exploi
            # continue, points, info
            if self.initialized_exploi:
                points, info = self._exploi(stop, self.exploitation, X, Y)  # type: ignore
            else:
                self.exploitation.search_space = self.current_subspace  # type: ignore
                points, info = self._exploi(stop, self.exploitation, None, None)  # type: ignore
                self.initialized_exploi = True

            if len(points) > 0:
                return points, info
            else:
                self.n_h += 1
                self.current_calls = 0
                self.initialized_exploi = False
                self.exploitation.reset()  # type: ignore
                subspace = self._next_subspace()
                if subspace:
                    self.do_explor = self._switch(subspace)
                    self.current_subspace = subspace
                    return self.forward(X, Y)
                else:
                    return [], {"algorithm": "EndExploi"}


class DBADirect(Metaheuristic):

    """DBA

    DBA works in the unit hypercube.

    Decomposition-Based-Algorithm (DBA) is made of 5 part:

        * **Geometry** : DBA uses hyper-spheres or hyper-cubes to decompose the search-space into smaller sub-spaces in a fractal way.
        * **Tree search**: Fractals are stored in a *k-ary rooted tree*. The tree search determines how to move inside this tree.
        * **Exploration** : To explore a fractal, DBA requires an exploration algorithm.
        * **Exploitation** : At the final fractal level (e.g. a leaf of the rooted tree) DBA performs an exploitation.
        * **Scoring method**: To score a fractal, DBA can use the best score found, the median, ...

    Attributes
    ----------
    search_space : Direct
        :ref:`sp` defined as a  :ref:`frac`. Contains decision
        variables of the search space.
    exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}, optional
        Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.
        If None, then the exploration phase is ignored.
    exploitation : (Metaheuristic, Stopping), optional
        Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
        of the partition tree.
        If None, then the exploitation phase is ignored.
    tree_search : TreeSearch
        Tree search algorithm applied on the partition tree.
    scoring : Scoring
        Scoring component used to compute a score of a given fractal.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a search space is in Zellij
    Tree_search : Tree search algorithm to explore and exploit the fractal tree.
    Fractal : Base class which defines what a fractal is.
    """

    def __init__(
        self,
        search_space: Direct,
        tree_search: TreeSearch,
        exploration: Tuple[Metaheuristic, Union[Stopping, List[Stopping]]],
        exploitation: Optional[Tuple[Metaheuristic, Stopping]],
        scoring: Scoring,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : Direct
            BaseFractal :ref:`sp`.
        tree_search : Tree_search
            Tree search algorithm applied on the partition tree.
        exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}
            Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.
        exploitation : (Metaheuristic, Stopping), optional
            Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
            of the partition tree.
        scoring : Scoring
            Function that defines how promising a space is according to sampled
            points.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super(DBADirect, self).__init__(search_space, verbose)
        ##############
        # PARAMETERS #
        ##############
        self.exploration = exploration
        self.exploitation = exploitation
        self.tree_search = tree_search
        self.scoring = scoring

        #############
        # VARIABLES #
        #############
        # Number of explored hypersphere
        self.n_h = 0
        self.initialized = False  # DBA status
        self.initialized_explor = False  # Exploration status
        self.initialized_exploi = False  # Exploitation status
        # Number of current computed points for exploration or exploitation
        self.current_calls = 0
        self.current_subspace = None
        # Queue to manage built subspaces
        self.subspaces_queue = deque()
        # If true do exploration, elso do exploitation
        self.do_explor = True

    @property
    def exploration(self) -> Metaheuristic:
        return self._exploration

    @exploration.setter
    def exploration(self, value: Tuple[Metaheuristic, Union[Stopping, List[Stopping]]]):
        if value:  # If there is exploration
            self._exploration = value[0]
            self.stop_explor = value[1]
        else:
            raise InitializationError(f"DBA must implement at least an exploration.")

    @property
    def stop_explor(self) -> List[Stopping]:
        return self._stop_explor  # type: ignore

    @stop_explor.setter
    def stop_explor(self, value: Union[Stopping, List[Stopping]]):
        if isinstance(value, list):
            self._stop_explor = value
        elif isinstance(value, Stopping):
            self._stop_explor = [value]
        else:  # if only 1 Stopping
            raise InitializationError(
                f"Wrong type for exploration Stopping. None, Stoppping or list of Stopping is required. Got {value}"
            )
        # If no target for stop -> self
        for s in self._stop_explor:
            if s and not s.target:
                s.target = self

    @property
    def exploitation(self) -> Optional[Metaheuristic]:
        return self._exploitation

    @exploitation.setter
    def exploitation(self, value: Optional[Tuple[Metaheuristic, Stopping]]):
        if value:
            self._exploitation = value[0]
            self.stop_exploi = value[1]
        else:
            self._exploitation = None
            self.stop_exploi = None

    @property
    def stop_exploi(self) -> Optional[Stopping]:
        return self._stop_exploi

    @stop_exploi.setter
    def stop_exploi(self, value: Optional[Stopping]):
        if value:
            self._stop_exploi = value
            self._stop_exploi.target = self
        else:  # if only 1 Stopping
            raise InitializationError(
                f"Wrong type for exploitation Stopping. None or Stoppping is required. Got {value}"
            )

    def reset(self):
        """reset

        Reset SA variables to their initial values.

        """
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False
        self.current_calls = 0
        self.current_subspace = None
        self.do_explor = True
        self.subspaces_queue = deque()

    # Add more info to ouputs
    def _add_info(self, info: dict) -> dict:
        info["level"] = self.current_subspace.level  # type: ignore
        info["father"] = self.current_subspace.father  # type: ignore
        info["f_id"] = self.current_subspace.f_id  # type: ignore
        info["c_id"] = self.current_subspace.f_id  # type: ignore

        return info

    def _explor(
        self,
        stop: Stopping,
        exploration: Metaheuristic,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        if stop():
            return [], {"algorithm": "EndExplor"}  # Exploration ending
        else:
            points, info = exploration.forward(X, Y, secondary, constraint)
            if len(points) > 0:
                info = self._add_info(info)
                self.current_calls += len(points)  # Add new computed points to counter
                return points, info  # Continue exploration
            else:
                return [], {"algorithm": "EndExplor"}  # Exploration ending

    def _exploi(
        self,
        stop: Stopping,
        exploitation: Metaheuristic,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        if stop():
            return [], {"algorithm": "EndExploi"}  # Exploitation ending
        else:
            points, info = exploitation.forward(X, Y, secondary, constraint)
            if len(points) > 0:
                info = self._add_info(info)  # add DBA information
                self.current_calls += len(points)  # Add new computed points to counter
                return points, info
            else:
                return [], {"algorithm": "EndExploi"}  # Exploitation ending

    def _new_children(self, subspace: BaseFractal) -> Sequence[BaseFractal]:
        children = subspace.create_children()
        subspace.losses = []
        subspace.solutions = []
        return children

    def _next_tree(self) -> bool:
        # continue, selected fractals
        subspaces = self.tree_search.get_next()  # Get next leaves to decompose
        return len(subspaces) > 0

    def _next_subspace(self) -> Optional[BaseFractal]:
        if len(self.subspaces_queue) < 1:  # if no more subspace in queue
            # if there is leaves, create children and add to queue
            if self._next_tree():
                return self._next_subspace()
            else:  # else end algorithm
                return None
        elif (
            self.current_subspace
            and self.current_subspace.level < self.tree_search.max_depth
        ):
            # add subspace to OPEN list
            self.tree_search.add(self.current_subspace)

        return self.subspaces_queue.pop()

    def _switch(self, subspace: BaseFractal) -> bool:
        # If not max level do exploration else exploitation
        if subspace.level < self.tree_search.max_depth:
            do_explor = True
        elif self.exploitation:
            do_explor = False
        else:
            do_explor = True
        return do_explor

    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of ILS.
        ILS is a local search and needs a starting point.
        X and Y must not be None.

        Parameters
        ----------
        X : list, optional
            List of points.
        Y : numpy.ndarray[float], optional
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if not self.initialized:
            self.initialized = True
            self.current_calls = 0
            # Select initial hypervolume (root) from the search tree
            subspace = self._next_subspace()
            if subspace:
                self.do_explor = self._switch(subspace)
                self.current_subspace = subspace
            else:  # Early stopping
                return [], {"algorithm": "EndDBA"}

        if not self.current_subspace:
            raise InitializationError(
                "An error occured in the initialization of current_searchspace."
            )
        elif self.do_explor:
            # select the stoping criterion
            index = min(self.current_subspace.level, len(self.stop_explor)) - 1
            stop = self.stop_explor[index]
            # continue, points, info
            if self.initialized_explor:
                self.exploration.search_space.add_solutions(X, Y)  # type: ignore
                points, info = self._explor(stop, self.exploration, X, Y)
            else:
                self.initialized_explor = True
                self.exploration.search_space = self.current_subspace
                points, info = self._explor(stop, self.exploration, None, None)

            if len(points) > 0:
                return points, info
            else:
                self.n_h += 1
                self.current_calls = 0
                self.initialized_explor = False
                self.current_subspace.score = self.scoring(self.current_subspace)
                self.exploration.reset()

                # creates children and add them to the queue
                if self.current_subspace < self.tree_search.max_depth:
                    children = self._new_children(self.current_subspace)
                    self.subspaces_queue.extendleft(children)

                subspace = self._next_subspace()
                if subspace:
                    self.do_explor = self._switch(subspace)
                    self.current_subspace = subspace
                    return self.forward(X, Y)
                else:
                    return [], {"algorithm": "EndExplor"}

        else:
            stop = self.stop_exploi
            # continue, points, info
            if self.initialized_exploi:
                points, info = self._exploi(stop, self.exploitation, X, Y)  # type: ignore
            else:
                self.exploitation.search_space = self.current_subspace  # type: ignore
                points, info = self._exploi(stop, self.exploitation, None, None)  # type: ignore
                self.initialized_exploi = True

            if len(points) > 0:
                return points, info
            else:
                self.n_h += 1
                self.current_calls = 0
                self.initialized_exploi = False
                self.exploitation.reset()  # type: ignore
                subspace = self._next_subspace()
                if subspace:
                    self.do_explor = self._switch(subspace)
                    self.current_subspace = subspace
                    return self.forward(X, Y)
                else:
                    return [], {"algorithm": "EndExploi"}
