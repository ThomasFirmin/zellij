# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-26T17:54:31+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import AMetaheuristic
from zellij.strategies.tools.scoring import Min

import numpy as np
from collections import deque

import logging

logger = logging.getLogger("zellij.DBA")


class ADBA(AMetaheuristic):

    """ADBA

    ADBA works in the unit hypercube.

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

    exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}, default=None
        Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.

    exploitation : (Metaheuristic, Stopping), default=None
        Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
        of the partition tree.

    tree_search : Tree_search
        Tree search algorithm applied on the partition tree.

    scoring : Scoring, default=Min()
        Scoring component used to compute a score of a given fractal.

    verbose : boolean, default=True
        Algorithm verbosity

    Example
    -------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold
    >>> from zellij.utils.benchmarks import himmelblau
    >>> from zellij.strategies import PHS, ILS, DBA
    >>> from zellij.strategies.tools import Hypersphere, Move_up, Distance_to_the_best
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = Hypersphere(
                    ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),
                    lf,
                    scoring=Distance_to_the_best())
    ...
    >>> explor = PHS(sp)
    >>> exploi = ILS(sp)
    >>> stop1 = Threshold(None, "current_calls", 3)  # set target to None, DBA will automatically asign it.
    >>> stop2 = Threshold(None,"current_calls", 100)  # set target to None, DBA will automatically asign it.
    >>> dba = DBA(sp, Move_up(sp,5),(explor,stop1),(exploi,stop2))
    >>> stop1.target = dba
    >>> stop2.target = dba
    ...
    >>> stop3 = Threshold(lf, "calls",5000)
    ...
    >>> exp = Experiment(dba, stop3)
    >>> exp.run()
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")

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
        search_space,
        tree_search,
        exploration=None,
        exploitation=None,
        scoring=Min(),
        verbose=True,
    ):
        """__init__(search_space, tree_search, exploration=None, exploitation=None, verbose=True)

        Initialize DBA class

        Parameters
        ----------
        search_space : Fractal
            :ref:`sp` defined as a  :ref:`frac`. Contains decision
            variables of the search space, converted to continuous and
            constrained to an EUclidean :ref:`frac`.

        tree_search : Tree_search
            Tree search algorithm applied on the partition tree.

        exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}, default=None
            Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.

        exploitation : (Metaheuristic, Stopping), default=None
            Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
            of the partition tree.

        scoring : Heuristic, default=Min()
            Function that defines how promising a space is according to sampled
            points. It is similar to the acquisition function in BO.

        verbose : boolean, default=True
            Algorithm verbosity
        """

        ##############
        # PARAMETERS #
        ##############

        super(ADBA, self).__init__(search_space, verbose)

        # Exploration and exploitation function
        self.exploration = exploration
        self.exploitation = exploitation

        #############
        # VARIABLES #
        #############

        self.tree_search = tree_search
        self.scoring = scoring

        # Number of explored hypersphere
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False

        # Number of current computed points for exploration or exploitation
        self.current_calls = 0
        self.current_subspace = self.search_space

        # Dict of fractals waiting for their score
        self.unscored = {i: {} for i in range(self.tree_search.max_depth)}

        # If true do exploration, elso do exploitation
        self.do_explor = True

    @property
    def exploration(self):
        return self._exploration

    @exploration.setter
    def exploration(self, value):
        if value:  # If there is exploration
            self._exploration = value[0]
            self.stop_explor = value[1]
        else:
            self.exploration = False
            self.stop_explor = None

    @property
    def stop_explor(self):
        return self._stop_explor

    @stop_explor.setter
    def stop_explor(self, value):
        if isinstance(value, list):
            self._stop_explor = value
        else:  # if only 1 Stopping
            self._stop_explor = [value]

        # If no target for stop -> self
        for s in self._stop_explor:
            if not s.target:
                s.target = self

    @property
    def exploitation(self):
        return self._exploitation

    @exploitation.setter
    def exploitation(self, value):
        if value:
            self._exploitation = value[0]
            self.stop_exploi = value[1]
        else:
            self._exploitation = False
            self.stop_exploi = None

    @property
    def stop_exploi(self):
        return self._stop_exploi

    @stop_exploi.setter
    def stop_exploi(self, value):
        self._stop_exploi = value
        if value is not None and not self._stop_exploi.target:
            self._stop_exploi.target = self

    def reset(self):
        """reset()

        Reset SA variables to their initial values.

        """
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False
        self.current_calls = 0
        self.current_subspace = self.search_space
        self.do_explor = True
        self.unscored = {i: {} for i in range(self.tree_search.max_depth)}

    # Add more info to ouputs
    def _add_info(self, info):
        info["level"] = self.current_subspace.level  # type: ignore
        info["father"] = self.current_subspace.father  # type: ignore
        info["f_id"] = self.current_subspace.f_id  # type: ignore
        info["c_id"] = self.current_subspace.f_id  # type: ignore

        return info

    def _explor(self, X, Y):
        # select the stoping criterion
        index = min(self.current_subspace.level, len(self.stop_explor)) - 1  # type: ignore
        stop = self.stop_explor[index]

        if stop():
            return False, None, None  # Exploration ending
        else:
            points, info = self.exploration.forward(X, Y)
            if len(points) != 0:
                info = self._add_info(info)

                self.current_calls += len(points)  # Add new computed points to counter

                return True, points, info  # Continue exploration
            else:
                return False, None, None  # Exploration ending

    def _exploi(self, X, Y):
        if self.stop_exploi():
            return False, None, None  # Exploitation ending
        else:
            points, info = self.exploitation.forward(X, Y)  # type: ignore
            if len(points) != 0:
                info = self._add_info(info)  # add DBA information

                self.current_calls += len(points)  # Add new computed points to counter

                return True, points, info
            else:
                return False, None, None  # Exploitation ending

    def _new_children(self, subspaces, subspaces_queue):
        for s in subspaces:
            children = s.create_children()
            for c in children:
                subspaces_queue.appendleft(c._essential_info())
                if c.level < self.tree_search.max_depth:
                    self.unscored[c.level][c.f_id] = c

        return subspaces_queue

    def _next_tree(self):
        subspaces_queue = deque()

        # continue, selected fractals
        cnt, subspaces = self.tree_search.get_next()  # Get next leaves to decompose
        if cnt:
            subspaces_queue = self._new_children(subspaces, subspaces_queue)

        return subspaces_queue, cnt

    def _assign_score(self, fractals):
        for f in fractals:
            if f["level"] < self.tree_search.max_depth:
                scored = self.unscored[f["level"]].pop(f["f_id"])
                scored._modify(**f)
                self.tree_search.add(scored)

    #################
    # AMeta Methods #
    #################

    def forward(self, X, Y):
        """forward(X, Y)
        Runs one step of Simulated_annealing.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if self.do_explor:
            # continue, points, info
            if self.initialized_explor:
                self.current_subspace.add_solutions(X, Y)
                cnt, points, info = self._explor(X, Y)
            else:
                cnt, points, info = self._explor(None, None)
                self.initialized_explor = True

            if cnt:
                return points, info
            else:
                self.n_h += 1
                self.current_subspace.score = self.scoring(self.current_subspace)
                return [], "explor"
        else:
            # continue, points, info
            if self.initialized_exploi:
                cnt, points, info = self._exploi(X, Y)
            else:
                cnt, points, info = self._exploi(None, None)
                self.initialized_exploi = True

            if cnt:
                return points, info
            else:
                self.n_h += 1
                return [], "exploi"

    def next_state(self, fractals):
        if fractals:
            self._assign_score(fractals)

        states, cnt = self._next_tree()
        return states, cnt

    def update_state(self, fractal):
        self.current_calls = 0
        self.current_subspace._modify(**fractal)

        # ### Do exploration
        if fractal["level"] < self.tree_search.max_depth:
            self.initialized_explor = False

            self.exploration.search_space = self.current_subspace
            self.exploration.reset()
            self.do_explor = True
            self.do_exploi = False

        # ### Do exploitation
        else:
            if self.exploitation:
                self.initialized_exploi = False

                self.exploitation.search_space = self.current_subspace  # type: ignore
                self.exploitation.reset()  # type: ignore
                self.do_explor = False
                self.do_exploi = True
            else:
                self.exploration.search_space = self.current_subspace
                self.exploration.reset()

    def get_state(self):
        return self.current_subspace._essential_info()
