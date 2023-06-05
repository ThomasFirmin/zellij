# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-26T17:54:31+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.tools.scoring import Min

import numpy as np
from queue import PriorityQueue, Queue

import logging

logger = logging.getLogger("zellij.DBA")


class DBA(Metaheuristic):

    """DBA

    Decomposition-Based-Algorithm (DBA) is made of 5 part:

        * **Geometry** : DBA uses hyper-spheres or hyper-cubes to decompose the search-space into smaller sub-spaces in a fractal way.
        * **Tree search**: Fractals are stored in a *k-ary rooted tree*. The tree search determines how to move inside this tree.
        * **Exploration** : To explore a fractal, DBA requires an exploration algorithm.
        * **Exploitation** : At the final fractal level (e.g. a leaf of the rooted tree) DBA performs an exploitation.
        * **Scoring method**: To score a fractal, DBA can use the best score found, the median, ...

    Attributes
    ----------

    search_space : Fractal
        :ref:`sp` defined as a  :ref:`frac`. Contains decision
        variables of the search space, converted to continuous and
        constrained to an Euclidean :ref:`frac`.

    exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}, default=None
        Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.

    exploitation : (Metaheuristic, Stopping), default=None
        Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
        of the partition tree.

    tree_search : Tree_search
        Tree search algorithm applied on the partition tree.

    verbose : boolean, default=True
        Algorithm verbosity


    Methods
    -------

    evaluate(hypervolumes)
        Evaluate a list of fractals using exploration and/or exploitation

    run(n_process=1)
        Runs DBA

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
    >>> stop1 = Threshold(None, "current_calls", 3)
    >>> stop2 = Threshold(None,"current_calls", 100)
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

        super(DBA, self).__init__(search_space, verbose)

        # Exploration and exploitation function

        if exploration:  # If there is exploration
            self.exploration = exploration[0]
            if type(exploration[1]) != list:  # if only 1 Stopping
                self.stop_explor = [exploration[1]]
            else:
                self.stop_explor = exploration[1]
        else:
            self.exploration = False

        if exploitation:
            self.exploitation = exploitation[0]
            self.stop_exploi = exploitation[1]
        else:
            self.exploitation = False

        # If no target for stop -> self
        for s in self.stop_explor:
            if not s.target:
                s.target = self

        if not self.stop_exploi.target:
            self.stop_exploi.target = self

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
        self.current_subspace = None

        # Queue to manage built subspaces
        self.subspaces_queue = Queue()
        # If true do exploration, elso do exploitation
        self.do_explor = True

    def reset(self):
        """reset()

        Reset SA variables to their initial values.

        """
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False
        self.current_calls = 0
        self.current_subspace = None
        self.do_explor = True
        self.subspaces_queue = Queue()

    # Add more info
    def _add_info(self, info):
        info["level"] = self.current_subspace.level
        info["id"] = self.current_subspace.id
        if isinstance(self.current_subspace.father, str):  # root
            info["father"] = self.current_subspace.father
        else:
            info["father"] = self.current_subspace.father.id

        return info

    def _explor(self, X, Y):
        index = min(self.current_subspace.level, len(self.stop_explor)) - 1
        stop = self.stop_explor[index]

        if stop():
            points, info = self.exploration.forward(X, Y)
            if len(points) != 0:
                info = self._add_info(info)

                self.current_calls += len(points)  # Add new computed points to counter

                return True, points, info  # Continue exploration
            else:
                return False, None, None  # Exploration ending
        else:
            return False, None, None  # Exploration ending

    def _exploi(self, X, Y):
        if self.stop_exploi():
            points, info = self.exploitation.forward(X, Y)
            if len(points) != 0:
                info = self._add_info(info)  # add DBA information

                self.current_calls += len(points)  # Add new computed points to counter

                return True, points, info
            else:
                return False, None, None  # Exploitation ending
        else:
            return False, None, None  # Exploitation ending

    def _new_children(self, subspace):
        subspace.create_children()
        for s in subspace.children:  # put subspaces in queue
            self.subspaces_queue.put(s)

    def _next_tree(self):
        stop, subspaces = self.tree_search.get_next()  # Get next leaves to decompose

        if stop:
            for s in subspaces:
                self._new_children(s)  # creates children and add them to the queue

        return stop

    def _next_subspace(self):
        if self.subspaces_queue.empty():  # if no more subspace in queue
            # if there is leaves, create children and add to queue
            if self._next_tree():
                return self._next_subspace()
            else:  # else end algorithm
                return False
        else:
            if (
                self.current_subspace
                and self.current_subspace.level < self.tree_search.max_depth
            ):  # add subspace to OPEN list
                self.tree_search.add(self.current_subspace)

            self.current_subspace = self.subspaces_queue.get()
            # If not max level do exploration else exploitation
            if self.current_subspace.level < self.tree_search.max_depth:
                self.exploration.search_space = self.current_subspace
                self.do_explor = True
                self.do_exploi = False
            else:
                if self.exploitation:
                    self.exploitation.search_space = self.current_subspace
                    self.do_explor = False
                    self.do_exploi = True
                else:
                    self.exploration.search_space = self.current_subspace
            return True

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

        if not self.initialized:
            self.initialized = True
            self.current_calls = 0
            # Select initial hypervolume (root) from the search tree
            stop = self._next_subspace()

            if not stop:  # Early stopping
                return [], "initialization"

        if self.do_explor:
            # continue, points, info
            if self.initialized_explor:
                cte, points, info = self._explor(X, Y)
            else:
                cte, points, info = self._explor(
                    [self.current_subspace.center], [float("inf")]
                )
                self.initialized_explor = True

            if cte:
                return points, info
            else:
                self.n_h += 1
                self.current_calls = 0
                self.initialized_explor = False
                self.exploration.reset()
                nostop = self._next_subspace()

                if nostop:
                    return self.forward(X, Y)
                else:
                    return [], "explor"

        else:
            # continue, points, info
            if self.initialized_exploi:
                cte, points, info = self._exploi(X, Y)
            else:
                cte, points, info = self._exploi(
                    [self.current_subspace.center], [float("inf")]
                )
                self.initialized_exploi = True
            if cte:
                return points, info
            else:
                self.n_h += 1
                self.current_calls = 0
                self.initialized_exploi = False
                self.exploitation.reset()
                nostop = self._next_subspace()

                if nostop:
                    return self.forward(X, Y)
                else:
                    return [], "exploi"
