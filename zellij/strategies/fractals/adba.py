# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-26T17:41:07+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic

import numpy as np
from queue import PriorityQueue, Queue
import time

import logging

logger = logging.getLogger("zellij.DBA")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class ADBA(Metaheuristic):

    """ADBA

    Asynchronous Decomposition-Based-Algorithm (DBA) is made of 5 part:

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

        exploration : {(Metaheuristic, Stopping), (Metaheuristic, list[Stopping])}, default=None
            Tuple made of a :ref:`meta` and one or a list of :ref:`stop` used to sample inside each subspaces.

        exploitation : (Metaheuristic, Stopping), default=None
            Tuple made of a :ref:`meta` and a :ref:`stop` applied on a subspace at the last level
            of the partition tree.

        tree_search : Tree_search
            Tree search algorithm applied on the partition tree.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super(ADBA, self).__init__(search_space, verbose)

        self.comm_workers = None
        self.comm_crossed = None
        self.comm_typed = None
        self.master = False

        self.com = MPI.COMM_WORLD
        self.status = MPI.Status()

        self.computed_fractals = 0
        self.computed_points = 0

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

        # Number of explored hypersphere
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False
        # Number of current computed points for exploration or exploitation
        self.current_calls = 0
        self.total_calls = 0
        self.loss_idle = False

        # Queue to manage built subspaces
        self.subspaces_queue = Queue()
        # If true do exploration, elso do exploitation
        self.do_explor = True

    def comm_crossed():
        doc = "The comm_crossed property."

        def fget(self):
            return self._comm_crossed

        def fset(self, value):
            if value:
                self.crossed_size = value.Get_size()
            self._comm_crossed = value

        def fdel(self):
            del self._comm_crossed

        return locals()

    comm_crossed = property(**comm_crossed())

    def comm_workers():
        doc = "The comm_workers property."

        def fget(self):
            return self._comm_workers

        def fset(self, value):
            if value:
                self.msgrecv = 0
                self.msgsnd = 0

                self.workers_size = value.Get_size()
                self.rank = value.Get_rank()
                if self.rank == 0:
                    self.master = True
                    self.p_name = f"meta_master"
                else:
                    self.master = False
                    self.p_name = f"meta_worker{self.rank}"
            self._comm_workers = value

        def fdel(self):
            del self._comm_workers

        return locals()

    comm_workers = property(**comm_workers())

    def comm_typed():
        doc = "The comm_typed property."

        def fget(self):
            return self._comm_typed

        def fset(self, value):
            if value:
                self.typed_size = value.Get_size()
            self._comm_typed = value

        def fdel(self):
            del self._comm_typed

        return locals()

    comm_typed = property(**comm_typed())

    def reset(self):
        """reset()

        Reset SA variables to their initial values.

        """
        self.n_h = 0
        self.initialized = False
        self.initialized_explor = False
        self.initialized_exploi = False
        self.current_calls = 0
        self.total_calls = 0
        self.do_explor = True
        self.subspaces_queue = Queue()

    # Add more info
    def _add_info(self, info):
        info["level"] = self.search_space.level
        info["id"] = self.search_space.id
        return info

    def _explor(self, X, Y):
        index = min(self.search_space.level, len(self.stop_explor)) - 1
        stop = self.stop_explor[index]

        if stop():
            points, info = self.exploration.forward(X, Y)
            if len(points) != 0:
                info = self._add_info(info)

                self.current_calls += len(points)  # Add new computed points

                return True, points, info  # Continue exploration
            else:
                return False, None, None  # Exploration ending
        else:
            return False, None, None  # Exploration ending

    def _exploi(self, X, Y):

        if self.stop_exploi():
            points, info = self.exploitation.forward(X, Y)
            if len(points) != 0:
                info = self._add_info(info)

                self.current_calls += len(points)

                return True, points, info
            else:
                return False, None, None
        else:
            return False, None, None

    def _new_children(self, subspace):
        self.search_space.create_children(subspace, self.subspaces_queue)

    def _next_tree(self):
        stop, subspaces = self.tree_search.get_next()

        if stop:
            for s in subspaces:
                self._new_children(s)

        return stop

    def _next_subspace(self):
        if self.subspaces_queue.empty():  # if no more subspace in queue
            # Build new children
            if self._next_tree():  # if there are children
                return self._next_subspace()
            else:  # else end algorithm
                return False, None
        else:
            return True, self.subspaces_queue.get()

    def _mparse_message(self, msg, evaluated_fractals, fractals, idle, status):
        self.msgrecv += 1
        tag = status.Get_tag()
        source = status.Get_source()
        # print(f"META MASTER{self.rank}, recv {tag}:{msg} from {source}")

        # receive fractal score
        if tag == 3:
            evaluated_fractals.append(fractals[source])
            evaluated_fractals[-1].score = msg
            idle.append(source)
            return True
        # receive calls
        elif tag == 5:
            self.total_calls = msg[0]
            self.loss_idle = msg[1]
            self.computed_fractals += 1
            return True
        # master stop
        elif tag == 8:
            return False
        else:
            raise NotImplementedError(f"Unknown message tag {tag}, {msg}")

    def _wparse_message(self, msg, status, loss, X):
        self.msgrecv += 1
        tag = status.Get_tag()
        source = status.Get_source()
        # print(f"META WORKER{self.rank}, recv {tag}:{msg} from {source}")

        # receive loss value
        if tag == 1:
            self.computed_points += 1
            loss[msg[0]] = msg[1]
            return True
        # receive fractal
        elif tag == 2:
            if self.search_space.level < self.tree_search.max_depth:
                self.exploration.search_space = (
                    self.exploration.search_space._modify(msg)
                )
                self.do_explor = True
            else:
                self.do_explor = False
                self.exploitation.search_space._modify(msg)
                self.exploitation.search_space = (
                    self.exploration.search_space._modify(msg)
                )

            return True
        # receive stop
        elif tag == 4:
            return False
        # receive calls
        elif tag == 5:
            self.total_calls = msg
        else:
            raise NotImplementedError(f"Unknown message tag {tag}")

    def _msend_message(self, dest, type, content):
        self.msgsnd += 1
        # print(f"META MASTER{self.rank}, send {type}:{content} to {dest}")
        # send fractal to workers
        if type == 2:
            self.comm_workers.send(dest=dest, tag=2, obj=content)
        # send stop to workers
        elif type == 4:
            self.comm_workers.send(dest=dest, tag=4, obj=content)
        # master stop
        elif type == 8:
            self.comm_typed.send(dest=dest, tag=8, obj=False)
        else:
            raise NotImplementedError(f"Unknown message type got {type}")

    def _wsend_message(self, dest, type, content):
        self.msgsnd += 1
        # print(f"META WORKER{self.rank}, send {type}:{content} to {dest}")
        # send point to loss
        if type == 0:
            self.comm_crossed.send(dest=dest, tag=0, obj=content)
        # send fractal score to master
        elif type == 3:
            self.computed_fractals += 1
            self.comm_workers.send(dest=dest, tag=3, obj=content)

    def dispatcher(self, stop_obj=None):

        stop = True
        idle = list(range(1, self.workers_size))
        fractals = {}
        evaluated_fractals = []
        total_workers = self.workers_size - 1
        new_fractal = True

        start = time.time()

        if stop_obj:
            stoping = stop_obj
        else:
            stoping = lambda *args: True

        while stoping() and stop:
            if len(idle) > 0 and new_fractal:
                dest = idle.pop(0)

                # continue, fractal
                cnt, fractals[dest] = self._next_subspace()
                if cnt:
                    # send fractal
                    self._msend_message(dest, 2, fractals[dest])
                else:
                    new_fractal = False

            # receive msg from workers
            if self.comm_workers.iprobe():
                msg = self.comm_workers.recv(status=self.status)
                stop = self._mparse_message(
                    msg, evaluated_fractals, fractals, idle, self.status
                )

            # add evaluated fractals to tree
            while evaluated_fractals:
                new_fractal = True
                current = evaluated_fractals.pop()
                if current.level < self.tree_search.max_depth:
                    self.tree_search.add(current)

            # receive calls from loss master
            if self.comm_typed.iprobe():
                msg = self.comm_typed.recv(source=1, status=self.status)
                stop = self._mparse_message(
                    msg, evaluated_fractals, fractals, idle, self.status
                )

            if self.loss_idle and not new_fractal:
                stop = False
            if len(idle) == total_workers:
                stop = False

        self._stop(idle, stoping)

    def worker(self):
        stop = True
        X, Y = None, None

        # replace heuristic /!\ TO BE MODIFIED
        min = float("inf")

        # Waiting for 1st message from master
        msg = self.comm_workers.recv(source=0, status=self.status)
        cnt = self._wparse_message(msg, self.status, None, X)

        while stop:

            # continue
            if cnt:
                X, info = self.forward(X, Y)

                if len(X) > 0:
                    loss = np.zeros(len(X))

                    # send point, point ID and point info
                    for i, p in enumerate(X):
                        # send point
                        self._wsend_message(0, 0, [p, i, info])

                    nb_recv = 0
                    while nb_recv < len(X):
                        # receive score from loss
                        msg = self.comm_crossed.recv(
                            source=0, status=self.status
                        )
                        cnt = self._wparse_message(msg, self.status, loss, X)
                        if cnt:
                            nb_recv += 1
                            broke = False
                        else:
                            nb_recv = len(X)
                            broke = True

                    if broke:
                        # print(f"META WORKER{self.rank} BROKE")
                        msg = self.comm_workers.recv(
                            source=0, status=self.status
                        )
                        cnt = self._wparse_message(msg, self.status, None, None)
                    else:
                        # REPLACE HEURISTIC /!\ TO BE MODIFIED
                        currenttmin = np.min(loss)
                        if currenttmin < min:
                            min = currenttmin

                        Y = loss

                else:
                    # scoring, only send minium /!\TO BE MODIFIED
                    self._wsend_message(0, 3, min)
                    min = float("inf")
                    # Waiting for a new fractal
                    msg = self.comm_workers.recv(source=0, status=self.status)
                    cnt = self._wparse_message(msg, self.status, None, None)

            else:
                stop = False

        # print(f"META WORKER{self.rank} |!| STOPPING |!|")

    def _stop(self, idle, stoping):

        """stop()

        Send a stop message to all workers.

        """

        for i in range(1, self.workers_size):
            self.comm_workers.send(dest=i, tag=4, obj=False)
        for i in range(1, self.typed_size):
            self.comm_typed.send(dest=i, tag=8, obj=False)

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
                cte, points, info = self._explor(X, Y)
            else:
                cte, points, info = self._explor(
                    [self.search_space.center], [float("inf")]
                )
                self.initialized_explor = True

            if cte:
                return points, info
            else:
                self.current_calls = 0
                self.initialized_explor = False
                self.exploration.reset()

                return [], "explor"

        else:
            # continue, points, info
            if self.initialized_exploi:
                cte, points, info = self._exploi(X, Y)
            else:
                cte, points, info = self._exploi(
                    [self.search_space.center], [float("inf")]
                )
                self.initialized_exploi = True
            if cte:
                return points, info
            else:
                self.current_calls = 0
                self.initialized_exploi = False
                self.exploitation.reset()

                return [], "exploi"
