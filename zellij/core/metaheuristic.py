# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T13:16:58+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from abc import abstractmethod, ABC
from zellij.core.search_space import ContinuousSearchspace, DiscreteSearchspace, Fractal
from zellij.core.loss_func import (
    MPILoss,
    _MultiAsynchronous_strat,
    _MultiSynchronous_strat,
)

import logging
from collections import deque

logger = logging.getLogger("zellij.meta")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class Metaheuristic(ABC):

    """Metaheuristic

    :ref:`meta` is a core object which defines the structure
    of a metaheuristic in Zellij. It is an abtract class.
    The :code:`forward`, describes one iteration of the :ref:`meta`, and returns
    the solutions that must be computed by the loss function.
    :code:`forwards` takes as parameters, a list of computed solutions and their objective values.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space, verbose=True):
        ##############
        # PARAMETERS #
        ##############
        self.search_space = search_space

        self.verbose = verbose

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, value):
        self._search_space = value

    @abstractmethod
    def forward(self, X, Y):
        """forward

        Abstract method describing one step of the :ref:`meta`.

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
            Dictionnary of additionnal information linked to :code:`points`.
        """
        pass

    def reset(self):
        """reset()

        reset :ref:`meta` to its initial value

        """
        pass


class ContinuousMetaheuristic(Metaheuristic):

    """ContinuousMetaheuristic

    ContinuousMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only with a continuous :ref:`sp`.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space, verbose=True):
        ##############
        # PARAMETERS #
        ##############
        self.search_space = search_space
        self.verbose = verbose

    @Metaheuristic.search_space.setter
    def search_space(self, value):
        if (
            isinstance(value, ContinuousSearchspace)
            or isinstance(value, Fractal)
            or hasattr(value, "converter")
        ):
            self._search_space = value
        else:
            raise ValueError(
                f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
            )

        if not (hasattr(value, "lower") and hasattr(value, "upper")):
            raise AttributeError(
                "Search space must have lower and upper bounds attributes, got {value}."
            )


class DiscreteMetaheuristic(Metaheuristic):

    """DiscreteMetaheuristic

    ContinuousMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only with a discrete :ref:`sp`.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space, verbose=True):
        ##############
        # PARAMETERS #
        ##############
        self.search_space = search_space

        self.verbose = verbose

    @Metaheuristic.search_space.setter
    def search_space(self, value):
        if isinstance(value, DiscreteSearchspace) or hasattr(value, "converter"):
            self._search_space = value
        else:
            raise ValueError(
                "Search space must be discrete or have a `converter` addon"
            )


class AMetaheuristic(Metaheuristic):

    """AMetaheuristic

    Asynchronous :ref:`meta` is a core object which defines the structure
    of a metaheuristic in Zellij. It is an abtract class.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    workers : int, default=None
        :code:`workers` is the number of processes dedicated to simultaneously
        compute independent :code:`forward`. If :ref:`lf` is not of type
        :code:`MPILoss`. Then the master has rank 0,
        and workers are of rank > 0. Else, master rank is equal to the number of


    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space, verbose=True):
        super().__init__(search_space, verbose)
        try:
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.p_name = MPI.Get_processor_name()

            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        if isinstance(search_space.loss, MPILoss):
            rank_start = search_space.loss.workers_size + 1
        else:
            rank_start = 0

        assert (
            rank_start + 1 <= self.p
        ), "In AMetaheuristic, invalid number of workers compared to comm size."

        self.workers = list(range(rank_start + 1, self.p))
        self.master_rank = rank_start
        self.is_master = self.rank == rank_start
        self.is_worker = self.rank > rank_start

        self.msgrecv = 0
        self.msgsend = 0

        self.send_state_lst = deque()
        self.recv_state_lst = []

        self._change_loss()

    def _change_loss(self):
        loss = self.search_space.loss
        if isinstance(loss, MPILoss):
            if loss.asynchronous:
                loss._strategy = _MultiAsynchronous_strat(loss, loss._master_rank)  # type: ignore
            else:
                loss._strategy = _MultiSynchronous_strat(loss, loss._master_rank)  # type: ignore

    @abstractmethod
    def forward(self, X, Y) -> tuple:
        """forward

        Abstract method describing one step of the :ref:`meta`.

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
            Dictionnary of additionnal information linked to :code:`points`.
        """
        pass

    @abstractmethod
    def next_state(self, state) -> list:
        pass

    @abstractmethod
    def update_state(self, state):
        pass

    @abstractmethod
    def get_state(self) -> object:
        pass

    def _msend_state(self, dest, state):
        self.msgsend += 1
        logger.debug(f"META MASTER{self.rank}, send state to {dest}\n{state}\n")
        self.comm.send(dest=dest, tag=1, obj=state)

    def _wsend_state(self, dest):
        state = self.get_state()
        self.msgsend += 1
        logger.debug(f"META WORKER{self.rank}, send state to {dest}\n{state}\n")
        self.comm.send(dest=dest, tag=2, obj=state)

    def _wsend_solution(self, dest, solution, info):
        self.msgsend += 1
        logger.debug(
            f"META WORKER{self.rank}, send solutions to {dest}\n{solution,info}\n"
        )
        self.comm.send(dest=dest, tag=2, obj=(solution, info))

    def _recv_msg(self):
        state = self.comm.recv(status=self.status)
        tag = self.status.Get_tag()
        source = self.status.Get_source()
        logger.debug(f"META WORKER{self.rank}, receive state from {source}\n{state}\n")

        if tag == 9:  # Stop
            return None, False
        elif tag == 1:  # new state
            return state, True

    def master(self, stop_obj=None):
        ctn = True
        idle = self.workers[:]
        total_workers = len(self.workers)

        states, cnt = self.next_state(None)
        self.send_state_lst.extendleft(states)

        if stop_obj:
            stopping = stop_obj
        else:
            stopping = lambda *args, **kwargs: False

        while ctn:
            ##---------MULTITHREAD---------##
            if len(idle) > 0 and len(self.send_state_lst) > 0:
                dest = idle.pop(0)
                state = self.send_state_lst.pop()
                self._msend_state(dest, state)
            elif (
                len(idle) == total_workers
                and len(self.send_state_lst) == 0
                and len(self.recv_state_lst) == 0
            ):
                ctn = False

            # receive msg from workers
            if self.comm.iprobe(tag=9):  # Stop
                state = self.comm.recv(status=self.status, tag=9)
                ctn = False
            elif stopping():
                ctn = False
                self._stop()
            elif self.comm.iprobe(tag=2):  # Received status
                state = self.comm.recv(status=self.status, tag=2)
                source = self.status.Get_source()
                idle.append(source)

                logger.debug(
                    f"META MASTER{self.rank}, received state from {source}\n{state}\n"
                )

                self.recv_state_lst.append(state)
                if len(self.send_state_lst) < 1:
                    states, cnt = self.next_state(self.recv_state_lst[:])
                    if cnt:
                        self.send_state_lst.extendleft(states)
                        self.recv_state_lst = []
                    else:
                        self._stop()
            ##---------MULTITHREAD---------##
        logger.debug(f"META MASTER {self.rank} is STOPPING")

    def _stop(self):
        """stop()

        Send a stop message to all processes.

        """
        logger.debug(f"MASTER {self.rank} sending stop message")
        for i in range(0, self.p):
            if i != self.rank:
                self.comm.send(dest=i, tag=9, obj=False)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._change_loss()
