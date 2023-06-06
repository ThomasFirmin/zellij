# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-01-02T12:54:33+01:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T13:01:37+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from abc import ABC, abstractmethod
from zellij.core.loss_func import (
    MPILoss,
    _MonoSynchronous_strat,
    _MonoAsynchronous_strat,
    _MultiSynchronous_strat,
    _MultiAsynchronous_strat,
)
from zellij.core.metaheuristic import Metaheuristic, AMetaheuristic
from zellij.core.search_space import ContinuousSearchspace, DiscreteSearchspace
from zellij.core.stop import Threshold

import time
import resource
import logging

logger = logging.getLogger("zellij.exp")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class Experiment(object):
    """Experiment

    Object defining the workflow of an expriment.
    It checks the stopping criterions, iterates over :code:`forward` method
    of the :ref:`meta`, and manages the different processes of the parallelization.

    Parameters
    ----------
    meta : Metaheuristic
        Metaheuristic to run.
    stop : Stopping
        Stopping criterion².

    Attributes
    ----------
    ttime : int
        Total running time of the :ref:`meta` in seconds.
    strategy : RunExperiment
        Describes how to run the experiment (parallel or not, conversion...).
    meta
    stop

    """

    def __init__(self, meta, stop):
        self.meta = meta
        self.stop = stop

        self.ttime = 0

        if isinstance(meta, AMetaheuristic) or isinstance(
            meta.search_space.loss, MPILoss
        ):
            self.strategy = RunParallelExperiment()  # type: ignore
        else:
            self.strategy = RunExperiment()  # type: ignore

    def strategy():
        doc = "Describes how to run an experiment"

        def fget(self):
            return self._strategy

        def fset(self, value):
            self._strategy = value

        def fdel(self):
            del self._strategy

        return locals()

    strategy = property(**strategy())  # type: ignore

    def run(self, X=None, Y=None):
        start = time.time()
        self.strategy.run(self.meta, self.stop, X, Y)
        end = time.time()
        self.ttime = end - start
        self.usage = resource.getrusage(resource.RUSAGE_SELF)


class AExperiment(object):
    def __init__(self, meta, stop, meta_size):
        self.meta = meta
        self.stop = stop
        self.strategy = RunAExperiment()  # type: ignore
        self.ttime = 0

        self.comm = MPI.COMM_WORLD
        self.n_process = self.comm.Get_size()
        self.meta_size = meta_size

        # if meta_size = 1, meta worker = meta master
        # -> Meta is not parallelized at algorithmic level
        # -> no competetion nor cooperation
        ################ (master meta to meta worker)
        # WORKER SPLIT #
        ################ (master loss to loss worker)

        # meta process < loss process
        color = int(self.comm.Get_rank() < meta_size)

        if meta_size > 1:
            key = self.comm.Get_rank() % meta_size
        else:
            key = self.comm.Get_rank() - 1 if self.comm.Get_rank() > 0 else 0

        self.comm_workers = self.comm.Split(color=color, key=key)

        ################# (master meta to loss worker)
        # CROSSED SPLIT #
        ################# (master loss to meta worker)

        if meta_size > 1:
            inter_color = int(
                (self.comm.Get_rank() <= meta_size and self.comm.Get_rank() > 0)
            )
        else:
            inter_color = int(
                (self.comm.Get_rank() <= meta_size and self.comm.Get_rank() >= 0)
            )

        if self.comm.Get_rank() == meta_size or self.comm.Get_rank() == 0:
            key = 0
        else:
            key = self.comm.Get_rank() % meta_size

        self.comm_crossed = self.comm.Split(color=inter_color, key=key)

        ############### (master meta to master loss)
        # TYPED SPLIT #
        ############### (loss worker to meta worker)

        if self.comm.Get_rank() == 0:
            typed_color = 0
            key = 0
        elif self.comm.Get_rank() == self.meta_size:
            typed_color = 0
            key = 1
        else:
            typed_color = 1
            work_size = self.comm.Get_size() - 2
            if self.comm.Get_rank() < work_size:
                key = self.comm.Get_rank()
            elif self.comm.Get_rank() % work_size == 0:
                key = 0
            else:
                key = self.meta_size

        self.comm_typed = self.comm.Split(color=typed_color, key=key)

        # loss workers
        if self.comm.Get_rank() >= self.meta_size:
            self.meta.search_space.loss.comm_workers = self.comm_workers
            self.meta.search_space.loss.comm_crossed = self.comm_crossed
            self.meta.search_space.loss.comm_typed = self.comm_typed
            self.meta_process = False

            self.p_name = self.meta.search_space.loss.p_name
        # meta workers
        else:
            self.meta.comm_workers = self.comm_workers
            self.meta.comm_crossed = self.comm_crossed
            self.meta.comm_typed = self.comm_typed
            self.meta_process = True

            self.p_name = self.meta.p_name
        self.comm.Barrier()

    def strategy():
        doc = "Describes how to run an experiment"

        def fget(self):
            return self._strategy

        def fset(self, value):
            self._strategy = value

        def fdel(self):
            del self._strategy

        return locals()

    strategy = property(**strategy())  # type: ignore

    def run(self, X=None, Y=None):
        start = time.time()
        self.strategy.run(
            self.meta,
            self.meta.search_space.loss,
            self.stop,
            self.comm,
            self.meta_size,
            X,
            Y,
        )
        end = time.time()
        self.ttime = end - start
        self.usage = resource.getrusage(resource.RUSAGE_SELF)


class RunExperiment(ABC):
    """RunExperiment

    Abstract class describing how to run an experiment

    """

    def __init__(self) -> None:
        super().__init__()

    def _run_forward_loss(self, meta, stop, X=None, Y=None):
        while stop():
            X, info = meta.forward(X, Y)

            if len(X) < 1:
                raise ValueError(
                    f"""
                A forward(X,Y) returned an empty list of solutions.
                """
                )
            if meta.search_space._convert_sol:
                # convert from metaheuristic space to loss space
                X = meta.search_space.converter.reverse(X)
                # compute loss
                X, Y = meta.search_space.loss(X, stop_obj=stop, **info)
                # convert from loss space to metaheuristic space
                X = meta.search_space.converter.convert(X)
            else:
                X, Y = meta.search_space.loss(X, stop_obj=stop, **info)

    def run(self, meta, stop, X=None, Y=None):
        self._run_forward_loss(meta, stop, X, Y)


class RunParallelExperiment(RunExperiment):
    """RunParallelExperiment

    Default class describing how to run a parallel experiment.

    """

    def _else_not_master(self, meta, stop):
        if meta.search_space.loss.is_worker:
            meta.search_space.loss.worker()
        else:
            logger.error(
                f"""Process of rank {meta.search_space.loss.rank}
                is undefined.
                It is not a master nor a worker."""
            )

    def run(self, meta, stop, X=None, Y=None):
        if isinstance(
            meta.search_space.loss._strategy, _MonoSynchronous_strat
        ) or isinstance(meta.search_space.loss._strategy, _MonoAsynchronous_strat):
            if meta.search_space.loss.is_master:
                self._run_forward_loss(meta, stop, X, Y)
            else:
                self._else_not_master(meta, stop)
        else:  # Have to be changed
            if isinstance(meta, AMetaheuristic):  # algorithmic parallelization
                if meta.is_master:
                    meta.master(stop_obj=stop)  # doing the outer while loop
                elif meta.is_worker:
                    meta.worker(stop_obj=stop)  # doing the inner while loop
            else:  # Asynchronous mode / iteration parallelisation
                self._run_forward_loss(meta, stop, X, Y)

            if meta.search_space.loss.is_master:
                meta.search_space.loss.master(stop_obj=stop)
            else:
                self._else_not_master(meta, stop)


class RunAExperiment(ABC):
    """RunAExperiment

    Default class describing how to run an asynchronous experiment

    """

    def run(self, meta, loss, stop, comm, meta_size, X=None, Y=None):
        if comm.Get_rank() >= meta_size:  # If process is in loss group
            if loss.master:
                # verify if the stoping criterion focuses loss or meta
                if stop.target == loss:
                    loss.dispatcher(stop)
                else:
                    loss.dispatcher()
            else:
                loss.worker()

        else:  # If process is in the meta group
            if meta.master:
                # verify if the stoping criterion focuses loss or meta
                if stop.target == meta:
                    meta.dispatcher(stop)
                else:
                    meta.dispatcher()
            else:
                meta.worker()
