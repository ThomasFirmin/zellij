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
from zellij.utils import AutoSave


import time
import resource
import os
import logging
import pickle

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
        Stopping criterion.
    save : {boolean, str}, default=False
        Creates a backup regularly
    backup_interval : int, default=300
        Interval of time (in seconds) between each backup.

    Attributes
    ----------
    ttime : int
        Total running time of the :ref:`meta` in seconds.
    strategy : RunExperiment
        Describes how to run the experiment (parallel or not, conversion...).
    meta
    stop

    """

    def __init__(self, meta, stop, save=False, backup_interval=300):
        self.meta = meta
        self.stop = stop
        self.save = save
        self.backup_interval = backup_interval
        self.backup_folder = ""
        self.folder_created = False

        self.ttime = 0

        if isinstance(meta, AMetaheuristic) or isinstance(
            meta.search_space.loss, MPILoss
        ):
            self.strategy = RunParallelExperiment(self)  # type: ignore
        else:
            self.strategy = RunExperiment(self)  # type: ignore

        if self.save:
            if isinstance(self.meta.search_space.loss, MPILoss):
                if self.meta.search_space.loss.is_master:
                    self._create_folder()
            else:
                self._create_folder()

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, value):
        self.meta.search_space.loss.save = value
        self._save = value

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy = value

    def run(self, X=None, Y=None):
        start = time.time()
        self.strategy.run(self.meta, self.stop, X, Y)
        end = time.time()
        self.ttime += end - start
        # self.usage = resource.getrusage(resource.RUSAGE_SELF)

    def __getstate__(self):
        state = self.__dict__.copy()
        # del state["usage"]
        return state

    def _create_folder(self):
        """create_foler()

        Create a save folder:

        """

        # Create a valid folder
        try:
            os.makedirs(self.save)
        except FileExistsError as error:
            raise FileExistsError(f"Folder already exists, got {self.save}")

        self.backup_folder = os.path.join(self.save, "backup")

        # Create a valid folder
        try:
            os.makedirs(self.backup_folder)
        except FileExistsError as error:
            raise FileExistsError(
                f"backup_folder already exists, got {self.backup_folder}"
            )
        self.folder_created = True

    def backup(self):
        logger.info(f"INFO: Saving BACKUP in {self.backup_folder}")
        print("SAVIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIING")
        pickle.dump(self, open(os.path.join(self.backup_folder, "experiment.p"), "wb"))  # type: ignore


class RunExperiment(ABC):
    """RunExperiment

    Abstract class describing how to run an experiment

    """

    def __init__(self, exp):
        super().__init__()
        self.exp = exp

        # current solutions
        self._cX = None
        self._cY = None

    def _run_forward_loss(self, meta, stop, X=None, Y=None):
        # useful after unpickling
        if self._cX is None and self._cY is None:
            self._cX, self._cY = X, Y

        while stop():
            X, info = meta.forward(self._cX, self._cY)

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

            self._cX, self._cY = X, Y

    def run(self, meta, stop, X=None, Y=None):
        print(f"I AM MASTER: {meta.search_space.loss.rank}")
        autosave = AutoSave(self.exp)
        try:
            self._run_forward_loss(meta, stop, X, Y)
        finally:
            autosave.stop()


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
                autosave = AutoSave(self.exp)
                try:
                    self._run_forward_loss(meta, stop, X, Y)
                finally:
                    autosave.stop()
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
