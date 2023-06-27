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
from zellij.core.backup import AutoSave


import time
import os
import logging
import pickle

logger = logging.getLogger("zellij.exp")


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
        if value:
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
        self._cConstraint = None

    def _run_forward_loss(self, meta, stop, X=None, Y=None, constraint=None):
        cnt = True
        if self._cX is None and self._cY is None:
            self._cX = X
            self._cY = Y
            self._cConstraint = constraint

        X, info = meta.forward(self._cX, self._cY)
        if len(X) < 1:
            self._cX, self._cY, self._cConstraint = None, None, None
            return None, None, None, cnt
        else:
            if meta.search_space._convert_sol:
                # convert from metaheuristic space to loss space
                X = meta.search_space.converter.reverse(X)
                # compute loss
                X, Y, constraint = meta.search_space.loss(X, stop_obj=stop, **info)
                # convert from loss space to metaheuristic space
                if X is None:
                    cnt = False
                else:
                    X = meta.search_space.converter.convert(X)
            else:
                X, Y, constraint = meta.search_space.loss(X, stop_obj=stop, **info)
                if X is None:
                    cnt = False

            self._cX = X
            self._cY = Y
            self._cConstraint = constraint

            return X, Y, constraint, cnt

    def run(self, meta, stop, X=None, Y=None, constraint=None):
        autosave = AutoSave(self.exp)
        try:
            cnt = True
            while not stop() and cnt:
                X, Y, constraint, cnt = self._run_forward_loss(
                    meta, stop, X, Y, constraint
                )
                if X is None and Y is None:
                    raise ValueError(
                        f"""
                    A forward(X,Y) returned an empty list of solutions.
                    """
                    )
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

    def run(self, meta, stop, X=None, Y=None, constraint=None):
        if isinstance(
            meta.search_space.loss._strategy, _MonoSynchronous_strat
        ) or isinstance(meta.search_space.loss._strategy, _MonoAsynchronous_strat):
            if meta.search_space.loss.is_master:
                autosave = AutoSave(self.exp)
                try:
                    while not stop():
                        X, Y, constraint, cnt = self._run_forward_loss(
                            meta, stop, X, Y, constraint=constraint
                        )
                finally:
                    autosave.stop()
            else:
                self._else_not_master(meta, stop)
        elif isinstance(meta, AMetaheuristic):  # algorithmic parallelization
            if meta.is_master:
                meta.master(stop_obj=stop)  # managing states

            elif meta.is_worker:  # Receiving states and updating meta state
                state, cnt = meta._recv_msg()  # type: ignore
                meta.update_state(state)

                while cnt and not stop():
                    X, Y, constraint, cnt = self._run_forward_loss(
                        meta, stop, X, Y, constraint=constraint
                    )
                    if X is None and Y is None and cnt:
                        meta._wsend_state(meta.master_rank)
                        state, cnt = meta._recv_msg()  # type: ignore
                        if cnt:
                            meta.update_state(state)

            if isinstance(
                meta.search_space.loss._strategy, _MultiSynchronous_strat
            ) or isinstance(meta.search_space.loss._strategy, _MultiAsynchronous_strat):
                if meta.search_space.loss.is_master:
                    meta.search_space.loss.master(stop_obj=stop)
                else:
                    self._else_not_master(meta, stop)
        else:
            raise TypeError("Unknown experiment configuration")
