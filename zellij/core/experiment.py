# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.stop import Stopping
    from zellij.core.metaheuristic import Metaheuristic
    import numpy as np

from zellij.core.errors import InitializationError, UnassignedProcess

from zellij.core.loss_func import LossFunc, MPILoss
from zellij.core.backup import AutoSave

import time
import os
import pickle

import logging

logger = logging.getLogger("zellij.exp")


class RunExperiment(ABC):
    """
    Abstract class describing how to run an experiment

    exp : Experiment
        A given :code:`Experiment` object.

    """

    def __init__(self, exp):
        super().__init__()
        self.exp = exp

        # current solutions
        self._cX = None
        self._cY = None
        self._cSecondary = None
        self._cConstraint = None

    def _run_forward_loss(
        self,
        meta: Metaheuristic,
        loss: LossFunc,
        stop: Stopping,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        """
        Runs one step of a :ref:`meta`, and describes how to compute solutions.

        Parameters
        ----------
        meta : :ref:`meta
            A given :ref:`meta` with a :code:`forward` method.
        loss : LossFunc
            A given :ref:`lf`.
        stop : :ref:`stop`
            :ref:`stop` object.
        X : list, optional
            List of computed solutions. Can be used to initialize a :ref:`meta` with initial solutions.
        Y : np.ndarray, optional
            List of computed loss values. Can be used to initialize a :ref:`meta` with initial loss values.
        secondary : np.ndarray, optional
            Array of floats, secondary objective values. See :ref:`lf`.
        constraint : np.ndarray, optional
            List of constraints values. See :ref:`lf`.

        Returns
        -------
        list[list, list, list, bool]
            Returns computed solutions :code:`X`, with computed loss values :code:`Y`, and
            computed :code:`constraints` values if available. :code:`cnt` a bool determining
            if optimization process can continue.
            If False, then a problem occured in the computation of a :code:`forward` in
            :ref:`meta`, which returned an empty list of solutions.
        """
        cnt = True  # continue optimization

        X, info = meta.forward(X, Y, secondary, constraint)
        if len(X) < 1:
            return None, None, None, None, False
        else:
            if meta.search_space._do_convert:
                # convert from metaheuristic space to loss space
                X = meta.search_space.reverse(X)
                # compute loss
                X, Y, secondary, constraint = loss(X, stop_obj=stop, **info)
                # if meta return empty solutions
                if X:
                    # convert from loss space to metaheuristic space
                    X = meta.search_space.convert(X)
                else:
                    cnt = False  # stop optimization
            else:
                X, Y, secondary, constraint = loss(X, stop_obj=stop, **info)
                # if meta return empty solutions
                if X is None:
                    cnt = False  # stop optimization

            return X, Y, secondary, constraint, cnt

    def run(
        self,
        meta: Metaheuristic,
        loss: LossFunc,
        stop: Stopping,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        """
        Optimization loop.

        Parameters
        ----------
        meta : :ref:`meta
            A given :ref:`meta` with a :code:`forward` method.
        stop : :ref:`stop`
            :ref:`stop` object.
        X : list, optional
            List of computed solutions. Can be used to initialize a :ref:`meta` with initial solutions.
        Y : np.ndarray, optional
            List of computed loss values. Can be used to initialize a :ref:`meta` with initial loss values.
        secondary : np.ndarray, optional
            Array of floats, secondary objective values. See :ref:`lf`.
        constraint : np.ndarray, optional
            List of constraints values. See :ref:`lf`.

        Raises
        ------
        ValueError
            Raise an error if a problem occured during a  :ref:`forward` of a :ref:`meta`.
        """
        cnt = True
        while not stop() and cnt:
            if self.exp.verbose:
                print(f"STATUS: {stop}", end="\r", flush=True)
            X, Y, secondary, constraint, cnt = self._run_forward_loss(
                meta, loss, stop, X, Y, secondary, constraint
            )
            if X is None:
                logger.warning(
                    "A forward(_,_,_,_) returned an empty list of solutions."
                )
                cnt = False
        if self.exp.verbose:
            print(f"ENDING: {stop}")


class RunParallelExperiment(RunExperiment):
    """RunParallelExperiment

    Default class describing how to run a parallel experiment.

    """

    def _else_not_master(self, meta: Metaheuristic, loss: MPILoss, stop: Stopping):
        """
        Defines what a worker do.

        Parameters
        ----------
        meta : Metaheuristic
            :ref:`meta`
        stop : Stopping
            :ref:`stop`

        """
        if loss.is_worker:  # type: ignore
            loss.worker()  # type: ignore
        else:
            raise UnassignedProcess(
                f"Role of process of rank {loss.rank} is undefined."  # type: ignore
            )

    def run(
        self,
        meta: Metaheuristic,
        loss: MPILoss,
        stop: Stopping,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        """
        Optimization loop.

        Parameters
        ----------
        meta : :ref:`meta
            A given :ref:`meta` with a :code:`forward` method.
        loss : MPILoss
            A MPILoss, :ref:`lf`.
        stop : :ref:`stop`
            :ref:`stop` object.
        X : list, optional
            List of computed solutions. Can be used to initialize a :ref:`meta` with initial solutions.
        Y : np.ndarray, optional
            List of computed loss values. Can be used to initialize a :ref:`meta` with initial loss values.
        secondary : np.ndarray, optional
            Array of floats, secondary objective values. See :ref:`lf`.
        constraint : np.ndarray, optional
            List of constraints values. See :ref:`lf`.


        Raises
        ------
        TypeError
            Raise an error if an unknown parallelisation configuration is detected.
        """

        cnt = True

        # Iteration parallelization
        if isinstance(loss, MPILoss):
            if loss.is_master:
                while not stop() and cnt:
                    if self.exp.verbose:
                        print(f"STATUS: {stop}", end="\r", flush=True)

                    X, Y, secondary, constraint, cnt = self._run_forward_loss(
                        meta, loss, stop, X, Y, secondary, constraint
                    )
                if self.exp.verbose:
                    print(f"ENDING: {stop}")
                logger.debug(f"MASTER{loss.rank}, sending STOP")
                loss._stop()
            else:
                self._else_not_master(meta, loss, stop)
        else:
            raise TypeError(
                "The LossFunc for RunParallelExperiment, must be a MPILoss."
            )


class Experiment:
    """
    Object defining the workflow of an expriment.
    It checks the stopping criterion, iterates over :code:`forward` method
    of the :ref:`meta`, and manages the different processes of the parallelization.

    Parameters
    ----------
    meta : Metaheuristic
        :ref:`meta` to run.
    stop : Stopping
        :ref:`stop` criterion.
    save : str, optionnal
        If a :code:`str` is given, then outputs will be saved in :code:`save`.
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

    def __init__(
        self,
        meta: Metaheuristic,
        loss: LossFunc,
        stop: Stopping,
        save: Optional[str] = None,
        backup_interval: Optional[int] = None,
        verbose: bool = True,
    ):
        ##############
        # PARAMETERS #
        ##############
        self.meta = meta
        self.loss = loss
        self.stop = stop
        self.save = save
        self.backup_interval = backup_interval
        self.verbose = verbose

        #############
        # VARIABLES #
        #############
        self.backup_folder = ""
        self.folder_created = False
        self.ttime = 0

        if self.save:
            if isinstance(self.loss, MPILoss):
                if self.loss.is_master:
                    self._create_folder()
            else:
                self._create_folder()

            self.meta._save = self.save

    @property
    def loss(self) -> LossFunc:
        return self._loss

    @loss.setter
    def loss(self, value: LossFunc):
        if isinstance(value, LossFunc):
            self._loss = value
            self._loss.labels = [v.label for v in self.meta.search_space.variables]  # type: ignore

            # Define run strategies
            if isinstance(value, MPILoss):
                self.strategy = RunParallelExperiment(self)
            else:
                self.strategy = RunExperiment(self)

        else:
            raise InitializationError(f"`loss` must be a `LossFunc`, got {type(value)}")

    @property
    def save(self) -> Optional[str]:
        return self._save

    @save.setter
    def save(self, value: Optional[str]):
        # All evaluations are 'seen' by loss
        # saving is made by loss
        self.loss.save = value
        self._save = value

    @property
    def strategy(self) -> RunExperiment:
        return self._strategy

    @strategy.setter
    def strategy(self, value: RunExperiment):
        self._strategy = value

    def run(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        start = time.time()
        self.strategy.run(self.meta, self.loss, self.stop, X, Y, secondary, constraint)
        end = time.time()
        self.ttime += end - start
        # self.usage = resource.getrusage(resource.RUSAGE_SELF)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # del state["usage"]
        return state

    def _create_folder(self):
        """create_foler
        Create a save folder:
        """

        # Create a valid folder
        self.loss._create_folder()

        if self.backup_interval:
            self.backup_folder = os.path.join(self.save, "backup")  # type: ignore
            # Create a valid folder
            try:
                os.makedirs(self.backup_folder)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Backup folder already exists, got {self.backup_folder}"
                )

        self.folder_created = True

    def backup(self):
        logger.info(f"INFO: Saving BACKUP in {self.backup_folder}")
        pickle.dump(self, open(os.path.join(self.backup_folder, "experiment.p"), "wb"))  # type: ignore
