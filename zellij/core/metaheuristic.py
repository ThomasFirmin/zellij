# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T13:16:58+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from abc import abstractmethod, ABC
from zellij.core.search_space import ContinuousSearchspace, DiscreteSearchspace, Fractal

import logging

logger = logging.getLogger("zellij.meta")


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

    stop : Stopping
        :ref:`stop` criterion.

    save : boolean, optional
        If True save results into a file

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

    def search_space():
        doc = "The search_space property."

        def fget(self):
            return self._search_space

        def fset(self, value):
            self._search_space = value

        def fdel(self):
            del self._search_space

        return locals()

    search_space = property(**search_space())

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

    stop : Stopping
        :ref:`stop` criterion.

    save : boolean, optional
        If True save results into a file

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

    def search_space():
        doc = "Search space property."

        def fget(self):
            return self._search_space

        def fset(self, value):
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

        def fdel(self):
            del self._search_space

        return locals()

    search_space = property(**search_space())


class DiscreteMetaheuristic(Metaheuristic):

    """DiscreteMetaheuristic

    ContinuousMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only with a discrete :ref:`sp`.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    stop : Stopping
        :ref:`stop` criterion.

    save : boolean, optional
        If True save results into a file

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

    def search_space():
        doc = "Search space property."

        def fget(self):
            return self._search_space

        def fset(self, value):
            if isinstance(value, DiscreteSearchspace) or hasattr(value, "converter"):
                self._search_space = value
            else:
                raise ValueError(
                    "Search space must be discrete or have a `converter` addon"
                )

        def fdel(self):
            del self._search_space

        return locals()

    search_space = property(**search_space())


class AMetaheuristic(ABC):

    """AMetaheuristic

    Asynchronous :ref:`meta` is a core object which defines the structure
    of a metaheuristic in Zellij. It is an abtract class.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    stop : Stopping
        :ref:`stop` criterion.

    save : boolean, optional
        If True save results into a file

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

    def __init__(self, search_space, workers, verbose=True):
        ##############
        # PARAMETERS #
        ##############
        self.search_space = search_space
        self.verbose = verbose

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
            rank_start = search_space.loss.workers + 1
            self.is_master = self.rank == rank_start
            self.is_worker = self.rank > rank_start
            self.master_rank = rank_start
        else:
            self.is_master = self.rank == 0
            self.is_worker = self.rank != rank_start
            self.master_rank = 0

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

    @abstractmethod
    def save(self, path):
        """save
        Method saving actual state of the metaheuristic
        """
        pass

    @abstractmethod
    def load(self, path, loss):
        """save
        Method loading a previously saved metaheuristic.

        """
        pass

    @abstractmethod
    def master(self, stop_obj=None):
        """master
        Master process
        """
        pass

    @abstractmethod
    def worker(self, stop_obj=None):
        """worker
        Worker process
        """
        pass

    def reset(self):
        """reset()

        reset :ref:`meta` to its initial value

        """
        pass
