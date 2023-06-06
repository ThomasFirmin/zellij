# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T13:19:21+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
import os
import shutil
import time

from abc import ABC, abstractmethod
from zellij.core.objective import Minimizer
import logging
from queue import PriorityQueue, Queue

logger = logging.getLogger("zellij.loss")

try:
    from mpi4py import MPI
except ImportError as err:
    print(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class LossFunc(ABC):

    """LossFunc

    LossFunc allows to wrap function of type :math:`f(x)=y`.
    With :math:`x` a set of hyperparameters.
    However, **Zellij** supports alternative pattern:
    :math:`f(x)=results,model` for example.
    Where:

        * :math:`results` can be a `list <https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>`__ or a `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`__. Be default the first element of the list or the dictionary is considered as the loss vale.
        * :math:`model` is optionnal, it is an object with a save() method. (e.g. a neural network from Tensorflow)

    You must wrap your function so it can be used in Zellij by adding
    several features, such as calls count, saves, parallelization, historic...

    Attributes
    ----------
    model : function
        Function of type :math:`f(x)=y` or :math:`f(x)=results,model. :math:`x`
        must be a solution. A solution can be a list of float, int...
        It can also be of mixed types...
    objective : Objective, default=Minimizer
        Objectve object determines what and and how to optimize.
        (minimization, maximization, ratio...)
    best_score : float
        Best score found so far.
    best_point : list
        Best solution found so far.
    all_scores : float
        Historic of all evaluated scores.
    all_solutions : float
        Historic of all evaluated solutions.
    calls : int
        Number of loss function calls

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc
    """

    def __init__(
        self,
        model,
        objective=Minimizer,
        historic=False,
        save=False,
        verbose=True,
        only_score=False,
        kwargs_mode=False,
        default=None,
    ):
        """__init__(model, save=False)

        Parameters
        ----------
        model : Callable
            Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
            of hyperparameters for example.
            And :math:`y` can be a single value, a list, a tuple, or a dict,
            containing the loss value and other optionnal information.
            It can also be of mixed types, containing, strings, float, int...
        objective : Objective, default=Minimizer
            An :code:`Objective` object determines what the optimization problem is.
            If :code:`objective` is :code:`Maximizer`, then the first argument
            of the object, list, tuple or dict, returned by the :code:`__call__`
            function will be maximized.
        historic : bool, optionnal
            If True, then all evaluation are saved within the :ref:`lf` object.
            Otherwise, only the best solution found so far is saved.
            All solutions and information can be saved by using the :code:`save`
            parameter;
        save : str, optionnal
            If a :code:`str` is given, then outputs will be saved in :code:`save`.
        verbose : bool, default=False
            Verbosity of the loss function.
        only_score : bool, default=False
            If True, then only the score of evaluated solutions are saved.
            Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
            saved.
        kwargs_mode : bool, default=False
            If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
            the names of the :ref:`var` within the :ref:`sp`.
        default : dict, optionnal
            Dictionnary of defaults arguments, kwargs, to pass to the loss function.
            They are not affected by any :ref:`metaheuristic` or other methods.

        """
        ##############
        # PARAMETERS #
        ##############

        self.model = model
        if isinstance(objective, type):
            self.objective = objective()
        else:
            self.objective = objective

        self.historic = historic
        self.save = save
        self.only_score = only_score
        self.kwargs_mode = kwargs_mode

        self.verbose = verbose

        #############
        # VARIABLES #
        #############

        self.best_score = float("inf")
        self.best_point = None

        self.all_scores = []
        self.all_solutions = []

        self.calls = 0

        self.labels = []

        if isinstance(self.save, str):
            self.folder_name = self.save
        else:
            self.folder_name = f"{self.model.__class__.__name__}_zlj_save"

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.file_created = False

        self._init_time = time.time()

        self.default = default

    @abstractmethod
    def _save_model(self, *args):
        """_save_model()

        Private abstract method to save a model.
        Be carefull, to be exploitable, the initial loss func must be of form
        :math:`f(x) = (y, model)`, :math:`y` are the results of the evaluation of :math:`x`
        by :math:`f`. :math:`model` is optional, if you want to save the best model
        found (e.g. a neural network) you can return the model.
        However the model must have a "save" method with a filename.
        (e.g. model.save(filename)).

        """
        pass

    @abstractmethod
    def __call__(self, X, **kwargs):
        pass

    def _compute_loss(self, point):
        if self.kwargs_mode:
            new_kwargs = {key: value for key, value in zip(self.labels, point)}
            if self.default:
                new_kwargs.update(self.default)

            start = time.time()
            res, trained_model = self._build_return(self.model(**new_kwargs))
            end = time.time()

            res["eval_time"] = end - start
            res["start_time"] = start - self._init_time
            res["end_time"] = end - self._init_time
        else:
            start = time.time()
            if self.default:
                lossv = self.model(point, **self.default)
            else:
                lossv = self.model(point)

            res, trained_model = self._build_return(lossv)
            end = time.time()

            res["eval_time"] = end - start
            res["start_time"] = start - self._init_time
            res["end_time"] = end - self._init_time

        return res, trained_model

    def _create_file(self, x, *args):
        """create_file(x, *args)

        Create a save file:

        Structure:

            foldername
            | model # if sav = True in LossFunc, contains model save
              | model_save
            | outputs # Contains loss function outputs
              | file_1.csv
              | ...
            | plots # if save = True while doing .show(), contains plots
              | plot_1.png
              | ...

        Parameters
        ----------
        solution : list
            Needs a solution to determine the header of the save file

        *args : list[label]
            Additionnal info to add after the score/evaluation of a point.

        """

        # Create a valid folder
        try:
            os.makedirs(self.folder_name)
            created = False
        except FileExistsError as error:
            created = True

        i = 0
        while created:
            try:
                nfolder = f"{self.folder_name}_{i}"
                os.mkdir(nfolder)
                created = False
                logger.warning(
                    f"WARNING: Folder {self.folder_name} already exists, results will be saved at {nfolder}"
                )
                self.folder_name = nfolder
            except FileExistsError as error:
                i += 1

        # Create ouputs folder
        self.outputs_path = os.path.join(self.folder_name, "outputs")
        os.mkdir(self.outputs_path)

        self.model_path = os.path.join(self.folder_name, "model")
        os.mkdir(self.model_path)

        self.plots_path = os.path.join(self.folder_name, "plots")
        os.mkdir(self.plots_path)

        # Additionnal header for the outputs file
        if len(args) > 0:
            suffix = "," + ",".join(str(e) for e in args)
        else:
            suffix = ""

        # Create base outputs file for loss func
        self.loss_file = os.path.join(self.outputs_path, "all_evaluations.csv")

        # Determine header
        if len(self.labels) != len(x):
            logger.warning(
                "WARNING: Labels are of incorrect size, it will be replaced in the save file header"
            )
            for i in range(len(x)):
                self.labels.append(f"attribute{i}")

        with open(self.loss_file, "w") as f:
            if self.only_score:
                f.write("objective\n")
            else:
                f.write(",".join(str(e) for e in self.labels) + suffix + "\n")

        print(f"INFO: Results will be saved at: {os.path.abspath(self.folder_name)}")

        self.file_created = True

    def _save_file(self, x, **kwargs):
        """_save_file(x, **kwargs)

        Private method to save informations about an evaluation of the loss function.

        Parameters
        ----------
        x : list
            Solution to save.
        **kwargs : dict, optional
            Other information to save linked to x.
        """

        if not self.file_created:
            self._create_file(x, *list(kwargs.keys()))

        # Determine if additionnal contents must be added to the save
        if len(kwargs) > 0:
            suffix = ",".join(str(e) for e in kwargs.values())
        else:
            suffix = ""

        # Save a solution and additionnal contents
        with open(self.loss_file, "a+") as f:
            if self.only_score:
                f.write(f"{kwargs['objective']}\n")
            else:
                f.write(",".join(str(e) for e in x) + "," + suffix + "\n")

    # Save best found solution
    def _save_best(self, x, y):
        """_save_best(x, y)

        Save point :code:`x` with score :code:`y`, and verify if this point is the best found so far.

        Parameters
        ----------
        x : list
            Set of hyperparameters (a solution)
        y : {float, int}
            Loss value (score) associated to x.

        """

        self.calls += 1

        # Save best
        if y < self.best_score:
            self.best_score = y
            self.best_point = list(x)[:]

        # historic
        if self.historic:
            self.all_solutions.append(list(x)[:])
            self.all_scores.append(y)

    def _build_return(self, r):
        """_build_return(r)

        This method builds a unique return according to the outputs of the loss function

        Parameters
        ----------
        r : {list, float, int}
            Returns of the loss function

        Returns
        -------
        rd : dict
            Dictionnary mapping outputs from the loss function

        model : object
            Model object with a 'save' method

        """

        # Separate results and model
        if isinstance(r, tuple):
            if len(r) > 1:
                results, model = r
            else:
                results, model = r, False
        else:
            results, model = r, False

        return self.objective(results), model

    def get_best(self, n_process=1, idx=None):
        if self.historic and idx is not None:
            best_idx = np.argpartition(self.all_scores[idx], n_process)
            best = [self.all_solutions[i] for i in best_idx[:n_process]]
            min = [self.all_scores[i] for i in best_idx[:n_process]]
        else:
            best = self.best_point
            min = self.best_score

        return best, min

    def reset(self):
        """reset()

        Reset all attributes of :code:`LossFunc` at their initial values.

        """

        self.best_score = float("inf")
        self.best_point = None
        self.best_argmin = None

        self.all_scores = []
        self.all_solutions = []

        self.calls = 0

        self.labels = []

        if isinstance(self.save, str):
            self.folder_name = self.save
        else:
            self.folder_name = f"{self.model.__class__.__name__}_zlj_save"

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.file_created = False

        self._init_time = time.time()


class SerialLoss(LossFunc):

    """SerialLoss

    SerialLoss adds methods to save and evaluate the original loss function.

    Methods
    -------

    __call__(X, filename='', **kwargs)
        Evaluate a list X of solutions with the original loss function.

    _save_model(score, source)
        See LossFunc, save a model according to its score and the worker rank.

    See Also
    --------
    Loss : Wrapper function
    LossFunc : Inherited class
    MPILoss : Distributed version of LossFunc
    """

    def __init__(
        self,
        model,
        objective=Minimizer,
        historic=False,
        save=False,
        verbose=True,
        only_score=False,
        kwargs_mode=False,
        default=None,
    ):
        """__init__(model, historic=False, save=False, verbose=True)

        Initialize SerialLoss.

        """

        super().__init__(
            model,
            objective,
            historic,
            save,
            verbose,
            only_score,
            kwargs_mode,
            default,
        )

    def __call__(self, X, stop_obj=None, **kwargs):
        """__call__(model, **kwargs)

        Evaluate a list X of solutions with the original loss function.

        Parameters
        ----------
        X : list
            List of solutions to evaluate. be carefull if a solution is a list X must be a list of lists.
        **kwargs : dict, optional
            Additionnal informations to save before the score.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """

        res = []

        for x in X:
            outputs, model = self._compute_loss(x)

            res.append(outputs["objective"])

            # Saving
            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(outputs["objective"], model)

                # Save score and solution into a file
                self._save_file(x, **outputs)

            self._save_best(x, outputs["objective"])
        return X, res

    def _save_model(self, score, trained_model):
        # Save model into a file if it is better than the best found one
        if score < self.best_score:
            save_path = os.path.join(
                self.model_path, f"{self.model.__class__.__name__}_best"
            )
            if hasattr(trained_model, "save") and callable(
                getattr(trained_model, "save")
            ):
                os.system(f"rm -rf {save_path}")
                trained_model.save(save_path)
            else:
                logger.error("Model/loss function does not have a method called `save`")
                exit()


class MPILoss(LossFunc):
    def __init__(
        self,
        model,
        objective=Minimizer,
        historic=False,
        save=False,
        verbose=True,
        only_score=False,
        kwargs_mode=False,
        asynchronous=False,
        workers=None,
        default=None,
    ):
        """MPILoss

        MPILoss adds method to dynamically distribute the evaluation
        of multiple solutions within a distributed environment, where a version of
        `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__
        is available.

        Attributes
        ----------
        model : Callable
            Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
            of hyperparameters for example.
            And :math:`y` can be a single value, a list, a tuple, or a dict,
            containing the loss value and other optionnal information.
            It can also be of mixed types, containing, strings, float, int...
        objective : Objective, default=Minimizer
            An :code:`Objective` object determines what the optimization problem is.
            If :code:`objective` is :code:`Maximizer`, then the first argument
            of the object, list, tuple or dict, returned by the :code:`__call__`
            function will be maximized.
        historic : bool, optionnal
            If True, then all evaluation are saved within the :ref:`lf` object.
            Otherwise, only the best solution found so far is saved.
            All solutions and information can be saved by using the :code:`save`
            parameter;
        save : str, optionnal
            If a :code:`str` is given, then outputs will be saved in :code:`save`.
        verbose : bool, default=False
            Verbosity of the loss function.
        only_score : bool, default=False
            If True, then only the score of evaluated solutions are saved.
            Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
            saved.
        kwargs_mode : bool, default=False
            If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
            the names of the :ref:`var` within the :ref:`sp`.
        asynchronous : bool, default=False
            Asynchronous mode. If True then :code:`__call__` will return
            the result from an evluation of a solution assoon as it receives a
            result from a worker. Other solutions, are still being evaluated in
            background.
            If False, then :code:`__call__` will return all results from all
            solutions passed, once all of them have been evaluated.
        workers : int, optionnal
            Number of workers among the total number of processes spawned by
            MPI. At least, one process is dedicated to the master.
        comm : MPI_COMM_WORLD
            All created processes and their communication context are grouped in comm.
        status : MPI_Status
            Data structure containing information about a received message.
        rank : int
            Process rank
        p : int
            comm size
        master : boolean
            If True the process is the master, else it is the worker.

        See Also
        --------
        Loss : Wrapper function
        LossFunc : Inherited class
        SerialLoss : Basic version of LossFunc
        """

        super().__init__(
            model,
            objective,
            historic,
            save,
            verbose,
            only_score,
            kwargs_mode,
            default,
        )

        #################
        # MPI VARIABLES #
        #################

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

        if workers:
            self.workers_size = workers
        else:
            if asynchronous:
                self.workers_size = self.p - 1
            else:
                self.workers_size = self.p - 1

        self.recv_msg = 0
        self.sent_msg = 0

        self._personnal_folder = os.path.join("tmp_wks", f"worker{self.rank}")

        self.asynchronous = asynchronous

        self._pqueue_mode = False
        self.pqueue = Queue()

        # list of idle workers
        self.idle = list(range(1, self.workers_size + 1))

        # loss worker rank : (point, id, info, source)
        self.p_historic = {
            i: [None, None, None, None] for i in range(1, self.workers_size + 1)
        }  # historic of points sent to workers

        self._strategy = None  # set to None for definitio issue

        self.is_master = self.rank == 0
        self.is_worker = self.rank != 0

        # Property, defines parallelisation strategy
        self._master_rank = 0  # type: ignore

    def _master_rank():
        def fget(self):
            return self.__master_rank

        def fset(self, value):
            self.__master_rank = value

            if self.rank == value:
                self.is_master = True

            if self.asynchronous:
                self._strategy = _MonoAsynchronous_strat(self, value)
            else:
                self._strategy = _MonoSynchronous_strat(self, value)

        def fdel(self):
            del self.__master_rank

        return locals()

    _master_rank = property(**_master_rank())  # type: ignore

    def __call__(self, X, stop_obj=None, **kwargs):
        new_x, score = self._strategy(X, stop_obj=stop_obj, **kwargs)  # type: ignore

        return new_x, score

    def master(self, pqueue=None, stop_obj=None):
        """master()

        Evaluate a list :code:`X` of solutions with the original loss function.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """

        print(f"Master of rank {self.rank} Starting")

        # if there is a stopping criterion
        if stop_obj:
            stopping = stop_obj
        else:
            stopping = lambda *args: True
        # second stopping criterion determine by the parallelization itself
        stop = True

        if pqueue:
            self._pqueue_mode = True
            self.pqueue = pqueue

        while stopping() and stop:
            # Send solutions to workers
            # if a worker is idle and if there are solutions
            while not self.comm.iprobe() and (
                len(self.idle) > 0 and not self.pqueue.empty()
            ):
                self._send_point(self.idle, self.pqueue, self.p_historic)

            if self.comm.iprobe():
                msg = self.comm.recv(status=self.status)
                stop = self._parse_message(
                    msg, self.pqueue, self.p_historic, self.idle, self.status
                )

        if self._pqueue_mode:
            if not stopping():
                print(f"MASTER{self.rank}, calls:{self.calls} |!| STOPPING |!|")
                self._stop()
        else:
            print(f"MASTER{self.rank}, calls:{self.calls} |!| STOPPING |!|")
            self._stop()

    def _parse_message(self, msg, pqueue, historic, idle, status):
        tag = status.Get_tag()
        source = status.Get_source()

        # receive score
        if tag == 1:
            (
                point,
                outputs,
                point_id,
                point_info,
                point_source,
            ) = self._recv_score(msg, source, idle, historic)

            # Save score and solution into the object
            self._save_best(point, outputs["objective"])
            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(outputs["objective"], source)

                # Save score and solution into a file
                self._save_file(point, **outputs, **point_info)

            stop = self._process_outputs(
                point, outputs, point_id, point_info, point_source
            )
        # receive a point to add to the queue
        elif tag == 2:
            stop = self._recv_point(msg, source, pqueue)
        # error: abort
        else:
            logger.error(
                f"Unknown message tag, got {tag} from process {source}. Processes will abort"
            )
            stop = True

        return stop

    # send point from master to worker
    def _send_point(self, idle, pqueue, historic):
        next_point = pqueue.get()
        dest = idle.pop()
        historic[dest] = next_point
        print(
            f"MASTER {self.rank} sending point to WORKER {dest}.\n Remaining points in queue: {pqueue.qsize()}"
        )
        self.comm.send(dest=dest, tag=0, obj=next_point[0])

    # receive a new point to put in the point queue. (from a forward)
    def _recv_point(self, msg, source, pqueue):
        print(f"MASTER {self.rank} receiving point from PROCESS {source}")
        msg.append(source)
        pqueue.put(msg)

        return True

    # receive score from workers
    def _recv_score(self, msg, source, idle, historic):
        print(
            f"MASTER {self.rank} receiving score from WORKER {source} : {msg}, historic : {historic[source]}"
        )
        point = historic[source][0][:]
        point_id = historic[source][1]
        point_info = historic[source][2].copy()
        point_source = historic[source][3]
        historic[source] = [None, None, None, None]

        outputs = msg

        idle.append(source)

        return point, outputs, point_id, point_info, point_source

    def _process_outputs(self, point, outputs, id, info, source):
        return self._strategy._process_outputs(point, outputs, id, info, source)  # type: ignore

    def _stop(self):
        """stop()

        Send a stop message to all workers.

        """
        print(f"MASTER {self.rank} sending stop message")
        for i in range(0, self.p):
            if i != self.rank:
                self.comm.send(dest=i, tag=9, obj=False)

    def _save_model(self, score, source):
        """_save_model(score, source)

        Be carefull, to be exploitable, the initial loss func must be of form
        :math:`f(x) = (y, model)`, :math:`y` are the results of the evaluation of :math:`x`
        by :math:`f`. :math:`model` is optional, if you want to save the best model
        found (e.g. a neural network) you can return the model.
        However the model must have a "save" method with a filename.
        (e.g. model.save(filename)).

        Parameters
        ----------

        score : int
            Score corresponding to the solution saved by the worker.
        source : int
            Worker rank which evaluate a solution and return score

        """

        # Save model into a file if it is better than the best found one
        if score < self.best_score:
            master_path = os.path.join(
                self.model_path, f"{self.model.__class__.__name__}_best"
            )
            worker_path = os.path.join("tmp_wks", f"worker{self.rank}")

            if os.path.isdir(worker_path):
                os.system(f"rm -rf {master_path}")
                os.system(f"cp -rf {worker_path} {master_path}")

    def _wsave_model(self, model):
        if hasattr(model, "save") and callable(getattr(model, "save")):
            os.system(f"rm -rf {self._personnal_folder}")
            model.save(self._personnal_folder)
        else:
            logger.error("The model does not have a method called save")

    def worker(self):
        """worker()

        Initialize worker. While it does not receive a stop message,
        a worker will wait for a solution to evaluate.

        """

        print(f"WORKER {self.rank} starting")

        stop = True

        while stop:
            print(f"WORKER {self.rank} receving message")
            # receive message from master
            msg = self.comm.recv(source=self._master_rank, status=self.status)  # type: ignore
            tag = self.status.Get_tag()
            source = self.status.Get_source()

            if tag == 9:
                print(f"WORKER{self.rank} |!| STOPPING |!|")
                stop = False

            elif tag == 0:
                print(f"WORKER {self.rank} receved a point, {msg}")
                point = msg
                # Verify if a model is returned or not
                outputs, model = self._compute_loss(point)

                # Save the model using its save method
                if model and self.save:
                    print(f"WORKER {self.rank} saving model")
                    self._wsave_model(model)

                # Send results
                print(f"WORKER {self.rank} sending {outputs} to {self._master_rank}")
                self.comm.send(dest=self._master_rank, tag=1, obj=outputs)  # type: ignore

            else:
                print(f"WORKER {self.rank} unknown tag, got {tag}")


class _Parallel_strat:
    def __init__(self, loss, master_rank):
        super().__init__()
        self.master_rank = master_rank
        try:
            self.comm = MPI.COMM_WORLD
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        # counter for computed point
        self._computed = 0

        self._lf = loss


# Mono Synchrone -> Save score return list of score
class _MonoSynchronous_strat(_Parallel_strat):
    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        pqueue = Queue()
        for id, x in enumerate(X):
            pqueue.put((x, id, kwargs, None))
        self.y = [None] * len(X)
        self._lf.master(pqueue, stop_obj=stop_obj)
        return X, self.y[:]

    # Executed by master when it receives a score from a worker
    # Here Meta master and loss master are the same process, so shared memory
    def _process_outputs(self, point, outputs, id, info, source):
        self.y[id] = outputs["objective"]
        self._computed += 1

        if self._computed < len(self.y):
            print(
                f"COMPUTED POINT {self._computed}/{len(self.y)}, calls:{self._lf.calls}"
            )
            return True
        else:
            print(
                f"STOP COMPUTED POINT {self._computed}/{len(self.y)}, calls:{self._lf.calls}"
            )
            self._computed = 0
            return False


class _MonoAsynchronous_strat(_Parallel_strat):
    def __init__(self, loss, master_rank):
        super().__init__(loss, master_rank)
        self._current_points = {}
        self._computed_point = (None, None)
        self._lf._pqueue_mode = True

    # send a point to loss master
    def _send_to_master(self, point, **kwargs):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self._lf.pqueue.put((point, id, kwargs, None))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        # send point, point ID and point info
        for point in X:
            self._send_to_master(point, **kwargs)  # send point

        self._lf.master(stop_obj=stop_obj)

        new_x, y = (
            self._computed_point[0].copy(),
            self._computed_point[1].copy(),
        )
        self._computed_point = (None, None)
        return [new_x], [y["objective"]]

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        self._computed_point = (point, outputs)
        return False


# Multi Synchronous -> Save score into groups, return groups to meta worker
class _MultiSynchronous_strat(_Parallel_strat):
    # send a point to loss master
    def _send_to_master(self, point):
        self.comm.send(dest=self.master_rank, tag=2, obj=point)

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        # Early stopping
        stop = True

        # score
        y = [None] * len(X)

        # send point, point ID and point info
        for i, p in enumerate(X):
            self._send_to_master((p, i, kwargs))  # send point

        nb_recv = 0
        while nb_recv < len(X) and stop:
            # receive score from loss
            print(f"call() of rank :{self.rank} receiveing message")
            msg = self.comm.recv(source=self.master_rank, status=self.status)
            tag = self.status.Get_tag()
            source = self.status.Get_source()

            if tag == 9:
                print(f"call() of rank :{self.rank} |!| STOPPING |!|")
                stop = False

            elif tag == 2:
                print(f"call() of rank :{self.rank} received a score")
                # id / score
                y[msg[1]] = msg[0]
                nb_recv += 1

        return X, y

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        self.comm.send(dest=source, tag=2, obj=(outputs, id))
        return True


# Multi Asynchronous -> Return unique score vers worker meta
class _MultiAsynchronous_strat(_Parallel_strat):
    def __init__(self, loss, master_rank):
        super().__init__(loss, master_rank)
        self._current_points = {}

    # send a point to loss master
    def _send_to_master(self, point, infos):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self.comm.send(dest=self.master_rank, tag=2, obj=(point, id, infos))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        # send point, point ID and point info
        for p in X:
            self._send_to_master(p, kwargs)  # send point

        # receive score from loss
        print(f"call() of rank :{self.rank} receiveing message")
        msg = self.comm.recv(source=self.master_rank, status=self.status)
        tag = self.status.Get_tag()
        source = self.status.Get_source()

        if tag == 9:
            print(f"call() of rank :{self.rank} |!| STOPPING |!|")
            stop = False
            new_x, y = None, None
        elif tag == 2:
            print(f"call() of rank :{self.rank} received a score")
            # id / score
            new_x, y = self._current_points.pop(msg[1]), msg[0]

        print("RETURN: ", [new_x], [y])
        return [new_x], [y]

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        self.comm.send(dest=source, tag=2, obj=(outputs, id))
        return True


# Wrap different loss functions
def Loss(
    model=None,
    objective=Minimizer,
    historic=False,
    save=False,
    verbose=True,
    MPI=False,
    only_score=False,
    kwargs_mode=False,
    workers=None,
    default=None,
):
    """Loss

    Wrap a function of type :math:`f(x)=y`. See :ref:`lf` for more info.

    Parameters
    ----------
    model : Callable
        Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
        of hyperparameters for example.
        And :math:`y` can be a single value, a list, a tuple, or a dict,
        containing the loss value and other optionnal information.
        It can also be of mixed types, containing, strings, float, int...
    objective : Objective, default=Minimizer
        An :code:`Objective` object determines what the optimization problem is.
        If :code:`objective` is :code:`Maximizer`, then the first argument
        of the object, list, tuple or dict, returned by the :code:`__call__`
        function will be maximized.
    historic : bool, optionnal
        If True, then all evaluation are saved within the :ref:`lf` object.
        Otherwise, only the best solution found so far is saved.
        All solutions and information can be saved by using the :code:`save`
        parameter;
    save : str, optionnal
        If a :code:`str` is given, then outputs will be saved in :code:`save`.
    verbose : bool, default=False
        Verbosity of the loss function.
    only_score : bool, default=False
        If True, then only the score of evaluated solutions are saved.
        Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
        saved.
    kwargs_mode : bool, default=False
        If True, then solutions are passed as kwargs to :ref:`lf`. Keys are
        the names of the :ref:`var` within the :ref:`sp`.
    MPI : {False, asynchronous, synchronous}, optional
        Wrap the function with :code:`MPILoss` if True, with SerialLoss else.
    workers : int, optionnal
        Number of workers among the total number of processes spawned by
        MPI. At least, one process is dedicated to the master.
    default : dict, optionnal
        Dictionnary of defaults arguments, kwargs, to pass to the loss function.
        They are not affected by any :ref:`metaheuristic` or other methods.

    Returns
    -------
    wrapper : LossFunc
        Wrapped original function

    Examples
    --------
    >>> import numpy as np
    >>> from zellij.core.loss_func import Loss
    >>> @Loss(save=False, verbose=True)
    ... def himmelblau(x):
    ...   x_ar = np.array(x)
    ...   return np.sum(x_ar**4 -16*x_ar**2 + 5*x_ar) * (1/len(x_ar))
    >>> print(f"Best solution found: f({himmelblau.best_point}) = {himmelblau.best_score}")
    Best solution found: f(None) = inf
    >>> print(f"Number of evaluations:{himmelblau.calls}")
    Number of evaluations:0
    >>> print(f"All evaluated solutions:{himmelblau.all_solutions}")
    All evaluated solutions:[]
    """
    if model:
        return SerialLoss(model)
    else:

        def wrapper(model):
            if MPI:
                if MPI == "asynchronous":
                    return MPILoss(
                        model,
                        objective,
                        historic,
                        save,
                        verbose,
                        only_score,
                        kwargs_mode,
                        workers=workers,
                        asynchronous=True,
                        default=default,
                    )
                elif MPI == "synchronous":
                    return MPILoss(
                        model,
                        objective,
                        historic,
                        save,
                        verbose,
                        only_score,
                        kwargs_mode,
                        workers=workers,
                        asynchronous=False,
                        default=default,
                    )
                else:
                    raise NotImplementedError(
                        f"""
                    {MPI} parallelisation is not implemented.
                    Use MPI='asynchronous', 'synchronous', or False, for non
                    distributed loss function.
                    """
                    )
            else:
                return SerialLoss(
                    model,
                    objective,
                    historic,
                    save,
                    verbose,
                    only_score,
                    kwargs_mode,
                    default=default,
                )

        return wrapper


class MockModel(object):
    """MockModel

    This object allows to replace your real model with a costless object,
    by mimicking different available configurations in Zellij.
    ** Be carefull: This object does not replace any Loss wrapper**

    Parameters
    ----------
    outputs : dict, default={'o1',lambda *args, **kwargs: np.random.random()}
        Dictionnary containing outputs name (keys)
        and functions to execute to obtain outputs.
        Pass *args and **kwargs to these functions when calling this MockModel.

    verbose : bool
        If True print information when saving and __call___.

    return_format : string
        Output format. It can be :code:`'dict'` > :code:`{'o1':value1,'o2':value2,...}`
        or :code:`list`>:code:`[value1,value2,...]`.

    return_model : boolean
        Return :code:`(outputs, MockModel)` if True. Else, :code:`outputs`.

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc

    Examples
    --------
    >>> from zellij.core.loss_func import MockModel, Loss
    >>> mock = MockModel()
    >>> print(mock("test", 1, 2.0, param1="Mock", param2=True))
    I am Mock !
        ->*args: ('test', 1, 2.0)
        ->**kwargs: {'param1': 'Mock', 'param2': True}
    ({'o1': 0.3440051802032301},
    <zellij.core.loss_func.MockModel at 0x7f5c8027a100>)
    >>> loss = Loss(save=True, verbose=False)(mock)
    >>> print(loss([["test", 1, 2.0, "Mock", True]], other_info="Hi !"))
    I am Mock !
        ->*args: (['test', 1, 2.0, 'Mock', True],)
        ->**kwargs: {}
    I am Mock !
        ->saving in MockModel_zlj_save/model/MockModel_best/i_am_mock.txt
    [0.7762604280531996]
    """

    def __init__(
        self,
        outputs={"o1": lambda *args, **kwargs: np.random.random()},
        return_format="dict",
        return_model=True,
        verbose=True,
    ):
        super().__init__()
        self.outputs = outputs
        self.return_format = return_format
        self.return_model = return_model
        self.verbose = verbose

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(filepath, "i_am_mock.txt")
        with open(filename, "wb") as f:
            if self.verbose:
                print(f"\nI am Mock !\n\t->saving in {filename}")

    def __call__(self, *args, **kwargs):
        if self.verbose:
            print(f"\nI am Mock !\n\t->*args: {args}\n\t->**kwargs: {kwargs}")

        if self.return_format == "dict":
            part_1 = {x: y(*args, **kwargs) for x, y in self.outputs.items()}
        elif self.return_format == "list":
            part_1 = [y(*args, **kwargs) for x, y in self.outputs.items()]
        else:
            raise NotImplementedError(
                f"return_format={self.return_format} is not implemented"
            )
        if self.return_model:
            return part_1, self
        else:
            return part_1
