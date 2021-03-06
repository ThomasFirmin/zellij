# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   ThomasFirmin
# @Last modified time: 2022-05-03T15:44:13+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import numpy as np
import os
import shutil
from abc import abstractmethod
import enlighten
import zellij.utils.progress_bar as pb

import logging

logger = logging.getLogger("zellij.loss")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class LossFunc(object):

    """LossFunc

    LossFunc allows to wrap function of type :math:`f(x)=y`. With :math:`x` a set of hyperparameters.
    However, **Zellij** supports alternative pattern: :math:`f(x)=results,model` for example.
    Where:

    * :math:`results` can be a `list <https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>`_ or a `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_. The first element of the list must be the loss value. If the return is a dictionary, the loss value must have the key *"score"*.
    * :math:`model` is optionnal, it is an object with a *save()* method. (e.g. a neural network from Tensorflow)

    You must wrap your function so it can be used in Zellij by adding several features,\
     such as calls count, saves, parallelization, historic...

    Attributes
    ----------
    model : function
        Function of type :math:`f(x)=y` or :math:`f(x)=results,model. :math:`x` must be a solution. A solution can be a list of float, int... It can also be of mixed types...
    best_score : float
        Best score found so far.
    best_sol : list
        Best solution found so far.
    best_argmin : int
        Index of the best solution found so far.
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

    def __init__(self, model, save=False, verbose=True):

        """__init__(model, save=False)

        Parameters
        ----------
        model : function
            Function of type :math:`f(x)=y` or :math:`f(x)=results,model. :math:`x` must be a solution. A solution can be a list of float, int... It can also be of mixed types...
        save : string
            Filename where to save the best found model and the historic of the loss function.
            Only one model is saved for memory issues. Be carefull, if you want to save a model,
            the object that you loss function returns, must have a "save" method with a filename parameter. (e.g. model.save(filename)).

        """
        ##############
        # PARAMETERS #
        ##############

        self.model = model
        self.save = save
        self.verbose = verbose
        #############
        # VARIABLES #
        #############

        self.best_score = float("inf")
        self.best_sol = None
        self.best_argmin = None

        self.all_scores = []
        self.all_solutions = []

        self.calls = 0
        # Must be private, ?? voir
        self.new_best = False

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

        if self.verbose:
            self.manager = enlighten.get_manager()
        else:
            self.manager = enlighten.get_manager(stream=None, enabled=False)

    def build_bar(self, total):
        """build_bar(total)

        build_bar is a method to build a progress bar.
        It is a purely aesthetic feature to get info on the execution.
        You can deactivate it, with `verbose=False`.

        Parameters
        ----------
        total : int
            Length of the progress bar.

        """

        if self.verbose:
            self.lf_pb = pb.calls_counter_inside(self.manager, total)
            self.best_pb = pb.best_found(self.manager, self.best_score)

    def close_bar(self):
        """close_bar()

        Delete the progress bar.

        """
        if self.verbose:
            self.lf_pb.close()
            self.best_pb.close()

    @abstractmethod
    def _save_model(self, *args):
        """ _save_model()

        Private abstract method to save a model. Be carefull, to be exploitable, the initial loss func must be of form :math:`f(x) = (y, model)`\
         `y` are the results of the evaluation of `x` by `f`. `model` is optional, if you want to save the best model found (e.g. a neural network)\
         you can return the model. However the model must have a "save" method with a filename. (e.g. model.save(filename)).

        """
        pass

    @abstractmethod
    def __call__(self, X, **kwargs):
        pass

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
            list of additionnal labels to add before the score/evaluation of a point.

        """

        # Create a valid folder
        try:
            os.mkdir(self.folder_name)
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
            f.write(
                ",".join(str(e) for e in self.labels) + ",loss" + suffix + "\n"
            )

        logger.info(
            f"INFO: Results will be saved at: {os.path.abspath(self.folder_name)}"
        )

        self.file_created = True

    def _save_file(self, x, y, **kwargs):

        """_save_file(x, y, **kwargs)

        Private method to save informations about an evaluation of the loss function.

        Parameters
        ----------
        x : list
            Solution to save.
        y : list
            Evaluation of the solution by the loss function;
        filename : str
            Name of the file created by a Metaheuristic, where to save informations.
        **kwargs : dict, optional
            Other informations to save after the score.
        """

        if not self.file_created:
            self._create_file(x, *list(kwargs.keys()))

        # Determine if additionnal contents must be added to the save
        if len(kwargs) > 0:
            suffix = "," + ",".join(str(e) for e in kwargs.values())
        else:
            suffix = ""

        # Save a solution and additionnal contents
        with open(self.loss_file, "a+") as f:
            f.write(",".join(str(e) for e in x) + "," + str(y) + suffix + "\n")

    # Save best found solution
    def _save_best(self, x, y):
        """_save_best(x, y)

        Save point x with score y, and verify if this point is the best found so far.

        Parameters
        ----------
        x : list
            Set of hyperparameters (a solution)
        y : {float, int}
            Loss value (score) associated to x.

        """

        # historic
        self.all_solutions.append(list(x)[:])
        self.all_scores.append(y)

        # Save best
        if y < self.best_score:
            self.best_score = y
            self.best_sol = list(x)[:]
            self.best_argmin = len(self.all_scores)

            self.new_best = True

        if self.verbose:
            self.lf_pb.update(1)
            self.best_pb.update(
                "      Current score:{:.3f} | Best score:{:.3f}".format(
                    y, self.best_score
                ),
                color="white",
            )

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

        rd = {}

        # Separate results and model
        if isinstance(r, tuple):
            if len(r) > 1:
                results, model = r
            else:
                results, model = r, False
        else:
            results, model = r, False

        # Separate results
        if isinstance(results, int) or isinstance(results, float):
            rd["score"] = results
        elif isinstance(results, dict):
            rd = results
            rd["score"] = r.values()[0]
        elif isinstance(results, list):
            rd["score"] = results[0]
            for i, j in enumerate(results):
                label = f"return{i}"
                rd[label] = j

        return rd, model

    def reset(self):
        """reset()

        Reset all attributes of `LossFunc` at their initial values.

        """

        self.best_score = float("inf")
        self.best_sol = None
        self.best_argmin = None

        self.all_scores = []
        self.all_solutions = []

        self.calls = 0
        # Must be private, ?? voir
        self.new_best = False

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

        if self.verbose:
            self.manager = enlighten.get_manager()
        else:
            self.manager = enlighten.get_manager(stream=None, enabled=False)


class FDA_loss_func:

    """FDA_loss_func

    FDA_loss_func allows to wrap function of type f(x)=(y, model), so it can be used in by the Fractal Decomposition Algorithm.

    Must be modified
    """

    def __init__(self, model, H, sp):

        ##############
        # PARAMETERS #
        ##############

        self.loss_func = model
        self.H = H
        self.search_space = sp

    @property
    def calls(self):
        return self.loss_func.calls

    @property
    def save(self):
        return self.loss_func.save

    @property
    def best_score(self):
        return self.loss_func.best_score

    @property
    def best_sol(self):
        return self.loss_func.best_sol

    @property
    def all_scores(self):
        return self.loss_func.all_scores

    @property
    def all_solutions(self):
        return self.loss_func.all_solutions

    @property
    def new_best(self):
        return self.loss_func.new_best

    @property
    def labels(self):
        return self.loss_func.labels

    @labels.setter
    def labels(self, v):
        self.loss_func.labels = v

    @property
    def folder_name(self):
        return self.loss_func.folder_name

    @property
    def outputs_path(self):
        return self.loss_func.outputs_path

    @property
    def model_path(self):
        return self.loss_func.model_path

    @property
    def plots_path(self):
        return self.loss_func.plots_path

    @property
    def loss_file(self):
        return self.loss_func.loss_file

    @property
    def file_created(self):
        return self.loss_func.file_created

    @file_created.setter
    def file_created(self, v):
        self.loss_func.file_created = v

    def __call__(self, X, **kwargs):
        res = self.loss_func(X, **kwargs)
        X_c = self.search_space.convert_to_continuous(X)
        self.H.add_point(res, X_c)
        return res


class MPILoss(LossFunc):

    """MPILoss

    MPILoss allows to wrap function of type f(x)=(y, model). MPILoss adds method to distribute dynamically the evaluation of multiple solutions.
    It does not distribute the original loss function itself

    Attributes
    ----------
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

    Methods
    -------
    __call__(X, filename='', **kwargs)
        Evaluate a list X of solutions with the original loss function.

    worker()
        Initialize a worker.

    stop()
        Stops all the workers and master.

    _save_model(score, source)
        See LossFunc, save a model according to its score and the worker rank.

    See Also
    --------
    Loss : Wrapper function
    LossFunc : Inherited class
    SerialLoss : Basic version of LossFunc
    """

    def __init__(self, model, save=False, verbose=True):

        """__init__(model, save=False)

        Initialize MPI variables. For more info, see LossFunc.

        """

        super().__init__(model, save, verbose)

        #################
        # MPI VARIABLES #
        #################

        try:
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
        except Exception as err:
            logger.error(
                "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
            You can use: pip install zellij[Parallel]"
            )

            raise err

        # Master or worker process
        self.master = self.rank == 0

        if self.master:
            if os.path.exists("tmp_wks"):
                shutil.rmtree("tmp_wks")
            os.makedirs("tmp_wks")
        else:
            self.worker()

    def __call__(self, X, label=[], **kwargs):

        """__call__(model, save_model='')

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

        logger.info("Master Starting")

        assert self.p > 1, "n_process must be > 1"

        self.build_bar(len(X))

        self.new_best = False

        res = [None] * len(X)
        send_history = [-1] * (self.p)

        nb_send = 0

        # Send a solution to all available processes
        while nb_send < len(X) and nb_send < (self.p - 1):

            logger.debug(f"MASTER {self.rank} sending to {nb_send}")

            self.comm.send(dest=nb_send + 1, tag=0, obj=X[nb_send])
            send_history[nb_send + 1] = nb_send
            nb_send += 1

        # Dynamically send and receive solutions and results to and from workers
        nb_recv = 0
        while nb_send < len(X):

            logger.debug(f"MASTER {self.rank} receiving | {nb_recv} < {len(X)}")

            msg, others = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=0, status=self.status
            )
            source = self.status.Get_source()

            logger.debug(f"MASTER {self.rank} received from {source}")

            res[send_history[source]] = msg

            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(msg, source)

                # Save score and solution into a file
                self._save_file(
                    X[send_history[source]],
                    res[send_history[source]],
                    **others,
                    **kwargs,
                )

            # Save score and solution into the object
            self._save_best(X[send_history[source]], res[send_history[source]])

            nb_recv += 1
            self.calls += 1

            logger.debug(f"MASTER {self.rank} sending to {nb_send}")

            self.comm.send(dest=source, tag=0, obj=X[nb_send])
            send_history[source] = nb_send

            nb_send += 1

        # Receive last results from workers
        while nb_recv < len(X):

            logger.debug(
                f"MASTER {self.rank} last receiving | {nb_recv} < {len(X)}"
            )

            msg, others = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=0, status=self.status
            )
            source = self.status.Get_source()

            logger.debug(f"MASTER {self.rank} received from {source}")

            nb_recv += 1
            self.calls += 1

            res[send_history[source]] = msg

            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(msg, source)

                # Save score and solution into a file
                self._save_file(
                    X[send_history[source]],
                    res[send_history[source]],
                    **others,
                    **kwargs,
                )

            # Save score and solution into the object
            self._save_best(X[send_history[source]], res[send_history[source]])

        self.close_bar()

        logger.info("Master ending")

        return res

    def worker(self):

        """worker()

        Initialize worker. Whilte it does not receive a stop message, a worker will wait for a solution to evaluate.

        """

        logger.info(f"Worker{self.rank} starting")

        stop = True

        while stop:

            logger.debug(f"WORKER {self.rank} receving")
            msg = self.comm.recv(source=0, tag=0, status=self.status)

            if msg != None:

                logger.debug(f"WORKER {self.rank} evaluating: {msg}")

                res, trained_model = self._build_return(self.model(msg))

                score, others = res["score"], res

                # Verify if a model is returned or not
                # Save the model using its save method
                if trained_model and self.save:
                    if hasattr(trained_model, "save") and callable(
                        getattr(trained_model, "save")
                    ):
                        worker_path = os.path.join(
                            "tmp_wks", f"worker{self.rank}"
                        )
                        os.system(f"rm -rf {worker_path}")
                        trained_model.save(worker_path)
                    else:
                        logger.error(
                            "Model/loss function does not have a method called `save`"
                        )
                        exit()

                # Send results
                logger.debug(f"WORKER {self.rank} sending {score}")

                self.comm.send(dest=0, tag=0, obj=[score, others])
            else:
                stop = False

        logger.info(f"Worker{self.rank} ending")

    def stop(self):

        """stop()

        Send a stop message to all workers.

        """

        for i in range(1, self.p):
            self.comm.send(dest=i, tag=0, obj=None)

        shutil.rmtree("tmp_wks", ignore_errors=True)

    def _save_model(self, score, source):

        """ _save_model()

        Private method to save a model. Be carefull, to be exploitable, the initial loss func must be of form f(x) = (y, model)\
         y is the results of the evaluation of x by f. model is optional, if you want to save the best found model (e.g. a neural network)\
         you can return the model. However the model must have a "save" method (e.g. model.save(filename)).

         score : int
            Score corresponding to the source file
        source : int
            Worker rank which evaluate a solution and return score

        """

        # Save model into a file if it is better than the best found one
        if score < self.best_score:

            master_path = ave_path = os.path.join(
                self.model_path, f"{self.model.__class__.__name__}_best"
            )
            worker_path = os.path.join("tmp_wks", f"worker{source}")

            if os.path.isdir(worker_path):
                os.system(f"rm -rf {master_path}")
                os.system(f"cp -rf {worker_path} {master_path}")


class SerialLoss(LossFunc):

    """SerialLoss

    SerialLoss allows to wrap function of type f(x)=(y, model). SerialLoss adds methods to save and evaluate the original loss function.

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

    def __init__(self, model, save=False, verbose=True):

        """__init__(model, save=False)

        Initialize SerialLoss.

        """

        super().__init__(model, save, verbose)

    def __call__(self, X, **kwargs):

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

        self.build_bar(len(X))

        self.new_best = False

        res = []

        for x in X:
            outputs, trained_model = self._build_return(self.model(x))
            score, others = outputs["score"], outputs

            res.append(score)

            self.calls += 1

            # Saving
            if self.save:
                self._save_file(x, score, **others, **kwargs)

                if trained_model:
                    self._save_model(score, trained_model)

            self._save_best(x, score)

        self.close_bar()
        return res

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
                logger.error(
                    "Model/loss function does not have a method called `save`"
                )
                exit()


# Wrap different loss functions
def Loss(model=None, save=False, verbose=True, MPI=False):
    """Loss(model=None, save=False, verbose=True, MPI=False)

    Wrap a function of type :math:`f(x)=y`. See `LossFunc` for more info.

    Parameters
    ----------
    model : function, default=None
        Function of type f(x)=y. x must be a solution. A solution can be a list of float, int... It can also be of mixed types, containing, strings, float, int...

    save : string, optional
        Filename where to save the best found model. Only one model is saved for memory issues.

    MPI : boolean, optional
        Wrap the function with MPILoss if True, with SerialLoss else.

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
    >>> print(f"Best solution found: f({himmelblau.best_sol}) = {himmelblau.best_score}")
    Best solution found: f(None) = inf
    >>> print(f"Number of evaluations:{himmelblau.calls}")
    Number of evaluations:0
    >>> print(f"All evaluated solutions:{himmelblau.all_solutions}")
    All evaluated solutions:[]
    >>> print(f"All loss values:{himmelblau.all_scores}")
    All loss values:[]



    """
    if model:
        return SerialLoss(model)
    else:

        def wrapper(model):
            if MPI:
                return MPILoss(model, save, verbose)
            else:
                return SerialLoss(model, save, verbose)

        return wrapper
