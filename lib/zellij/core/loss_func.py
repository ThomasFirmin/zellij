import numpy as np
import os
import shutil
from abc import abstractmethod

try:
    from mpi4py import MPI
except ImportError:
    print(
        "To use MPILoss class you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[Parallel]"
    )


class LossFunc(object):

    """LossFunc

    LossFunc allows to wrap function of type f(x)=(y, model), so it can be used in Zellij by adding several features,\
     such as calls count, saves, parallelization, historic... y is the results of the evaluation of x by f. model is optional,\
      if you want to save the best found model (e.g. a neural network) you can return the model.
    However the model must have a "save" method (e.g. model.save(filename)).

    Attributes
    ----------
    model : function
        Function of type f(x)=y. x must be a solution. A solution can be a list of float, int... It can also be of mixed types, containing, strings, float, int...
    best_score : float
        Best found score among all loss function evaluations.
    best_score : float
        Best found score among all loss function evaluations.
    best_sol : list
        Best found solution among all loss function evaluations.
    all_scores : float
        Historic of all evaluated scores.
    all_solutions : float
        Historic of all evaluated solutions.
    calls : int
        Number of loss function calls

    Methods
    -------
    _save_file(self,solution, score, filename, add, others=[])
        Save informations into a file created by a Metaheuristic.

    _save_best(self, solution, score)
        Save best found solution and score into self.

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc
    """

    def __init__(self, model, save=False):

        """__init__(self, model, save=False)

        Parameters
        ----------
        model : function
            Function of type f(x)=y. x must be a solution. A solution can be a list of float, int... It can also be of mixed types, containing, strings, float, int...
        save : string
            Filename where to save the best found model and the historic of the loss function.
            Only one model is saved for memory issues. Be carefull, to be exploitable, the initial loss func must be of form f(x) = (y, model)\
             y is the results of the evaluation of x by f. model is optional, if you want to save the best found model (e.g. a neural network)\
             you can return the model. However the model must have a "save" method (e.g. model.save(filename)).

        """
        ##############
        # PARAMETERS #
        ##############

        self.model = model
        self.save = save

        #############
        # VARIABLES #
        #############

        self.best_score = float("inf")
        self.best_sol = None

        self.all_scores = []
        self.all_solutions = []

        self.calls = 0
        # Must be private, to modify
        self.new_best = False

        self.labels = []

        if isinstance(self.save, str):
            self.folder_name = self.save
        else:
            self.folder_name = f"{self.model.__name__}_zlj_save"

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.file_created = False

    def _create_file(self, x, *args):

        """create_file(self, *args)

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
                print(f"WARNING: Folder {self.folder_name} already exists, results will be saved at {nfolder}")
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
            print("WARNING: Labels are of incorrect size, it will be replaced in the save file header")
            for i in range(len(x)):
                self.labels.append(f"attribute{i}")

        with open(self.loss_file, "w") as f:
            f.write(",".join(str(e) for e in self.labels) + ",loss" + suffix + "\n")

        print("INFO: Results will be saved at: " + os.path.abspath(self.folder_name))

        self.file_created = True

    def _save_file(self, x, y, **kwargs):

        """_save_file(self, model, save_model='')

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

        # historic
        self.all_solutions.append(x)
        self.all_scores.append(y)

        # Save best
        if y < self.best_score:
            self.best_score = y
            self.best_sol = x
            self.new_best = True

    def _build_return(self, r):
        """_build_return(self, r)

        This method builds a unique return according to the outputs of the loss function

        Parameters
        ----------
        r : {tuple, float, int}
            Return of of the loss function

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
            results, model = r
        else:
            results, model = r, False

        # Separate results
        if isinstance(results, int) or isinstance(results, float):
            rd["score"] = results
        elif isinstance(results, dict):
            rd = results
            rd["score"] = r.values()[0]
        elif isinstance(results, tuple):
            rd["score"] = results[0]
            for i, j in enumerate(results):
                label = f"return{i}"
                rd[label] = j

        return rd, model

    @abstractmethod
    def _save_model(self, *args):
        """ _save_model(self)

        Private abstract method to save a model. Be carefull, to be exploitable, the initial loss func must be of form f(x) = (y, model)\
         y is the results of the evaluation of x by f. model is optional, if you want to save the best found model (e.g. a neural network)\
         you can return the model. However the model must have a "save" method (e.g. model.save(filename)).

        """
        pass

    @abstractmethod
    def __call__(self, X, **kwargs):
        pass


class FDA_loss_func:

    """FDA_loss_func

    FDA_loss_func allows to wrap function of type f(x)=(y, model), so it can be used in by the Fractal Decomposition Algorithm.

    Must be modified
    """

    def __init__(self, model, H):

        ##############
        # PARAMETERS #
        ##############

        self.loss_func = model
        self.H = H

    def __call__(self, X):
        res = self.loss_func(X)
        self.H.add_point(res, X)

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
    __call__(self, X, filename='', **kwargs)
        Evaluate a list X of solutions with the original loss function.

    worker(self)
        Initialize a worker.

    stop(self)
        Stops all the workers and master.

    _save_model(self, score, source)
        See LossFunc, save a model according to its score and the worker rank.

    See Also
    --------
    Loss : Wrapper function
    LossFunc : Inherited class
    SerialLoss : Basic version of LossFunc
    """

    def __init__(self, model, save=False):

        """__init__(self, model, save=False)

        Initialize MPI variables. For more info, see LossFunc.

        """

        super().__init__(model, save)

        #################
        # MPI VARIABLES #
        #################

        try:
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
        except Exception as e:
            print(e)
            print(
                "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
            You can use: pip install zellij[Parallel]"
            )
            exit()

        # Master or worker process
        self.master = self.rank == 0

        if master:
            os.mkdir("tmp_wks")
        else:
            self.worker()

    def __call__(self, X, label=[], **kwargs):

        """__call__(self, model, save_model='')

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

        assert self.p > 1, "n_process must be > 1"

        res = [None] * len(X)
        send_history = [-1] * (self.p)

        nb_send = 0

        # Send a solution to all available processes
        while nb_send < len(X) and nb_send < (self.p - 1):
            print("MASTER " + str(self.rank) + " sending to" + str(nb_send))
            self.comm.send(dest=nb_send + 1, tag=0, obj=X[nb_send])
            send_history[nb_send + 1] = nb_send
            nb_send += 1

        # Dynamically send and receive solutions and results to and from workers
        nb_recv = 0
        while nb_send < len(X):

            print("MASTER " + str(self.rank) + " receiving | " + str(nb_recv) + "<" + str(len(X)))
            msg, others = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
            source = self.status.Get_source()
            print("MASTER " + str(self.rank) + " received from " + str(source))

            res[send_history[source]] = msg

            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(msg, source)

                # Save score and solution into a file
                self._save_file(X[send_history[source]], res[send_history[source]], **others, **kwargs)

            # Save score and solution into the object
            self._save_best(X[send_history[source]], res[send_history[source]])

            nb_recv += 1
            self.calls += 1

            print("MASTER " + str(self.rank) + " sending to" + str(source))
            self.comm.send(dest=source, tag=0, obj=X[nb_send])
            send_history[source] = nb_send

            nb_send += 1

        # Receive last results from workers
        while nb_recv < len(X):

            print("MASTER " + str(self.rank) + " end receiving | " + str(nb_recv) + "<" + str(len(X)))
            msg, others = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
            source = self.status.Get_source()
            print("MASTER " + str(self.rank) + " received from " + str(source))

            nb_recv += 1
            self.calls += 1

            res[send_history[source]] = msg

            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(msg, source)

                # Save score and solution into a file
                self._save_file(X[send_history[source]], res[send_history[source]], **others, **kwargs)

            # Save score and solution into the object
            self._save_best(X[send_history[source]], res[send_history[source]])

            print("MASTER FINISHING")

        return res

    def worker(self):

        """worker(self)

        Initialize worker. Whilte it does not receive a stop message, a worker will wait for a solution to evaluate.

        """

        stop = True

        while stop:

            print("WORKER " + str(self.rank) + " receving")
            msg = self.comm.recv(source=0, tag=0, status=self.status)

            if msg != None:

                print("WORKER " + str(self.rank) + " evaluating:\n" + str(msg))

                res, trained_model = self._build_return(self.model(msg))

                score, others = res["score"], res

                # Verify if a model is returned or not
                # Save the model using its save method
                if trained_model and self.save:
                    if hasattr(trained_model, "save") and callable(getattr(trained_model, "save")):
                        worker_path = os.path.join("tmp_wks", f"worker{self.rank}")
                        os.system(f"rm -rf {worker_path}")
                        trained_model.save(worker_path)
                    else:
                        print("Error: model does not have a method called save")
                        exit()

                # Send results
                print("WORKER " + str(self.rank) + " sending " + str(score))
                self.comm.send(dest=0, tag=0, obj=[score, others])
            else:
                stop = False

    def stop(self):

        """stop(self)

        Send a stop message to all workers.

        """

        for i in range(1, self.p):
            self.comm.send(dest=i, tag=0, obj=None)

        shutil.rmtree("tmp_wks", ignore_errors=True)

    def _save_model(self, score, source):

        """ _save_model(self)

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

            master_path = ave_path = os.path.join(self.model_path, f"{self.model.__name__}_best")
            worker_path = os.path.join("tmp_wks", f"worker{source}")

            if os.path.isdir(worker_path):
                os.system(f"rm -rf {master_path}")
                os.system(f"cp -rf {worker_path} {master_path}")


class SerialLoss(LossFunc):

    """SerialLoss

    SerialLoss allows to wrap function of type f(x)=(y, model). SerialLoss adds methods to save and evaluate the original loss function.

    Methods
    -------

    __call__(self, X, filename='', **kwargs)
        Evaluate a list X of solutions with the original loss function.

    _save_model(self, score, source)
        See LossFunc, save a model according to its score and the worker rank.

    See Also
    --------
    Loss : Wrapper function
    LossFunc : Inherited class
    MPILoss : Distributed version of LossFunc
    """

    def __init__(self, model, save=False):

        """__init__(self, model, save=False)

        Initialize SerialLoss.

        """

        super().__init__(model, save)

    def __call__(self, X, **kwargs):

        """__call__(self, model, **kwargs)

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

        return res

    def _save_model(self, score, trained_model):
        # Save model into a file if it is better than the best found one
        if score < self.best_score:
            save_path = os.path.join(self.model_path, f"{self.model.__name__}_best")
            if hasattr(trained_model, "save") and callable(getattr(trained_model, "save")):
                os.system(f"rm -rf {save_path}")
                trained_model.save(save_path)
            else:
                print("Error: model does not have a method called save")
                exit()


# Wrap different loss functions
def Loss(model=None, save=False, MPI=False):
    """Loss(model, save_model='', MPI=False)

    Wrap a function of type f(x)=y. See LossFunc for more info.

    Parameters
    ----------
    model : function, default=None
        Function of type f(x)=y. x must be a solution. A solution can be a list of float, int... It can also be of mixed types, containing, strings, float, int...

    save_model : string, default=''
        Filename where to save the best found model. Only one model is saved for memory issues.

    MPI : boolean, default=False
        Wrap the function with MPILoss if True, with SerialLoss else.

    Returns
    -------
    wrapper : LossFunc
        Wrapped original function

    """
    if model:
        return SerialLoss(model)
    else:

        def wrapper(model):
            if MPI:
                return MPILoss(model, save)
            else:
                return SerialLoss(model, save)

        return wrapper
