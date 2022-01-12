import numpy as np
import os
from abc import abstractmethod

try:
    from mpi4py import MPI
except ImportError:
    print("To use MPILoss class you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[Parallel]")

class LossFunc(object):

    """LossFunc

    LossFunc allows to wrap function of type f(x)=(y, model), so it can be used in Zellij by adding several features,\
     such as calls count, saving, parallelization, historic... y is the results of the evaluation of x by f. model is optional, if you want to save the best found model (e.g. a neural network)\
     you can return the model. However the model must have a "save" method (e.g. model.save(filename)).

    Attributes
    ----------
    model : function
        Function of type f(x)=y. x must be a solution. A solution can be a list of float, int... It can also be of mixed types, containing, strings, float, int...
    save_model : string
        Filename where to save the best found model. Only one model is saved for memory issues.
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
    __init__(self, model, save='')
        Initialize LossFunc class

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

    def __init__(self, model, save_model=''):

        """__init__(self, model, save_model='')

        Parameters
        ----------
        model : function
            Function of type f(x)=y. x must be a solution. A solution can be a list of float, int... It can also be of mixed types, containing, strings, float, int...
        save_model : string
            Filename where to save the best found model. Only one model is saved for memory issues. Be carefull, to be exploitable, the initial loss func must be of form f(x) = (y, model)\
             y is the results of the evaluation of x by f. model is optional, if you want to save the best found model (e.g. a neural network)\
             you can return the model. However the model must have a "save" method (e.g. model.save(filename)).

        """
        ##############
        # PARAMETERS #
        ##############

        self.model = model
        self.save_model = save_model

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

    def _save_file(self, solution, score, filename, add=[], others=[]):

        """_save_file(self, model, save_model='')

        Private method to save informations about an evaluation of the loss function. The file is created by a Metaheuristic.

        Parameters
        ----------
        solution : list
            Solution to save.
        solution : list
            Evaluation of the solution by the loss function;
        filename : str
            Name of the file created by a Metaheuristic, where to save informations.
        add : {list, str, float, int}, optional
            Additionnal informations to save before the score.
        others : {list, str, float, int}, optional
            Other informations to save after the score.
        """

        # Determine if additionnal contents must be added to the save
        if type(others) != list:
            others = [others]
        if type(add) != list:
            add = [add]

        if len(others)>0:
            suffix = ','+','.join(str(e) for e in others)
        else:
            suffix = ''

        if len(add)>0:
            prefix = ','+','.join(str(e) for e in others)
        else:
            prefix = ''

        # Save a solution and additionnal contents
        with open(filename,"a+") as f:
            f.write(",".join(str(e) for e in solution)+prefix+str(score)+suffix+"\n")

    # Save best found solution
    def _save_best(self, solution, score):

        # historic
        self.all_solutions.append(solution)
        self.all_scores.append(score)

        # Save best
        if score < self.best_score:
            self.best_score = score
            self.best_sol = solution
            self.new_best = True

    @abstractmethod
    def _save_model(self):
        """ _save_model(self)

        Private abstract method to save a model. Be carefull, to be exploitable, the initial loss func must be of form f(x) = (y, model)\
         y is the results of the evaluation of x by f. model is optional, if you want to save the best found model (e.g. a neural network)\
         you can return the model. However the model must have a "save" method (e.g. model.save(filename)).

        """
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

    def evaluate(self,X):
        res = self.loss_func(X)
        self.H.add_point(res,X)

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
    __init__(self, model, save='')
        Initialize MPILoss class.

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

    def __init__(self, model, save_model=''):

        """__init__(self, model, save_model='')

        Initialize MPI variables. For more info, see LossFunc.

        """

        super().__init__(model, save_model)

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
            print("To use MPILoss object you need to install mpi4py and an MPI distribution\n\
            You can use: pip install zellij[Parallel]")
            exit()

        # Master or worker process
        self.master = self.rank==0

        if self.rank != 0:
            self.worker()

    def __call__(self, X, filename='', **kwargs):

        """__call__(self, model, save_model='')

        Evaluate a list X of solutions with the original loss function.

        Parameters
        ----------
        X : list
            List of solutions to evaluate. be carefull if a solution is a list X must be a list of lists.
        filename : str
            Name of the file created by a Metaheuristic, where to save informations.
        **kwargs : dict, optional
            Additionnal informations to save before the score.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """

        if len(kwargs.keys()) != 0:
            add = ","+",".join(str(e) for e in kwargs.values())
        else:
            add = ""

        assert self.p > 1, "n_process must be > 1"

        res = [None]*len(X)
        send_history = [-1]*(self.p)

        nb_send = 0

        # Send a solution to all available processes
        while nb_send < len(X) and nb_send < (self.p-1):
            print("MASTER "+str(self.rank)+" sending to"+str(nb_send))
            self.comm.send(dest=nb_send+1,tag=0,obj=X[nb_send])
            send_history[nb_send+1] = nb_send
            nb_send += 1

        # Dynamically send and receive solutions and results to and from workers
        nb_recv = 0
        while nb_send < len(X):

            print("MASTER "+str(self.rank)+" receiving | "+str(nb_recv)+"<"+str(len(X)))
            msg,others = self.comm.recv(source=MPI.ANY_SOURCE,tag=0,status=self.status)
            source = self.status.Get_source()
            print("MASTER "+str(self.rank)+" received from "+str(source))

            res[send_history[source]] = msg

            # Save model into a file if it is better than the best found one
            self._save_model(msg, source)

            # Save score and solution into the object
            self._save_best(X[send_history[source]],res[send_history[source]])

            # Save score and solution into a file
            self._save_file(X[send_history[source]], res[send_history[source]], filename, add, others)

            nb_recv += 1
            self.calls += 1

            print("MASTER "+str(self.rank)+" sending to"+str(source))
            self.comm.send(dest=source,tag=0,obj=X[nb_send])
            send_history[source] = nb_send

            nb_send += 1


        # Receive last results from workers
        while nb_recv < len(X):

            print("MASTER "+str(self.rank)+" end receiving | "+str(nb_recv)+"<"+str(len(X)))
            msg,others = self.comm.recv(source=MPI.ANY_SOURCE,tag=0,status=self.status)
            source = self.status.Get_source()
            print("MASTER "+str(self.rank)+" received from "+str(source))

            nb_recv += 1
            self.calls += 1

            res[send_history[source]] = msg

            # Save model into a file if it is better than the best found one
            self._save_model(msg, source)

            # Save score and solution into the object
            self._save_best(X[send_history[source]],res[send_history[source]])

            # Save score and solution into a file
            if filename != '':
                self._save_file(X[send_history[source]], res[send_history[source]], filename, add, others)

            print("MASTER FINISHING")

        return res

    def worker(self):

        """worker(self)

        Initialize worker. Whilte it does not receive a stop message, a worker will wait for a solution to evaluate.

        """

        stop = True

        while stop:

            print("WORKER "+str(self.rank)+" receving")
            msg = self.comm.recv(source=0,tag=0,status=self.status)

            if msg != None :

                print("WORKER "+str(self.rank)+" evaluating:\n"+str(msg))

                res = self.model(msg)

                # Verify if a model is returned or not
                if type(res) != tuple:
                    results = res
                else:
                    results,trained_model = res[0],res[1]

                    # Save the model using its save method
                    if self.save_model:
                        if hasattr(trained_model, "save") and callable(getattr(trained_model, "save")):
                            os.system("rm -rf worker"+str(self.rank))
                            trained_model.save("worker"+str(self.rank))
                        else:
                            print("Error: model does not have a method called save")
                            exit()

                # Verify if the results is just a score or more
                if type(results) == list or type(results) == tuple:
                    score,others = results[0],results[1:]
                else:
                    score,others = results,[]

                # Send results
                print("WORKER "+str(self.rank)+" sending "+str(score))
                self.comm.send(dest=0,tag=0,obj=[score,others])
            else:
                stop = False

    def stop(self):

        """stop(self)

        Send a stop message to all workers.

        """

        for i in range(1,self.p):
            self.comm.send(dest=i,tag=0,obj=None)


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
            os.system(f"rm -rf {self.save_model}")
            os.system(f'cp -rf worker{source} {self.save_model}')

class SerialLoss(LossFunc):

    """SerialLoss

    SerialLoss allows to wrap function of type f(x)=(y, model). SerialLoss adds methods to save and evaluate the original loss function.

    Methods
    -------
    __init__(self, model, save='')
        Initialize MPILoss class.

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

    def __init__(self, model, save_model=''):

        """__init__(self, model, save_model='')

        Initialize SerialLoss.

        """

        super().__init__(model, save_model)

    def __call__(self, X, filename='',**kwargs):

        """__call__(self, model, save_model='')

        Evaluate a list X of solutions with the original loss function.

        Parameters
        ----------
        X : list
            List of solutions to evaluate. be carefull if a solution is a list X must be a list of lists.
        filename : str
            Name of the file created by a Metaheuristic, where to save informations.
        **kwargs : dict, optional
            Additionnal informations to save before the score.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """

        self.new_best = False

        res = []

        if len(kwargs.keys()) != 0:
            add = ","+",".join(str(e) for e in kwargs.values())
        else:
            add = ""

        for x in X:
            ouputs = self._model(x)

            # Verify if a model is returned or not
            if type(ouputs) != tuple:
                results, trained_model = ouputs, None
            else:
                results, trained_model = ouputs[0], ouputs[1]

            # Verify if the results is just a score or more
            if type(results) == list or type(results) == tuple:
                score,others = results[0],results[1:]
            else:
                score,others = results,[]

            # Save model into a file if it is better than the best found one
            _save_model(self, score, trained_model)

            res.append(score)

            self.calls += 1

            # Saving

            self._save_best(x,score)

            if filename != '':
                self._save_file(x,score,filename,add,others)

        return res

    def _save_model(self, score, trained_model):
        # Save model into a file if it is better than the best found one
        if self.save_model and score < self.best_score:
            if hasattr(trained_model, "save") and callable(getattr(trained_model, "save")):
                os.system(f"rm -rf {self.save_model}")
                trained_model.save(self.save_model)
            else:
                print("Error: model does not have a method called save")
                exit()

# Wrap different loss functions
def Loss(model=None, save_model='', MPI = False):
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
                return MPILoss(model, save_model)
            else:
                return SerialLoss(model, save_model)

        return wrapper
