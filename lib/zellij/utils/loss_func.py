import numpy as np
import os

try:
    from mpi4py import MPI
except ImportError:
    print("To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[Parallel]")

class FDA_loss_func:

    def __init__(self, loss_func, H):

        ##############
        # PARAMETERS #
        ##############

        self.loss_func = loss_func
        self.H = H

    def evaluate(self,X):
        res = self.loss_func(X)
        self.H.add_point(res,X)

        return res

class MPILoss(object):

    def __init__(self, model, save_model=''):

        ##############
        # PARAMETERS #
        ##############

        self.model = model
        self.save_model = save_model

        #############
        # VARIABLES #
        #############

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

        self.best_score = float("inf")
        self.best_sol = None

        self.all_solutions = []
        self.all_scores = []

        self.calls = 0
        self.new_best = False

        # Master or worker process
        self.master = self.rank==0

        if self.rank != 0:
            self.worker()

    def __call__(self, X, filename='', **kwargs):

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
            if self.save_model and msg < self.best_score:
                os.system(f"rm -rf {self.save_model}")
                os.system(f'cp -rf worker{source} {self.save_model}')

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
            if self.save_model and msg < self.best_score:
                os.system(f"rm -rf {self.save_model}")
                os.system(f'cp -rf worker{source} {self.save_model}')

            # Save score and solution into the object
            self._save_best(X[send_history[source]],res[send_history[source]])

            # Save score and solution into a file
            if filename != '':
                self._save_file(X[send_history[source]], res[send_history[source]], filename, add, others)

            print("MASTER FINISHING")

        return res

    def worker(self):

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
                            print("Warning model does not have a method called save")

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

    def _save_file(self, solution, score, filename, add, others=[]):

        if type(others) != list:
            others = [others]
        if len(others)>0:
            suffix = ','+','.join(str(e) for e in others)
        else:
            suffix = ''

        with open(filename,"a+") as f:
            f.write(",".join(str(e) for e in solution)+add+","+str(score)+suffix+"\n")

    def _save_best(self, solution, score):

        self.all_solutions.append(solution)
        self.all_scores.append(score)

        if score < self.best_score:
            self.best_score = score
            self.best_sol = solution
            self.new_best = True

    def stop(self):
        for i in range(1,self.p):
            self.comm.send(dest=i,tag=0,obj=None)



class SerialLoss(object):
    def __init__(self, model, save_model=''):

        ##############
        # PARAMETERS #
        ##############

        self._model = model
        self.save_model = save_model

        #############
        # VARIABLES #
        #############

        self.best_score = float("inf")
        self.best_sol = None

        self.all_solutions = []
        self.all_scores = []

        self.calls = 0
        self.new_best = False

    def __call__(self, X, filename='',**kwargs):

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

            if self.save_model and score < self.best_score:
                if hasattr(trained_model, "save") and callable(getattr(trained_model, "save")):
                    os.system(f"rm -rf {self.save_model}")
                    trained_model.save(self.save_model)
                else:
                    print("Warning model does not have a method called save")

            res.append(score)

            self.calls += 1

            # Saving

            self._save_best(x,score)

            if filename != '':
                self._save_file(x,score,filename,add,others)

        return res

    def _save_file(self, solution, score, filename, add, others=[]):

        if type(others) != list:
            others = [others]
        if len(others)>0:
            suffix = ','+','.join(str(e) for e in others)
        else:
            suffix = ''

        with open(filename,"a+") as f:
            f.write(",".join(str(e) for e in solution)+add+","+str(score)+suffix+"\n")

    def _save_best(self, solution, score):

        self.all_solutions.append(solution)
        self.all_scores.append(score)

        if score < self.best_score:
            self.best_score = score
            self.best_sol = solution
            self.new_best = True

# Wrap different loss functions
def Loss(model=None, save_model=False, MPI = False):
    if model:
        return SerialLoss(model)
    else:
        def wrapper(model):
            if MPI:
                return MPILoss(model, save_model)
            else:
                return SerialLoss(model, save_model)

        return wrapper
