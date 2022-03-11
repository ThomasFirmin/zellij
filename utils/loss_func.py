import numpy as np
import pandas as pd
import os
from mpi4py import MPI


class FDA_loss_func:

    def __init__(self, H, loss_func):
        self.loss_func = loss_func
        self.H = H
        self.f_calls = 0

    def evaluate(self, X):
        res = self.loss_func(X)
        self.f_calls += len(res)
        self.H.add_point(res, X)

        return res


class MPI_loss_func:

    def __init__(self, model, model_kwargs={}, verbose=False, save_model=False):

        self.model = model
        self.model_kwargs = model_kwargs
        self.verbose = verbose

        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.rank = self.comm.Get_rank()
        self.p = self.comm.Get_size()

        self.best_score = float("inf")
        self.save_model = save_model

    def evaluate(self, X, generation):

        if self.p < 2:
            raise ValueError("n_process must be > 1")

        else:

            res = [None] * len(X)
            send_history = [-1] * self.p

            nb_send = 0

            while nb_send < len(X) and nb_send < (self.p - 1):
                print("MASTER " + str(self.rank) + " sending to " + str(nb_send))
                self.comm.send(dest=nb_send + 1, tag=0, obj=[X[nb_send], generation])
                send_history[nb_send + 1] = nb_send
                nb_send += 1

            nb_recv = 0
            while nb_send < len(X):

                print("MASTER " + str(self.rank) + " receiving | " + str(nb_recv) + "<" + str(len(X)))
                msg = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
                source = self.status.Get_source()
                print("MASTER " + str(self.rank) + " received from " + str(source))

                nb_recv += 1
                res[send_history[source]] = msg

                print("MASTER " + str(self.rank) + " sending to " + str(source))
                self.comm.send(dest=source, tag=0, obj=[X[nb_send], generation])
                send_history[source] = nb_send

                nb_send += 1
                if type(msg) != list:
                    if msg < self.best_score:
                        self.best_score = msg

                        if self.save_model:
                            os.system("rm -rf best_model")
                            os.system('cp -rf worker' + str(source) + ' best_model')

            while nb_recv < len(X):

                print("MASTER " + str(self.rank) + " end receiving | " + str(nb_recv) + "<" + str(len(X)))
                msg = self.comm.recv(source=MPI.ANY_SOURCE, tag=0, status=self.status)
                source = self.status.Get_source()
                print("MASTER " + str(self.rank) + " received from " + str(source))

                nb_recv += 1
                res[send_history[source]] = msg

                if type(msg) != list:

                    if msg < self.best_score:
                        self.best_score = msg

                        if self.save_model:
                            os.system("rm -rf best_model")
                            os.system('cp -R worker' + str(source) + ' best_model')

            print("MASTER FINISHING")

            return res

    def worker(self):

        stop = True

        while stop:

            print("WORKER " + str(self.rank) + " receving")
            msg = self.comm.recv(source=0, tag=0, status=self.status)
            print("msg: ", msg[0])

            if msg is not None:

                args = msg[0]
                generation = msg[1]

                print("WORKER " + str(self.rank) + " evaluating:\n" + str(args) + "\ngeneration: " + str(generation))

                res = self.model(args, self.rank, generation)

                score = res

                """if self.save_model:
                    os.system("rm -rf worker"+str(self.rank))
                    trained_model.save("worker"+str(self.rank))"""

                print("WORKER " + str(self.rank) + " sending " + str(score))
                self.comm.send(dest=0, tag=0, obj=score)
            else:
                stop = False

    def stop(self):
        for i in range(1, self.p):
            self.comm.send(dest=i, tag=0, obj=None)


class loss_func:
    def __init__(self, model, model_kwargs={}, verbose=False):
        self.model = model
        self.model_kwargs = model_kwargs
        self.verbose = verbose

        self.best_score = float("inf")

    def evaluate(self, X):
        res = []

        for x in X:
            res.append(self.model(x, **self.model_kwargs))

        return res
