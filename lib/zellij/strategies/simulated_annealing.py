import numpy as np

class Simulated_annealing(Metaheuristic):

    """Simulated_annealing

    Simulated_annealing (SA) is an exploitation strategy allowing to do hill climbing by starting from\
    an initial solution and iteratively moving to next one,\
     better than the previous one, or slightly worse to escape from local optima.

    It uses a cooling schedule which partially drives the acceptance probability. This is the probability\
    to accept a worse solution according to the temperature, the best solution found so far and the actual solution.

    Attributes
    ----------

    max_iter : int
        Maximum iterations of the inner loop. Determine how long the algorithm should sampled neighbors of a solution,\
        before decreasing the temperature.

    T_0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    T_end : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        T_0. It allows to periodically easily escape from local optima.

    n_peaks : int
        Maximum number of crossed threshold according to T_end. The temperature will be increased\
        <n_peaks> times.

    red_rate : float
        Reduction rate of the initial temperature.
        Each time the threshold is crossed, temperature is increased by <red_rate>*<T_0>.


    Methods
    -------

    run(self, n_process=1)
        Runs Genetic_algorithm

    decrease_temperature(self, T)
        Linear cooling schedule. Deprecated. Must be replaced by a cooling schedule object.

    number_of_iterations(self)
        Determine the number of iterations with the actual parameters.

    show(filename=None)
        Plots results

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    # Initialize simulated annealing
    def __init__(self,loss_func, search_space, f_calls, max_iter, T_0, T_end, n_peaks=1, red_rate=0.80,save=False,verbose=True):

        """__init__(self,loss_func, search_space, f_calls, max_iter, T_0, T_end, n_peaks=1, red_rate=0.80,save=False,verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        max_iter : int
            Maximum iterations of the inner loop. Determine how long the algorithm should sampled neighbors of a solution,\
            before decreasing the temperature.

        T_0 : float
            Initial temperature of the cooling schedule.\
             Higher temperature leads to higher acceptance of a worse solution. (more exploration)

        T_end : float
            Temperature threshold. When reached the temperature is violently increased proportionately to\
            T_0. It allows to periodically easily escape from local optima.

        n_peaks : int, default=1
            Maximum number of crossed threshold according to T_end. The temperature will be increased\
            <n_peaks> times.

        red_rate : float, default=0.80
            Reduction rate of the initial temperature.
            Each time the threshold is crossed, temperature is increased by <red_rate>*<T_0>.

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity


        """

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        # Max iteration after each temperature decrease
        self.max_iter = max_iter

        # Initial temperature
        self.T_0 = T_0
        # Minimum temperature
        self.T_end = T_end

        self.n_scores = []
        self.n_best = []

        # Determine the number of violent temperature rises during the decrease
        self.n_peaks = n_peaks

        # Temperature reduction rate T = T*red_rate
        self.red_rate = red_rate


        self.all_scores = []
        self.record_temp = [self.T_0]
        self.record_proba = [0]

    # Determine the number of iterations
    def number_of_iterations(self):

        """number_of_iterations(self)

        Determine the number of iteration with the current parametrization.

        Returns
        -------

        iteration : int
            Total number of iteration.

        """

        T_init = self.T_0

        T_actu = self.T_0

        iteration = 0
        iteration_temp = 0
        T = []

        while iteration_temp < self.n_peaks:

            iteration += self.max_iter
            inter = self.decrease_temperature(T_actu)
            T_actu = inter[0]
            iteration_temp += inter[1]

        self.T_0 = T_init

        return iteration

    # Decreasing function for the temperature
    def decrease_temperature(self, T):

        """decrease_temperature(self, T)

        Cooling schedule.

        Deprecated must be replaced by a cooling schedule object.

        """

        if T < self.T_end:
            self.T_0 = self.T_0 * self.red_rate
            T = self.T_0

            res = 1

        else:
            T = T*self.red_rate
            res = 0

        return T, res

    # RUN SA
    def run(self, X0 , Y0, n_process=1,save=False):

        """run(self,shift=1, n_process=1)

        Parameters
        ----------
        X0 : list[float]
            Initial solution
        Y0 : {int, float}
            Score of the initial solution
        chaos_level : int, default=0
            Determine at which level of the chaos map, the algorithm starts
        shift : int, default=1
            Determine the starting point of the chaotic map.
        n_process : int, default=1
            Determine the number of best solution found to return.
        save: boolean, default=False
            Deprecated must be removed.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        # Initial solution
        self.X_0 = X0

        # Score of the initial solution
        self.Y_0 = Y0

        print("Simulated Annealing starting")
        print(self.X_0,self.Y_0)

        # Determine the number of iteration according to the function parameters
        print("Determining number of iterations")
        nb_iteration = self.number_of_iterations()
        print("Number of iterations: "+str(nb_iteration))

        if save:

            # Initialize an empty file for saving best parameters
            f = open("simulated_annealing_save.csv", "w")
            f.write(str(self.search_space.label)[1:-1].replace(" ","").replace("'","")+",loss_value\n")
            f.close()

            # Initialize an empty file to save analysis data
            g = open("analyse_rs.csv", "w")
            g.write(str(self.search_space.label)[1:-1].replace(" ","").replace("'","")+",loss_value,temperature,probability\n")
            g.close()

        # Number of temperature variation
        iteration_temp = 0
        total_iteration = 0 # A revoir avec self.loss_func.call

        # Initialize variable for simulated annealing
        # Best solution so far
        X = self.X_0[:]

        # Best solution in the neighborhood
        X_p = X[:]

        # Current solution
        Y = X[:]

        # Initialize score
        cout_X = self.Y_0
        cout_X_p = self.Y_0

        self.all_scores.append(self.Y_0)

        T_actu = self.T_0

        # Simulated annealing starting
        while iteration_temp < self.n_peaks and total_iteration<self.f_calls:
            iteration = 0
            while iteration < self.max_iter and total_iteration<self.f_calls:

                population = []

                neighbors = self.search_space.get_neighbor(X,size=n_process)
                loss_values = self.loss_func(neighbors)

                index_min = np.argmin(loss_values)
                Y = neighbors[index_min][:]
                cout_Y = loss_values[index_min]


                # Compute previous cost minus new cost
                delta = cout_Y - cout_X

                out = "\nNew model score: " + str(cout_Y) + "\nOld model score: " + str(
                    cout_X) + "\nBest model score: " + str(cout_X_p)

                # Save the actual best parameters
                if save:
                    f = open("simulated_annealing_save.csv", "a")
                    for i,j in zip(neighbors,loss_values):
                        f.write(str(i)[1:-1].replace(" ","").replace("'","")+","+str(j).replace(" ","")+ "\n")
                    f.close()

                    g = open("analyse_rs.csv", "a")
                    g.write(str(X)[1:-1].replace(" ","").replace("'","")+","+str(cout_X).replace(" ","")+","+str(self.record_temp[-1]).replace(" ","")+","+str(self.record_proba[-1]).replace(" ","") + "\n")
                    g.close()

                self.all_scores.append(cout_X)

                # If a better model is found do...
                if delta < 0:
                    X = Y[:]
                    cout_X = cout_Y
                    if cout_Y < cout_X_p:

                        # Print if best model is found
                        out += "\nBest model found: /!\ Yes /!\ "

                        X_p = X[:]
                        cout_X_p = cout_X

                    else:
                        out += "\nBest model found: No "

                    self.record_proba.append(0)

                else:
                    out += "\nBest model found: No "
                    p = np.random.uniform(0, 1)
                    emdst = np.exp(-delta / T_actu)

                    self.record_proba.append(emdst)

                    out += "\nEscaping :  p<exp(-df/T) -->" + str(p) + "<" + str(emdst)
                    if p <= emdst:
                        X = Y[:]
                        cout_X = cout_Y
                    else:
                        Y = X[:]

                iteration += 1
                total_iteration += 1*n_process

                out += "\nITERATION:"+str(total_iteration)+"/"+str(nb_iteration)
                out += "\n==============================================================\n"

                self.record_temp.append(T_actu)

                if self.verbose:
                    print(out)

            inter = self.decrease_temperature(T_actu)
            T_actu = inter[0]
            iteration_temp += inter[1]

        #print the best solution from the simulated annealing
        print("Best parameters: " +str(X_p)+" score: "+str(cout_X_p))
        print("Simulated Annealing ending")

        #return self.n_best,self.n_scores

    def show(self, filepath = None):

        """show(self, filename=None)

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        """

        import matplotlib.pyplot as plt
        import pandas as pd

        if filepath == None:

            plt.plot(list(range(len(self.all_scores))),self.all_scores,"-")
            argmin = np.argmin(self.all_scores)
            plt.scatter(argmin, self.all_scores[argmin], color="red",label="Best score: "+str(self.all_scores[argmin]))
            plt.scatter(0, self.Y_0, color="green",label="Initial score: "+str(self.Y_0))

            plt.title('Simulated annealing')
            plt.xlabel("Iteration")
            plt.ylabel("Score")
            plt.legend(loc='upper right')

            plt.show()

            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(list(range(len(self.all_scores))),self.record_temp,"-")
            argmin = np.argmin(self.all_scores)
            ax1.scatter(argmin, self.record_temp[argmin], color="red", label="Temperature of best score: "+str(self.record_temp[argmin]))

            ax1.set_title('Temperature decrease')
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Temperature")
            ax1.legend(loc='upper right')

            if len(self.all_scores) < 100:
                s = 5
            else:
                s = 5000/len(self.all_scores)

            ax2.scatter(list(range(len(self.all_scores))),self.record_proba,s=s)
            argmin = np.argmin(self.all_scores)
            ax2.scatter(argmin, self.record_proba[argmin], color="red", label="Probability of best score: "+str(self.record_proba[argmin]))

            ax2.set_title('Escaping probability')
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Probability")
            ax2.legend(loc='upper right')

            plt.show()

        else:

            data_sa = pd.read_table(filepath+"/analyse_rs.csv",sep=",",decimal=".")
            data_all = pd.read_table(filepath+"/simulated_annealing_save.csv",sep=",",decimal=".")

            all_scores = data_all["loss_value"]
            sa_scores = data_sa["loss_value"]

            argmin = np.argmin(sa_scores)
            f, (l1,l2) = plt.subplots(2, 2)

            ax1,ax2=l1
            ax3,ax4=l2

            ax1.plot(list(range(len(sa_scores))),sa_scores,"-")
            argmin = np.argmin(sa_scores)
            ax1.scatter(argmin, sa_scores[argmin], color="red",label="Best score: "+str(sa_scores[argmin]))
            ax1.scatter(0, sa_scores[0], color="green",label="Initial score: "+str(sa_scores[0]))

            ax1.set_title('Simulated annealing')
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Score")
            ax1.legend(loc='upper right')

            if len(all_scores) < 100:
                s = 5
            else:
                s = 2500/len(all_scores)

            ax2.scatter(list(range(len(all_scores))),all_scores,s=s)
            ax2.scatter(argmin, sa_scores[argmin], color="red", label="Best score: "+str(sa_scores[argmin]))
            ax2.scatter(0,sa_scores[0], color="green",label="Initial score: "+str(sa_scores[0]))

            ax2.set_title('All evaluated solutions')
            ax2.set_xlabel("Solution ID")
            ax2.set_ylabel("Score")
            ax2.legend(loc='upper right')

            try:
                sa_temp = data_sa["temperature"]

                ax3.plot(list(range(len(sa_scores))),sa_temp,"-")
                argmin = np.argmin(sa_scores)
                ax3.scatter(argmin, sa_temp[argmin], color="red", label="Temperature of best score: "+str(sa_temp[argmin]))

                ax3.set_title('Temperature decrease')
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Temperature")
                ax3.legend(loc='upper right')


                sa_proba = data_sa["probability"]

                if len(sa_scores) < 100:
                    s = 5
                else:
                    s = 2500/len(all_scores)

                ax4.scatter(list(range(len(sa_scores))),sa_proba,s=s)
                argmin = np.argmin(sa_scores)
                ax4.scatter(argmin, sa_proba[argmin], color="red", label="Probability of best score: "+str(sa_proba[argmin]))

                ax4.set_title('Escaping probability')
                ax4.set_xlabel("Iteration")
                ax4.set_ylabel("Probability")
                ax4.legend(loc='upper right')
            except:
                pass

            plt.show()

            self.search_space.show(data_all.iloc[:,0:self.search_space.n_variables],data_all.iloc[:,-1])
