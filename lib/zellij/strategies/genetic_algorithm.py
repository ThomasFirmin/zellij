from deap import base
from deap import creator
from deap import tools
import numpy as np

class Genetic_algorithm(Metaheuristic):

    """Genetic_algorithm

    Genetic_algorithm (GA) implements a steady state genetic algorithm. It can be used for exploration and exploitation.
    Indeed when the population has converged, GA can ,thanks to the mutation and crossover operators, perform an intensification phase arround best solutions.
    It is algorithm which can work with a mixed search space, by adapting its operator.

    Here the mutation operator is the neighborhood defined in the Searchspace object.
    Available crossover operator are those compatible with a mixed individual (1-point, 2-points...). Same with the slection.

    It uses DEAP.
    See Metaheuristic for more info.

    Attributes
    ----------

    pop_size : int
        Population size of the GA.\
        In a distributed environment (e.g. MPILoss), it has an influence on the parallelization quality.\
        It must be tuned according the available hardware.

    generation : int
        Generation number of the GA.

    Methods
    -------

    run(self, n_process=1)
        Runs Genetic_algorithm

    show(filename=None)
        Plots results

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self,loss_func, search_space, f_calls, pop_size = 10, generation = 1000, save=False, verbose=True):

        """__init__(self,loss_func, search_space, f_calls, pop_size = 10, generation = 1000, save=False, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        pop_size : int
            Population size of the GA.\
            In a distributed environment (e.g. MPILoss), it has an influence on the parallelization quality.\
            It must be tuned according the available hardware.

        generation : int
            Generation number of the GA.

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.pop_size = pop_size
        self.generation = generation

    # Define what an individual is
    def define_individual(self):
        """define_individual(self)

        Describe how an individual should be initialized. Here a random point from SearchSpace is sampled.

        """
        # Select one random point from the search space
        solution = self.search_space.random_point()[0]

        return solution

    # Initialize an individual extracted from a file
    def initIndividual(self, icls, content):
        """initIndividual(self, icls, content)

        Initialize an individual to DEAP.

        """
        return icls([content.to_list()])

    # Initialize a population extracted from a file
    def initPopulation(self, pcls, ind_init, filename):
        """initPopulation(self, pcls, ind_init, filename)

        Initialize a population of individual, from a file, to DEAP.

        """
        data = pd.read_csv(filename, sep = ",", decimal=".", usecols = self.search_space.n_variables)
        contents = data.tail(taille_population)

        return pcls(ind_init(c) for index,c in contents.iterrows())

    # Mutate operator
    def mutate(self,individual,proba):

        """mutate(self, individual, proba)

        Mutate a given individual, using Searchspace neighborhood.

        Parameters
        ----------
        individual : list[{int, float, str}]
            Individual to mutate, in the mixed format.

        proba : float
            Probability to mutate a gene.

        Returns
        -------

        individual : list[{int, float, str}]
            Mutated individual

        """

        # For each dimension of a solution draw a probability to be muted
        for index,label in enumerate(self.search_space.label):
            t = np.random.random()
            if t < proba:

                # Get the a neighbor of the selected attribute
                individual[0][index] = self.search_space.get_neighbor(individual[0],attribute = label)[0]

        return individual,

    # Run GA
    def run(self, n_process = 1,save=False):

        """run(self, n_process = 1,save=False)

        Runs GA

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        save : boolean, default=False
            Deprecated must be removed.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        # Save file
        if save:
            f_pop = open("ga_population.txt","w")
            f_pop.write(str(self.search_space.label)[1:-1].replace(" ","").replace("'","")+",loss_value\n")
            f_pop.close()

        print("Genetic Algorithm starting")

        print("Constructing tools...")

        # Define problem type "fitness", weights = -1.0 -> minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Define what an individual is for the algorithm
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox contains all the operator of GA. (mutate, select, crossover...)
        toolbox = base.Toolbox()

        # Start from a random population
        if self.filename == None:

            # Determine what is an individual
            toolbox.register("hyperparameters", self.define_individual)

            # Determine the way to build individuals for the population
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.hyperparameters, n=1)

            # Determine the way to build a population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            print("Creation of the initial population...")

            # Build the population
            pop = toolbox.population(n=self.pop_size)


        # Start from a saved population
        else:

            toolbox.register("individual_guess", self.initIndividual, creator.Individual)
            toolbox.register("population_guess", self.initPopulation, list, toolbox.individual_guess, self.filename)

            print("Creation of the initial population...")
            pop = toolbox.population_guess()

        # Create crossover tool
        toolbox.register("mate", tools.cxOnePoint)
        # Create mutation tool
        toolbox.register("mutate", self.mutate, proba=1/self.search_space.n_variables)
        # Create selection tool
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create a tool to select best individuals from a population
        toolbox.register("best",tools.selBest, k=int(self.pop_size/2))

        best_of_all = tools.HallOfFame(n_process)

        # Ga initialization

        print("Evaluating the initial population...")
        # Compute dynamically fitnesses
        solutions = []
        solutions = [p[0] for p in pop]
        fitnesses = self.loss_func(solutions)

        self.all_scores += fitnesses

        # Map computed fitness to individual fitness value
        for ind,fit in zip(pop,fitnesses):
            ind.fitness.values = fit,

        fits = [ind.fitness.values[0] for ind in pop]

        if save:
            f_pop = open("ga_population.txt","a")
            for ind,cout in zip(pop,fits):
                f_pop.write(str(ind)[2:-2].replace(" ","").replace("'","")+","+str(cout).replace(" ","")+"\n")
            f_pop.close()


        print("Initial population evaluated")

        print("Evolution starting...")
        g=0
        while g < self.generation: # A revoir avec self.loss_func.call
            g += 1

            # Update all of fame
            best_of_all.update(pop)

            if self.verbose:
                print("Génération: "+str(g))

                # Selection operator
                print("Selection...")

            offspring = toolbox.select(pop,k=len(pop))

            # /!\ On clone (copy), la population selectionnée, pour pouvoir la faire reproduire et la muté sans impacter les individus selectionner
            offspring = list(map(toolbox.clone,offspring))

            children = []

            # Crossover operator
            if self.verbose:
                print("Crossover...")

            i = 0
            for child1,child2 in zip(offspring[::2],offspring[1::2]):

                # Clone individuals from crossover
                children1 = toolbox.clone(child1)
                children2 = toolbox.clone(child2)

                # Apply crossover
                toolbox.mate(children1[0],children2[0])
                # Delete children fitness inherited from the parents
                del children1.fitness.values
                del children2.fitness.values

                # Add new children to list
                children.append(children1)
                children.append(children2)

            # Mutate children
            if self.verbose:
                print("Mutation...")
            for mutant in children:
                toolbox.mutate(mutant)

            if self.verbose:
                print("Evaluating population n°"+str(g))

            # Compute dynamically fitnesses
            solutions = []
            solutions = [p[0] for p in children]
            fitnesses = self.loss_func(solutions)

            # Map computed fitness to individual fitness value
            for ind, fit in zip(children, fitnesses):
                ind.fitness.values = fit,


            # Build new population
            pop[:] = toolbox.best(offspring)+toolbox.best(children)

            # Get fitnesses from the new population
            fits = [ind.fitness.values[0] for ind in pop]

            self.all_scores += fits

            # Save new population
            if save:
                f_pop = open("ga_population.txt","a")
                for ind,cout in zip(pop,fits):
                    f_pop.write(str(ind)[2:-2].replace(" ","").replace("'","")+","+str(cout).replace(" ","")+"\n")
                f_pop.close()

            # End populaiton evaluation
            if self.verbose:
                print("Evaluation n°"+str(g)+"ending...")


        best = []
        min = []

        print("Genetic Algorithm ending")
        for b in best_of_all:
            min.append(b.fitness.values[0])
            best.append(b[0])

            #print best parameters from genetic algorithm
            print("Best parameters: " + str(b[0])+" | score: "+str(b.fitness.values[0]))

        return best, min

    def show(self, filename = None):

        """show(self, filename=None)

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        """


        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        if filename == None:
            scores = np.array(self.loss_func.all_scores)
        else:
            data = pd.read_table(filename,sep=",",decimal   =".")
            scores = data["loss_value"].to_numpy()

        quantile = np.quantile(scores,0.75)
        argmin = np.argmin(scores)
        min = scores[argmin]
        heatmap = scores.reshape((int(len(scores)/self.pop_size),self.pop_size))

        minimums = np.min(heatmap,axis=1)
        means = np.mean(heatmap,axis=1)

        heatmap.sort(axis=1)
        heatmap = heatmap.transpose()

        ax = sns.heatmap(heatmap,vmin=min,vmax=quantile, cmap="YlGnBu",cbar_kws={'label': 'Score'})
        ax.invert_yaxis()
        ax.set_title("Fitness evolution through generations, mininimum="+str(min))
        ax.set(xlabel='Generation number', ylabel='Individual number')
        plt.legend()
        plt.show()

        plt.plot(np.arange(len(minimums)),minimums,"-",label="Best individual",color="red")
        plt.plot(np.arange(len(means)),means,":",label="Mean",color="blue")
        plt.title("Best individual and population's mean through generations")
        plt.xlabel("Generations")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

        if filename != None:
            self.search_space.show(data,scores)
