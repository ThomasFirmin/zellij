from deap import base
from deap import creator
from deap import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Genetic_algorithm:

    def __init__(self, loss_func, search_space, f_calls, pop_size=10, generation=1000, filename=None, verbose=True,
                 logger=None):

        self.loss_func = loss_func
        self.search_space = search_space
        self.f_calls = f_calls
        self.pop_size = pop_size
        self.generation = generation
        self.filename = filename
        self.verbose = verbose
        self.logger = logger

        self.all_scores = []

    # Define what an individual is
    def define_individual(self):

        # Select one random point from the search space
        solution = self.search_space.random_point()[0]

        return solution

    # Initialize an individual extracted from a file
    def initIndividual(self, icls, content):
        return icls([content.to_list()])

    # Initialize a population extracted from a file
    def initPopulation(self, pcls, ind_init, filename):

        data = pd.read_csv(filename, sep=",", decimal=".", usecols=self.search_space.n_variables)
        contents = data.tail(taille_population)

        return pcls(ind_init(c) for index, c in contents.iterrows())

    # Mutate operator
    def mutate(self, individual, proba):

        # For each dimension of a solution draw a probability to be muted
        for index, label in enumerate(self.search_space.label):
            t = np.random.random()
            if t < proba:
                # Get the a neighbor of the selected attribute
                individual[0][index] = self.search_space.get_neighbor(individual[0], attribute=label)[0]

        return individual,

    # Run GA
    def run(self, n_process=1, save=None):

        # Save file
        if save != None:
            f_pop = open(save + "/ga_population.txt", "w")
            f_pop.write(str(self.search_space.label)[1:-1].replace(" ", "").replace("'", "") + ",loss_value\n")
            f_pop.close()

        if self.logger:
            self.logger.info("Genetic Algorithm starting")

            self.logger.info("Constructing tools...")
        else:
            print("Genetic Algorithm starting")

            print("Constructing tools...")

        # Define problem type "fitness", weights = -1.0 -> minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Define what an individual is for the algorithm
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox contains all the operator of GA. (mutate, select, crossover...)
        toolbox = base.Toolbox()

        # Ici on part d'une population tirée aléatoirement.
        if self.filename == None:

            # Determine what is an individual
            toolbox.register("hyperparameters", self.define_individual)

            # Determine the way to build individuals for the population
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.hyperparameters, n=1)

            # Determine the way to build a population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            if self.logger:
                self.logger.info("Creation of the initial population...")
            else:
                print("Creation of the initial population...")

            # Build the population
            pop = toolbox.population(n=self.pop_size)


        # Ici on peut partir d'une population initiale enregistrée dans un fichier, on continue l'exécution d'un algorithme
        else:

            toolbox.register("individual_guess", self.initIndividual, creator.Individual)
            toolbox.register("population_guess", self.initPopulation, list, toolbox.individual_guess, self.filename)

            if self.logger:
                self.logger.info("Creation of the initial population...")
            else:
                print("Creation of the initial population...")
            pop = toolbox.population_guess()

        # Create the crossover tool
        toolbox.register("mate", tools.cxOnePoint)
        # Create the mutation tool
        toolbox.register("mutate", self.mutate, proba=1 / self.search_space.n_variables)
        # Create the selection tool
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create a tool to select best individuals from a population
        toolbox.register("best", tools.selBest, k=int(self.pop_size / 2))

        best_of_all = tools.HallOfFame(n_process)

        # Ga initialization
        if self.logger:
            self.logger.info("Evaluating the initial population...")
        else:
            print("Evaluating the initial population...")
        # Compute dynamically fitnesses
        solutions = []
        solutions = [p[0] for p in pop]
        fitnesses = self.loss_func(solutions, 0)
        self.all_scores += fitnesses

        # Map computed fitness to individual fitness value
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit,

        fits = [ind.fitness.values[0] for ind in pop]

        if save != None:
            f_pop = open(save + "/ga_population.txt", "a")
            for ind, cout in zip(pop, fits):
                f_pop.write(str(ind)[2:-2].replace(" ", "").replace("'", "") + "," + str(cout).replace(" ", "") + "\n")
            f_pop.close()

        if self.logger:
            self.logger.info("Initial population evaluated")
            self.logger.info("Evolution starting...")
        else:
            print("Initial population evaluated")
            print("Evolution starting...")
        g = 0
        while g < self.generation:
            g += 1

            # Optionnel, mets à jour le HallOfFame avec les n meilleurs individus
            best_of_all.update(pop)

            if self.verbose:
                if self.logger:
                    self.logger.info("Génération: " + str(g))
                    # Opération de selection, on selectionne taille_pop individus
                    self.logger.info("Selection...")
                else:
                    print("Génération: " + str(g))
                    # Opération de selection, on selectionne taille_pop individus
                    print("Selection...")

            offspring = toolbox.select(pop, k=len(pop))

            # /!\ On clone (copy), la population selectionnée, pour pouvoir la faire reproduire et la muté sans impacter les individus selectionner
            offspring = list(map(toolbox.clone, offspring))

            children = []

            # Reproduction de la population
            if self.verbose:
                if self.logger:
                    self.logger.info("Crossover...")
                else:
                    print("Crossover...")

            i = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # /!\ On clone les individus selectionnée pour la reproduction
                children1 = toolbox.clone(child1)
                children2 = toolbox.clone(child2)

                # Operateur de reproduction, on reproduit les 2 parents, les caractéristique sont directement modifié (d'où le clonage)
                toolbox.mate(children1[0], children2[0])
                # On supprime les scores des enfants, les scores sont ceux des parents (à cause du clonage)
                del children1.fitness.values
                del children2.fitness.values

                # On ajoute les enfant créé à la list des enfants
                children.append(children1)
                children.append(children2)

            # On fait muter les enfants
            if self.verbose:
                if self.logger:
                    self.logger.info("Mutation...")
                else:
                    print("Mutation...")
            for mutant in children:
                toolbox.mutate(mutant)

            if self.verbose:
                if self.logger:
                    self.logger.info("Evaluating population n°" + str(g))
                else:
                    print("Evaluating population n°" + str(g))

            # Compute dynamically fitnesses
            solutions = []
            solutions = [p[0] for p in children]
            fitnesses = self.loss_func(solutions, g)

            # Map computed fitness to individual fitness value
            for ind, fit in zip(children, fitnesses):
                ind.fitness.values = fit,

            # On reconstruit une population à partir des meilleurs parents et des meilleurs enfants
            pop[:] = toolbox.best(offspring) + toolbox.best(children)

            # Recupération des scores des individus de la population
            fits = [ind.fitness.values[0] for ind in pop]

            self.all_scores += fits
            # Sauvegarde de la population
            if save:
                f_pop = open(save + "/ga_population.txt", "a")
                for ind, cout in zip(pop, fits):
                    f_pop.write(
                        str(ind)[2:-2].replace(" ", "").replace("'", "") + "," + str(cout).replace(" ", "") + "\n")
                f_pop.close()

            # Fin d'évalutation de la population g
            if self.verbose:
                if self.logger:
                    self.logger.info("Evaluation n°" + str(g) + "ending...")
                else:
                    print("Evaluation n°" + str(g) + "ending...")

        best = []
        min = []

        if self.logger:
            self.logger.info("Genetic Algorithm ending")
        else:
            print("Genetic Algorithm ending")
        for b in best_of_all:
            min.append(b.fitness.values[0])
            best.append(b[0])

            # print best parameters from genetic algorithm
            if self.logger:
                self.logger.info("Best parameters: " + str(b[0]) + " | score: " + str(b.fitness.values[0]))
            else:
                print("Best parameters: " + str(b[0]) + " | score: " + str(b.fitness.values[0]))

        return best, min

    def show(self, filename=None, save=None):

        if filename == None:
            scores = np.array(self.all_scores)
        else:
            data = pd.read_table(filename, sep=",", decimal=".")
            scores = data["loss_value"].to_numpy()

        quantile = np.quantile(scores, 0.75)
        argmin = np.argmin(scores)
        min = scores[argmin]
        heatmap = scores.reshape((int(len(scores) / self.pop_size), self.pop_size))

        minimums = np.min(heatmap, axis=1)
        means = np.mean(heatmap, axis=1)

        heatmap.sort(axis=1)
        heatmap = heatmap.transpose()

        fig1, ax1 = plt.subplots()
        ax = sns.heatmap(heatmap, vmin=min, vmax=quantile, cmap="YlGnBu", cbar_kws={'label': 'Score'}, ax=ax1)
        ax.invert_yaxis()
        ax.set_title("Fitness evolution through generations, \n mininimum=" + str(min))
        ax.set(xlabel='Generation number', ylabel='Individual number')
        fig1.legend()
        fig1.show()
        if save != None:
            fig1.savefig(save + '/ga_heatmap.png')

        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(len(minimums)), minimums, "-", label="Best individual", color="red")
        ax2.plot(np.arange(len(means)), means, ":", label="Mean", color="blue")
        ax2.set_title("Best individual and population's mean through generations")
        ax2.set(xlabel = "Generations", ylabel="Score")
        fig2.legend()
        fig2.show()

        if save != None:
            fig2.savefig(save + '/ga_best_ind_and_pop.png')

        if filename != None:
            self.search_space.show(data, scores)
