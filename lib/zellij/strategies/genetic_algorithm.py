# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-09T15:06:43+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from zellij.core.metaheuristic import Metaheuristic
from zellij.core.addons import Mutator, Selector, Crossover
from deap import base
from deap import creator
from deap import tools
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

logger = logging.getLogger("zellij.GA")


class Genetic_algorithm(Metaheuristic):

    """Genetic_algorithm

    Genetic_algorithm (GA) implements a steady state genetic algorithm. It can be used for exploration and exploitation.
    Indeed when the population has converged, GA can ,thanks to the mutation and crossover operators, perform an intensification phase arround best solutions.
    It can work with a mixed search space, by adapting its operator.

    **Will be modified soon:**
    Used operators are: One-point crossover and Tournament selection of size 3.

    Here the mutation operator is the neighborhood defined in the :ref:`sp` object.
    Available crossover operator are those compatible with a mixed individual (
    `1-point <https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.cxOnePoint>`_,
    `2-points <https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.cxTwoPoint>`_
    ...).
    Same with the slection (
    `Roulette <https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selRoulette>`_,
    `Tournament <https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournament>`_
    ).

    It uses `DEAP <https://deap.readthedocs.io/>`_.
    See :ref:`meta` for more info.

    Attributes
    ----------

    pop_size : int
        Population size of the GA.\
        In a distributed environment (e.g. MPILoss), it has an influence on the parallelization quality.\
        It must be tuned according the available hardware.

    generation : int
        Generation number of the GA.

    elitism : float, default=0.5
        Percentage of the best parents to keep in the next population by replacing the worst children.
        Default 50%.

    filename : str, optional
        If a file containing initial solutions. GA will initialize the population with it.


    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.

    Examples
    --------
    >>> from zellij.core.loss_func import Loss
    >>> from zellij.core.search_space import ContinuousSearchspace
    >>> from zellij.core.variables import FloatVar, ArrayVar
    >>> from zellij.utils.neighborhoods import FloatInterval, ArrayInterval, Intervals
    >>> from zellij.strategies.genetic_algorithm import Genetic_algorithm
    >>> from zellij.utils.operators import NeighborMutation, DeapTournament, DeapOnePoint
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(
    ...                           FloatVar("a",-5,5, neighbor=FloatInterval(0.5)),
    ...                           FloatVar("b",-5,5,neighbor=FloatInterval(0.5)),
    ...                           neighbor=ArrayInterval())
    ...                         ,lf, neighbor=Intervals(),
    ...                         mutation = NeighborMutation(0.5),
    ...                         selection = DeapTournament(3),
    ...                         crossover = DeapOnePoint())
    ...
    >>> ga = Genetic_algorithm(sp, 1000, pop_size=25, generation=40,elitism=0.5)
    >>> ga.run()
    >>> ga.show()


    .. image:: ../_static/ga_sp_ex.png
        :width: 924px
        :align: center
        :height: 487px
        :alt: alternate text
    .. image:: ../_static/ga_res1_ex.png
        :width: 924px
        :align: center
        :height: 487px
        :alt: alternate text
    .. image:: ../_static/ga_res2_ex.png
        :width: 924px
        :align: center
        :height: 487px
        :alt: alternate text
    """

    def __init__(
        self,
        search_space,
        f_calls,
        pop_size=10,
        generation=1000,
        elitism=0.5,
        filename="",
        verbose=True,
    ):

        """__init__(search_space, f_calls, pop_size = 10, generation = 1000, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
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

        elitism : float
            Percentage of the best parents to keep in the next population by replacing the worst children.

        filename : str, optional
            If a file containing initial solutions. GA will initialize the population with it.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, f_calls, verbose)
        assert hasattr(search_space, "mutation") and isinstance(
            search_space.mutation, Mutator
        ), f"""When using :ref:`ga`, :ref:`sp` must have a `mutation` operator
        and of type: Mutator, use `mutation` kwarg when defining the :ref:`sp`
        ex:\n
        >>> ContinuousSearchspace(values, loss, mutation=...)"""

        assert hasattr(search_space, "selection") and isinstance(
            search_space.selection, Selector
        ), f"""When using :ref:`ga`, :ref:`sp` must have a `selection` operator
        and of type: Selector, use `mutation` kwarg when defining the :ref:`sp`
        ex:\n
        >>> ContinuousSearchspace(values, loss, selection=...)"""

        assert hasattr(search_space, "crossover") and isinstance(
            search_space.crossover, Crossover
        ), f"""When using :ref:`ga`, :ref:`sp` must have a `mutation` operator
        and of type: Selector, use `crossover` kwarg when defining the :ref:`sp`
        ex:\n
        >>> ContinuousSearchspace(values, loss, crossover=...)"""

        self.pop_size = pop_size
        self.generation = generation
        self.elitism = elitism

        self.pop_historic = []
        self.fitness_historic = []

        # Population save
        self.filename = filename
        self.ga_save = ""

    # Define what an individual is
    def define_individual(self, sp):
        """define_individual(self)

        Describe how an individual should be initialized. Here a random point from SearchSpace is sampled.

        """
        # Select one random point from the search space
        solution = sp.random_point()

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
        data = pd.read_csv(filename, sep=",", usecols=self.search_space.size)
        contents = data.tail(self.pop_size)

        return pcls(ind_init(c) for index, c in contents.iterrows())

    # Run GA
    def run(self, H=None, n_process=1):

        """run(H=None, n_process=1)

        Runs GA

        Parameters
        ----------
        H : Fractal, optional
            When used by FDA, a fractal corresponding to the current subspace is given

        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        # Progress bar
        self.build_bar(self.generation)

        self.search_space.loss.file_created = False

        logger.info("Starting")

        logger.info("Constructing tools...")

        # Define problem type "fitness", weights = -1.0 -> minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Define what an individual is for the algorithm
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox contains all the operator of GA. (mutate, select, crossover...)
        toolbox = base.Toolbox()

        # Start from a saved population
        if self.filename:

            toolbox.register(
                "individual_guess", self.initIndividual, creator.Individual
            )
            toolbox.register(
                "population_guess",
                self.initPopulation,
                list,
                toolbox.individual_guess,
                self.filename,
            )

            logger.info("Creation of the initial population...")
            pop = toolbox.population_guess()

        # Start from a random population
        else:
            if H:
                sp = H
            else:
                sp = self.search_space

            # Determine what an individual is
            toolbox.register("hyperparameters", self.define_individual, sp)

            # Determine the way to build individuals for the population
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.hyperparameters,
                n=1,
            )

            # Determine the way to build a population
            toolbox.register(
                "population", tools.initRepeat, list, toolbox.individual
            )

            logger.info("Creation of the initial population...")

            # Build the populationze
            pop = toolbox.population(n=self.pop_size)

        # Create operators
        self.search_space.mutation._build(toolbox)
        self.search_space.selection._build(toolbox)
        self.search_space.crossover._build(toolbox)

        # Create a tool to select best individuals from a population
        bpn = int(self.pop_size * self.elitism)
        bcn = self.pop_size - bpn
        toolbox.register("best_p", tools.selBest, k=bpn)
        toolbox.register("best_c", tools.selBest, k=bcn)

        best_of_all = tools.HallOfFame(n_process)

        # Ga initialization

        logger.info("Evaluating the initial population...")
        # Compute dynamically fitnesses
        solutions = []
        solutions = [p[0] for p in pop]

        # Progress bar
        self.pending_pb(len(solutions))
        fitnesses = self.search_space.loss(solutions, generation=0)

        self.update_main_pb(
            len(solutions), explor=True, best=self.search_space.loss.new_best
        )

        # Map computed fitness to individual fitness value
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)

        fits = [ind.fitness.values[0] for ind in pop]

        # Save file
        if self.search_space.loss.save:
            self.ga_save = os.path.join(
                self.search_space.loss.outputs_path, "ga_population.csv"
            )
            with open(self.ga_save, "a") as f:
                """f.write(
                    ",".join(e for e in self.search_space.labels) + ",loss\n"
                )"""
                for ind, cout in zip(pop, fits):
                    f.write(
                        ",".join(str(e) for e in ind[0])
                        + ","
                        + str(cout)
                        + "\n"
                    )

        for ind, cout in zip(pop, fits):
            self.pop_historic.append(ind[0])
            self.fitness_historic.append(cout)

        logger.info("Initial population evaluated")

        logger.info("Evolution starting...")
        g = 0
        while (
            g < self.generation and self.search_space.loss.calls < self.f_calls
        ):
            g += 1

            # Progress bar
            self.meta_pb.update()

            # Update all of fame
            best_of_all.update(pop)

            logger.debug("Generation: " + str(g))

            # Selection operator
            logger.debug("Selection...")

            offspring = self.search_space.selection(pop, k=len(pop))

            children = []

            # Crossover operator
            logger.debug("Crossover...")

            i = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # Clone individuals from crossover
                children1 = toolbox.clone(child1)
                children2 = toolbox.clone(child2)

                # Apply crossover
                self.search_space.crossover(children1[0], children2[0])
                # Delete children fitness inherited from the parents
                del children1.fitness.values
                del children2.fitness.values

                # Add new children to list
                children.append(children1)
                children.append(children2)

            # Mutate children
            logger.debug("Mutation...")

            for mutant in children:
                toolbox.mutate(mutant[0])

            logger.debug("Evaluating population n°" + str(g))

            # Compute dynamically fitnesses
            solutions = []
            solutions = [p[0] for p in children]

            # progress bar
            self.pending_pb(len(solutions))

            fitnesses = self.search_space.loss(solutions, generation=g)

            # Progress bar
            self.update_main_pb(
                len(solutions),
                explor=True,
                best=self.search_space.loss.new_best,
            )

            # Map computed fitness to individual fitness value
            for ind, fit in zip(children, fitnesses):
                ind.fitness.values = (fit,)

            # Build new population
            pop[:] = toolbox.best_p(offspring) + toolbox.best_c(children)

            # Get fitnesses from the new population
            fits = [ind.fitness.values[0] for ind in pop]

            # Save new population
            """if self.search_space.loss.save:
                with open(self.ga_save, "a") as f:
                    for ind, cout in zip(pop, fits):
                        f.write(
                            ",".join(str(e) for e in ind[0])
                            + ","
                            + str(cout)
                            + "\n"
                        )"""

            for ind, cout in zip(pop, fits):
                self.pop_historic.append(ind[0])
                self.fitness_historic.append(cout)

            # End population evaluation
            logger.info(f"Evaluation n°{g} ending...")

        best = []
        min = []

        logger.info("Ending")
        for b in best_of_all:
            min.append(b.fitness.values[0])
            best.append(b[0])

            # print best parameters from genetic algorithm
            logger.info(
                "Best parameters: "
                + str(b[0])
                + " | score: "
                + str(b.fitness.values[0])
            )

        self.close_bar()
        return best, min

    def show(self, filepath="", save=False):

        """show(filepath="")

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        all_data, all_scores = super().show(filepath, save)

        if filepath:
            gapth = os.path.join(filepath, "outputs", "ga_population.csv")
            data = pd.read_table(gapth, sep=",", decimal=".")
            scores = data["loss"].to_numpy()
        else:
            data = self.pop_historic
            scores = np.array(self.fitness_historic)

        quantile = np.quantile(scores, 0.75)
        argmin = np.argmin(scores)
        min = scores[argmin]

        # Padding missing individual for reshape
        m, n = int(np.ceil(len(scores) / self.pop_size)), self.pop_size
        scores = np.pad(
            scores,
            (0, m * n - scores.size),
            mode="constant",
            constant_values=np.nan,
        )

        heatmap = scores.reshape((m, n))

        minimums = np.min(heatmap, axis=1)
        means = np.mean(heatmap, axis=1)

        heatmap.sort(axis=1)
        heatmap = heatmap.transpose()

        fig, ax = plt.subplots(figsize=(19.2, 14.4))
        ax = sns.heatmap(
            heatmap,
            vmin=min,
            vmax=quantile,
            cmap="YlGnBu",
            cbar_kws={"label": "Fitness"},
        )
        ax.invert_yaxis()
        ax.set_title(f"Fitness of each individual through generations")
        ax.set(xlabel="Generation number", ylabel="Individual number")

        if save:
            save_path = os.path.join(
                self.search_space.loss.plots_path, f"heatmap_ga.png"
            )
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()

        fig, ax = plt.subplots(figsize=(19.2, 14.4))
        plt.plot(
            np.arange(len(minimums)),
            minimums,
            "-",
            label="Best individual",
            color="red",
        )
        plt.plot(np.arange(len(means)), means, ":", label="Mean", color="blue")
        plt.title("Best individual and population's mean through generations")
        plt.xlabel("Generations")
        plt.ylabel("Score")
        plt.legend()

        if save:
            save_path = os.path.join(
                self.search_space.loss.plots_path, f"lineplot_ga.png"
            )

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()
