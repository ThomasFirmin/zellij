# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-06T17:28:46+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic
from zellij.core.addons import Mutator, Selector, Crossover
from deap import base
from deap import creator
from deap import tools
import numpy as np
import os
import pandas as pd

import logging

logger = logging.getLogger("zellij.GA")


class Genetic_algorithm(Metaheuristic):

    """Genetic_algorithm

    Genetic_algorithm (GA) implements a classic genetic algorithm.

    It uses `DEAP <https://deap.readthedocs.io/>`__.
    See :ref:`meta` for more info.

    Attributes
    ----------

    search_space : Searchspace
        Search space object containing bounds of the search space.

    pop_size : int
        Population size.

    elitism : float
        Percentage of the best parents to keep in the next population by replacing the worst children.

    verbose : boolean, default=True
        Algorithm verbosity

    init_pop : boolean, default=None
        If a a initial population (list of individuals) is given.
        The population will be initialized with :code:`init_pop`,
        otherwise randomly .


    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.

    Examples
    --------
    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.neighborhoods import FloatInterval, ArrayInterval, Intervals
    >>> from zellij.strategies.genetic_algorithm import Genetic_algorithm
    >>> from zellij.utils.operators import NeighborMutation, DeapTournament, DeapOnePoint
    >>> from zellij.utils.benchmarks import himmelblau
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
    >>> stop = Threshold(lf, 'calls', 100)
    >>> ga = Genetic_algorithm(sp, 1000, pop_size=25, generation=40,elitism=0.5)
    >>> exp = Experiment(ga, stop)
    >>> exp.run()
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")
    """

    def __init__(
        self,
        search_space,
        pop_size=10,
        elitism=0.5,
        verbose=True,
        init_pop=None,
    ):

        """__init__(search_space, pop_size = 10, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        pop_size : int
            Population size.

        elitism : float
            Percentage of the best parents to keep in the next population by replacing the worst children.

        verbose : boolean, default=True
            Algorithm verbosity

        init_pop : boolean, default=None
            If a a initial population (list of individuals) is given.
            The population will be initialized with :code:`init_pop`,
            otherwise randomly .

        """

        super().__init__(search_space, verbose)
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

        ##############
        # PARAMETERS #
        ##############

        self.pop_size = pop_size
        self.elitism = elitism
        self.init_pop = init_pop

        #############
        # VARIABLES #
        #############

        self.initialized = False
        self.first_offspring = False

        logger.info("Constructing tools...")

        # Define problem type "fitness", weights = -1.0 -> minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Define what an individual is for the algorithm
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # toolbox contains all the operator of GA. (mutate, select, crossover...)
        self.toolbox = base.Toolbox()

        # Create operators
        self.search_space.mutation._build(self.toolbox)
        self.search_space.selection._build(self.toolbox)
        self.search_space.crossover._build(self.toolbox)

        # Create a tool to select best individuals from a population
        bpn = int(self.pop_size * self.elitism)
        bcn = self.pop_size - bpn
        self.toolbox.register("best_p", tools.selBest, k=bpn)
        self.toolbox.register("best_c", tools.selBest, k=bcn)

        self.toolbox.register(
            "individual_guess",
            self.initIndividual,
            creator.Individual,
        )
        self.toolbox.register(
            "population_guess",
            self.initPopulation,
            list,
            self.toolbox.individual_guess,
        )

    # Initialize an individual extracted from a file
    def initIndividual(self, icls, content):
        """initIndividual(self, icls, content)

        Initialize an individual to DEAP.

        """
        return icls([content])

    # Initialize a population extracted from a file
    def initPopulation(self, pcls, ind_init, X):
        """initPopulation(self, pcls, ind_init, X)

        Initialize a population of individual, from a given population.

        """

        return pcls(ind_init(c) for index, c in enumerate(X))

    def reset(self):
        """reset()

        Reset GA variables to their initial values.

        """
        self.initialized = False
        self.first_offspring = False

    # Run GA
    def forward(self, X, Y):

        """forward(X, Y)
        Runs one step of Genetic_algorithm.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        logger.info("GA Starting")

        if not self.initialized:
            self.g = 0
            # Start from a saved population
            if self.init_pop:
                logger.info("Creation of the initial population...")
                pop = self.toolbox.population_guess(self.init_pop)

            # Start from a random population
            else:
                # Determine what an individual is
                self.toolbox.register(
                    "hyperparameters", self.search_space.random_point
                )

                # Determine the way to build individuals for the population
                self.toolbox.register(
                    "individual",
                    tools.initRepeat,
                    creator.Individual,
                    self.toolbox.hyperparameters,
                    n=1,
                )

                # Determine the way to build a population
                self.toolbox.register(
                    "population",
                    tools.initRepeat,
                    list,
                    self.toolbox.individual,
                )

                logger.info("Creation of the initial population...")

                # Build the population
                pop = self.toolbox.population(n=self.pop_size)

            logger.info("Evaluating the initial population...")
            solutions = [p[0] for p in pop]

            self.initialized = True

            return solutions, {"algorithm": "GA", "generation": 0}

        fitnesses = Y
        pop = self.toolbox.population_guess(X)

        # Map computed fitness to individual fitness value
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)

        if self.first_offspring:
            # Build new population
            pop[:] = self.toolbox.best_p(self.offspring) + self.toolbox.best_c(
                pop
            )
        else:
            self.first_offspring = True

        self.g += 1

        logger.debug(f"Generation: {self.g}")

        # Selection operator
        logger.debug("Selection...")

        self.offspring = self.search_space.selection(pop, k=len(pop))

        children = []

        # Crossover operator
        logger.debug("Crossover...")

        i = 0
        for child1, child2 in zip(self.offspring[::2], self.offspring[1::2]):

            # Clone individuals from crossover
            children1 = self.toolbox.clone(child1)
            children2 = self.toolbox.clone(child2)

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
            self.toolbox.mutate(mutant[0])

        solutions = [p[0] for p in children]

        # End population evaluation
        logger.info(f"Evaluating n°{self.g}...")

        return solutions, {"algorithm": "GA", "generation": self.g}


class Steady_State_GA(Metaheuristic):

    """Steady_State_GA

    Steady State genetic algorithm.

    It uses `DEAP <https://deap.readthedocs.io/>`__.
    See :ref:`meta` for more info.

    Attributes
    ----------

    pop_size : int
        Initial population size.

    generation : int
        Generation number of the GA.

    elitism : float, default=0.5
        Percentage of the best parents to keep in the next population by replacing the worst children.
        Default 50%.


    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.

    Examples
    --------
    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core import FloatVar, ArrayVar
    >>> from zellij.utils.neighborhoods import FloatInterval, ArrayInterval, Intervals
    >>> from zellij.strategies.genetic_algorithm import Genetic_algorithm
    >>> from zellij.utils.operators import NeighborMutation, DeapTournament, DeapOnePoint
    >>> from zellij.utils.benchmarks import himmelblau
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
    """

    def __init__(
        self,
        search_space,
        pop_size=10,
        verbose=True,
    ):

        """__init__(search_space, pop_size = 10, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        pop_size : int
            Population size.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)
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

        ##############
        # PARAMETERS #
        ##############

        self.pop_size = pop_size

        #############
        # VARIABLES #
        #############

        self.initialized = False
        self.first_offspring = False

        logger.info("Constructing tools...")

        # Define problem type "fitness", weights = -1.0 -> minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Define what an individual is for the algorithm
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # toolbox contains all the operator of GA. (mutate, select, crossover...)
        self.toolbox = base.Toolbox()

        # Create operators
        self.search_space.mutation._build(self.toolbox)
        self.search_space.selection._build(self.toolbox)
        self.search_space.crossover._build(self.toolbox)

        self.toolbox.register(
            "individual_guess",
            self.initIndividual,
            creator.Individual,
        )

        self.toolbox.register(
            "population_guess",
            self.initPopulation,
            list,
            self.toolbox.individual_guess,
        )

    # Initialize an individual extracted from a file
    def initIndividual(self, icls, content):
        """initIndividual(self, icls, content)

        Initialize an individual to DEAP.

        """
        return icls([content])

    # Initialize a population extracted from a file
    def initPopulation(self, pcls, ind_init, X):
        """initPopulation(self, pcls, ind_init, X)

        Initialize a population of individual, from a given population.

        """

        return pcls(ind_init(c) for index, c in enumerate(X))

    def reset(self):
        """reset()

        Reset GA variables to their initial values.

        """
        self.initialized = False
        self.first_offspring = False

    def _do_selcrossmut(self):
        # Selection operator
        logger.debug("Selection...")

        new_x = []
        selected = self.search_space.selection(self.pop, k=2)

        # Clone individuals from crossover
        children1 = self.toolbox.clone(selected[0])
        children2 = self.toolbox.clone(selected[1])

        # Crossover operator
        logger.debug("Crossover...")

        # Apply crossover
        self.search_space.crossover(children1[0], children2[0])
        # Delete children fitness inherited from the parents
        del children1.fitness.values
        del children2.fitness.values

        # Add new children to list
        new_x.append(children1)
        new_x.append(children2)

        # Mutate children
        logger.debug("Mutation...")
        self.toolbox.mutate(new_x[0][0])
        self.toolbox.mutate(new_x[1][0])

        solutions = [p[0] for p in new_x]

        return solutions

    # Run GA
    def forward(self, X, Y):

        """forward(X, Y)
        Runs one step of Genetic_algorithm.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        logger.info("GA Starting")
        if not self.initialized:
            self.pop = []
            if X:
                self.initialized = True
                if not Y:
                    return X, {"algorithm": "GA"}
                else:
                    # Map computed fitness to individual fitness value
                    for ind, fit in zip(self.pop, Y):
                        ind.fitness.values = (fit,)
                    return self._do_selcrossmut(), {"algorithm": "GA"}
            else:
                # Determine what an individual is
                self.toolbox.register(
                    "hyperparameters", self.search_space.random_point
                )

                # Determine the way to build individuals for the population
                self.toolbox.register(
                    "individual",
                    tools.initRepeat,
                    creator.Individual,
                    self.toolbox.hyperparameters,
                    n=1,
                )

                # Determine the way to build a population
                self.toolbox.register(
                    "population",
                    tools.initRepeat,
                    list,
                    self.toolbox.individual,
                )

                logger.info("Creation of the initial population...")

                # Build the population
                new_pop = self.toolbox.population(n=self.pop_size)

                logger.info("Evaluating the initial population...")
                solutions = [p[0] for p in new_pop]

                self.initialized = True

                return solutions, {"algorithm": "GA"}

        if (X and Y) and (len(X) > 0 and len(Y) > 0):
            children = []
            # Map computed fitness to individual fitness value
            for ind, fit in zip(X, Y):
                if np.isfinite(fit):
                    ind = self.toolbox.individual_guess(ind)
                    ind.fitness.values = (fit,)
                    children.append(ind)

            if len(children) > 0:
                offspring = self.pop + children
                min_pop = np.minimum(len(offspring), self.pop_size)
                self.pop = tools.selBest(offspring, k=min_pop)

        if len(self.pop) >= self.pop_size:
            sol = self._do_selcrossmut()
            return sol, {"algorithm": "GA"}
        else:
            return [self.search_space.random_point(1)], {"algorithm": "GA"}
