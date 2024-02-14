# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.metaheuristic import Metaheuristic
from zellij.core.errors import InputError

from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.addons import Mutation, Selection, Crossover
    from zellij.core.search_space import Searchspace

from deap import creator, base, tools
import numpy as np

import logging

logger = logging.getLogger("zellij.GA")


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Define what an individual is for the algorithm
creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore


class GeneticAlgorithm(Metaheuristic):

    """GeneticAlgorithm

    GeneticAlgorithm (GA) implements a usual genetic algorithm.

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
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.mixed import GeneticAlgorithm
    >>> from zellij.strategies.tools import DeapOnePoint, DeapTournament, NeighborMutation
    >>> from zellij.utils import ArrayDefaultN, FloatInterval

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}
    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, neighborhood=FloatInterval(0.5)),
    ...     FloatVar("i2", -5, 5, neighborhood=FloatInterval(0.5)),
    ...     neighborhood=ArrayDefaultN(),
    ... )
    >>> sp = ContinuousSearchspace(a)
    >>> selection = DeapTournament(4)
    >>> mutation = NeighborMutation(1.0)
    >>> crossover = DeapOnePoint()
    >>> opt = GeneticAlgorithm(sp, selection, mutation, crossover, 20)
    >>> stop = Calls(himmelblau, 400)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([2.9854881033107623, 1.9529960098733676])=0.058047272291031515
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 400
    """

    def __init__(
        self,
        search_space: Searchspace,
        selection: Selection,
        mutation: Mutation,
        crossover: Crossover,
        pop_size: int = 10,
        elitism: float = 0.5,
        verbose: bool = True,
    ):
        """__init__(search_space, pop_size = 10, verbose=True)

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

        """
        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############
        self.pop_size = pop_size
        self.elitism = elitism
        logger.info("Making tools...")
        # toolbox contains all the operator of GA. (mutate, select, crossover...)
        self.toolbox = base.Toolbox()

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

        #############
        # VARIABLES #
        #############
        self.initialized = False
        self.first_offspring = False
        self.pop = []

        self.g = 0

    @property
    def mutation(self) -> Mutation:
        return self._mutation

    @mutation.setter
    def mutation(self, value: Mutation):
        value.target = self.search_space
        value._build(self.toolbox)
        self._mutation = value

    @property
    def selection(self) -> Selection:
        return self._selection

    @selection.setter
    def selection(self, value: Selection):
        value.target = self.search_space
        value._build(self.toolbox)
        self._selection = value

    @property
    def crossover(self) -> Crossover:
        return self._crossover

    @crossover.setter
    def crossover(self, value: Crossover):
        value.target = self.search_space
        value._build(self.toolbox)
        self._crossover = value

    @property
    def toolbox(self):
        return self._toolbox

    @toolbox.setter
    def toolbox(self, value):
        self._toolbox = value
        # Create a tool to select best individuals from a population
        bpn = int(self.pop_size * self.elitism)
        bcn = self.pop_size - bpn
        value.register("best_p", tools.selBest, k=bpn)
        value.register("best_c", tools.selBest, k=bcn)

        value.register(
            "individual_guess",
            self.initIndividual,
            creator.Individual,  # type: ignore
        )
        value.register(
            "population_guess",
            self.initPopulation,
            list,
            self.toolbox.individual_guess,
        )

        # Determine what an individual is
        value.register("hyperparameters", self.search_space.random_point)

        # Determine the way to build individuals for the population
        value.register(
            "individual",
            tools.initRepeat,
            creator.Individual,  # type: ignore
            value.hyperparameters,
            n=1,
        )

        # Determine the way to build a population
        value.register(
            "population",
            tools.initRepeat,
            list,
            value.individual,
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

    # Run Random
    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of GA.

        Secondary, or constraints are not necessary.

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
            self.initialized = True
            # Start from a saved population
            if X is None and Y is None:
                # Build the population
                logger.info("Creation of the initial population...")
                pop = self.toolbox.population(n=self.pop_size)

                logger.info("Evaluating the initial population...")
                solutions = [p[0] for p in pop]
                return solutions, {"algorithm": "GA", "generation": 0}

        if X is None:
            raise InputError(
                "In GeneticAlgorithm,  X and Y cannot be of NoneType after initialization."
            )
        elif Y is None:
            return X, {"algorithm": "GA", "generation": 0}
        else:
            o_fitnesses = Y
            offspring = self.toolbox.population_guess(X)

            # Map computed fitness to individual fitness value
            for ind, fit in zip(offspring, o_fitnesses):
                ind.fitness.values = (fit,)

            if self.first_offspring:
                # Build new population
                self.pop[:] = self.toolbox.best_p(self.pop) + self.toolbox.best_c(
                    offspring
                )

            else:
                # Initialize computed population
                self.pop = offspring[:]
                self.first_offspring = True

            self.g += 1

            logger.debug(f"Generation: {self.g}")

            # Selection operator
            logger.debug("Selection...")

            offspring = self.selection(self.pop, k=len(self.pop))

            children = []

            # Crossover operator
            logger.debug("Crossover...")

            i = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # Clone individuals from crossover
                children1 = self.toolbox.clone(child1)
                children2 = self.toolbox.clone(child2)

                # Apply crossover
                self.crossover(children1[0], children2[0])
                # Delete children fitness inherited from the parents
                del children1.fitness.values
                del children2.fitness.values

                # Add new children to list
                children.append(children1)
                children.append(children2)

            # Mutate children
            logger.debug("Mutation...")

            for mutant in children:
                self.mutation(mutant[0])

            solutions = [p[0] for p in children]

            # End population evaluation
            logger.info(f"Evaluating n°{self.g}...")

            return solutions, {"algorithm": "GA", "generation": self.g}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_toolbox"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.toolbox = base.Toolbox()


class SteadyStateGA(Metaheuristic):

    """SteadyStateGA

    Steady State genetic algorithm.

    It uses `DEAP <https://deap.readthedocs.io/>`__.
    See :ref:`meta` for more info.

    Attributes
    ----------
    pop_size : int
        Initial population size.


    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.

    Examples
    --------
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.mixed import SteadyStateGA
    >>> from zellij.strategies.tools import DeapOnePoint, DeapTournament, NeighborMutation
    >>> from zellij.utils import ArrayDefaultN, FloatInterval

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}
    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, neighborhood=FloatInterval(0.5)),
    ...     FloatVar("i2", -5, 5, neighborhood=FloatInterval(0.5)),
    ...     neighborhood=ArrayDefaultN(),
    ... )
    >>> sp = ContinuousSearchspace(a)
    >>> selection = DeapTournament(4)
    >>> mutation = NeighborMutation(1.0)
    >>> crossover = DeapOnePoint()
    >>> opt = SteadyStateGA(sp, selection, mutation, crossover, 20)
    >>> stop = Calls(himmelblau, 400)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([3.0211050307215714, 2.005915799414448])=0.01969404306327753
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 400

    """

    def __init__(
        self,
        search_space: Searchspace,
        selection: Selection,
        mutation: Mutation,
        crossover: Crossover,
        pop_size: int = 10,
        elitism: float = 0.5,
        verbose: bool = True,
    ):
        """__init__(search_space, pop_size = 10, verbose=True)

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

        """
        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############
        self.pop_size = pop_size
        self.elitism = elitism
        logger.info("Making tools...")
        # toolbox contains all the operator of GA. (mutate, select, crossover...)
        self.toolbox = base.Toolbox()

        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

        #############
        # VARIABLES #
        #############
        self.initialized = False
        self.first_offspring = False
        self.pop = []

        self.g = 0

    @property
    def mutation(self) -> Mutation:
        return self._mutation

    @mutation.setter
    def mutation(self, value: Mutation):
        value.target = self.search_space
        value._build(self.toolbox)
        self._mutation = value

    @property
    def selection(self) -> Selection:
        return self._selection

    @selection.setter
    def selection(self, value: Selection):
        value.target = self.search_space
        value._build(self.toolbox)
        self._selection = value

    @property
    def crossover(self) -> Crossover:
        return self._crossover

    @crossover.setter
    def crossover(self, value: Crossover):
        value.target = self.search_space
        value._build(self.toolbox)
        self._crossover = value

    @property
    def toolbox(self):
        return self._toolbox

    @toolbox.setter
    def toolbox(self, value):
        self._toolbox = value
        # Create a tool to select best individuals from a population
        bpn = int(self.pop_size * self.elitism)
        bcn = self.pop_size - bpn
        value.register("best_p", tools.selBest, k=bpn)
        value.register("best_c", tools.selBest, k=bcn)

        value.register(
            "individual_guess",
            self.initIndividual,
            creator.Individual,  # type: ignore
        )
        value.register(
            "population_guess",
            self.initPopulation,
            list,
            self.toolbox.individual_guess,
        )

        # Determine what an individual is
        value.register("hyperparameters", self.search_space.random_point)

        # Determine the way to build individuals for the population
        value.register(
            "individual",
            tools.initRepeat,
            creator.Individual,  # type: ignore
            value.hyperparameters,
            n=1,
        )

        # Determine the way to build a population
        value.register(
            "population",
            tools.initRepeat,
            list,
            value.individual,
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
        selected = self.selection(self.pop, k=2)

        # Clone individuals from crossover
        children1 = self.toolbox.clone(selected[0])
        children2 = self.toolbox.clone(selected[1])

        # Crossover operator
        logger.debug("Crossover...")

        # Apply crossover
        self.crossover(children1[0], children2[0])
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
    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of GA.

        Secondary, or constraints are not necessary.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`


        """
        if not self.initialized:
            self.g = 0
            self.initialized = True
            # Start from a saved population
            if X is None and Y is None:
                # Build the population
                logger.info("Creation of the initial population...")
                pop = self.toolbox.population(n=self.pop_size)

                logger.info("Evaluating the initial population...")
                solutions = [p[0] for p in pop]
                return solutions, {"algorithm": "GA", "generation": 0}

        if X is None:
            raise InputError(
                "In SteadyStateGA,  X and Y cannot be of NoneType after initialization."
            )
        elif Y is None:
            return X, {"algorithm": "GA", "generation": 0}
        else:
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
                return [self.search_space.random_point(1)], {"algorithm": "SSGA"}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_toolbox"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.toolbox = base.Toolbox()
