from zellij.core.metaheuristic import Metaheuristic
from zellij.core.loss_func import FDA_loss_func

import numpy as np
import copy

import logging

logger = logging.getLogger("zellij.FDA")


class FDA(Metaheuristic):

    """FDA

    Fractal Decomposition Algorithm (FDA) is composed of 5 part:
        –  Fractal decomposition : FDA uses hyper-spheres or hyper-cubes to decompose the search-space into smaller sub-spaces in a fractal way.
        –  Tree search algorithm : Fractals form a tree, so FDA is also a tree search problem. It can use Best First Search, Beam Search or others algorithms from the A* family.
        –  Exploration : To explore a fractal, FDA requires an exploration algorithm, for example GA,or in our case CGS.
        –  Exploitation : At the final fractal level (e.g. a leaf of the rooted tree) FDA performs an exploitation.
        –  Scoring method: To score a fractal, FDA can use the best score found, the median, ... See heuristics.py.

    It a continuous optimization algorithm. SO the search space is converted to continuous.

    Attributes
    ----------

    heuristic : callable
        Determine using using current state of the algorithm, how to score the current fractal. Used informations are given to the function at the following order:
        - The current fractal
        - The best solution found so far (converted to continuous)
        - The best score found so far (computed with the loss function)

    exploration : Metaheuristic
        At each node of a fractal FDA applies an exploration algorithm to determine if this fractal is promising or not.

    exploitation : Metaheuristic
        At a leaf of the rooted fractal tree, FDA applies an exploitation algorithm, which ignores subspace bounds (not SearchSpace bounds),
        to refine the best solution found inside this fractal.

    level : int
        Depth of the fractal tree

    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous

    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous

    fractal : Fractal
        Fractal object used to build the fractal tree

    explor_kwargs : list[dict]
        List of keyword arguments to pass to the exploration strategy at each level of the tree.
        If len(explor_kwargs) < level, then that last element of the list will be used for the next levels.

    explor_kwargs : dict
        Keyword arguments to pass to the exploitation strategy.

    start_H : Fractal
        Root of the fractal tree

    tree_search : Tree_search
        Tree_search object to use to explore and exploit the fractal tree.

    n_h : int
        Number of explored nodes of the tree

    total_h : int
        Theoretical number of nodes.


    Methods
    -------

    evaluate(hypervolumes)
        Evaluate a list of fractals using exploration and/or exploitation

    run(n_process=1)
        Runs FDA

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a search space is in Zellij
    Tree_search : Tree search algorithm to explore and exploit the fractal tree.
    Fractal : Base class which defines what a fractal is.
    """

    def __init__(self, loss_func, search_space, f_calls, exploration, exploitation, fractal, tree_search, heuristic, level=5, verbose=True, **kwargs):

        """__init__(loss_func, search_space, f_calls, exploration, exploitation, fractal, tree_search, heuristic, level=5, verbose=True, **kwargs)

        Initialize FDA class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        exploration : {Metaheuristic, list[Metaheuristic]}
            At each node of a fractal FDA applies an exploration algorithm to determine if this fractal is promising or not.
            If a list of metaheuristic is given, at each level FDA will use the metaheuristic at the index equel to the current level.
            If len(exploration) < level, the last metaheuristic will be used for following levels.

        exploitation : Metaheuristic
            At a leaf of the rooted fractal tree, FDA applies an exploitation algorithm, which ignores subspace bounds (not SearchSpace bounds),
            to refine the best solution found inside this fractal.

        level : int, default=5
            Fractal tree depth.

        tree_search : Tree_search
            BFS : Breadth first search
            DFS : Depth first search
            BS : Beam_search
            BestFS : Best first search
            CBFS : Cyclic best first search
            DBFS : Diverse best first search
            EGS : Epsilon greedy search

        fractal : Fractal
            Fractal used to build the fractal tree

        heuristic : callable
            Determine using using current state of the algorithm, how to score the current fractal. Used informations are given to the function at the following order:
            - The current fractal
            - The best solution found so far (converted to continuous)
            - The best score found so far (computed with the loss function)

        verbose : boolean, default=True
            Algorithm verbosity

        **kwargs : dict
            Keyword arguments for fractals and tree search algorithm.

        """

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func, search_space, f_calls, verbose)

        self.heuristic = heuristic
        self.level = level

        # Exploration and exploitation function
        if type(exploration) != list:
            self.exploration = [exploration]
        else:
            self.exploration = exploration

        if exploitation:
            self.exploitation = exploitation
            self.exploi_calls = self.exploitation.f_calls
        else:
            self.exploitation = False

        #############
        # VARIABLES #
        #############

        # Save f_calls from metaheuristic, to adapt them during FDA.
        self.explor_calls = [i.f_calls for i in self.exploration]

        # Working variables
        self.up_bounds = np.array([1.0 for _ in self.search_space.values])
        self.lo_bounds = np.array([0.0 for _ in self.search_space.values])

        self.fractal = fractal
        self.tree_search = tree_search

        self.volume_kwargs = {key: kwargs[key] for key in kwargs if key in self.fractal.__init__.__code__.co_varnames}
        self.ts_kwargs = {key: kwargs[key] for key in kwargs if key in self.tree_search.__init__.__code__.co_varnames}

        # Initialize first fractal
        self.root = self.fractal(self.lo_bounds, self.up_bounds, **self.volume_kwargs)

        # Initialize tree search
        self.tree_search = self.tree_search([self.root], self.level, **self.ts_kwargs)

        # Number of explored hypersphere
        self.n_h = 0

        self.executed = False  # A voir

        # Best solution converted to continuous
        self.best_ind_c = []

        self.n_process = 1

    # Evaluate a list of hypervolumes
    def evaluate(self, hypervolumes):

        """evaluate(hypervolumes)

        Evaluate a list of fractals using exploration and/or exploitation.

        Parameters
        ----------
        hypervolumes : list[Fractal]
            list of hypervolumes to evaluate with exploration and/or exploitation

        """

        # While there are hypervolumes to evaluate do...
        i = 0
        while i < len(hypervolumes) and self.loss_func.calls < self.f_calls:

            # Select parent hypervolume
            H = hypervolumes[i]
            j = 0

            # While there are children do...
            while j < len(H.children) and self.loss_func.calls < self.f_calls:

                # Select children of parent H
                child = H.children[j]

                j += 1

                # Link the loss function to the actual hypervolume (children)
                modified_loss_func = FDA_loss_func(self.loss_func, child, self.search_space)

                # Count the number of explored hypervolume
                self.n_h += 1

                # Exploration
                if child.level != self.level:

                    explor_idx = np.min([child.level, len(self.exploration)]) - 1
                    calls_left = np.min([self.explor_calls[explor_idx], self.f_calls - self.loss_func.calls])

                    if calls_left > 0:

                        # Compute bounds of child hypervolume
                        lo = self.search_space.convert_to_continuous([child.lo_bounds], True, True)[0]
                        up = self.search_space.convert_to_continuous([child.up_bounds], True, True)[0]

                        # Create a search space for the metaheuristic
                        sp = self.search_space.subspace(lo, up)
                        self.exploration[explor_idx].search_space = sp
                        self.exploration[explor_idx].loss_func = modified_loss_func

                        self.exploration[explor_idx].f_calls = calls_left + self.loss_func.calls

                        logger.info(f"Exploration {self.fractal.__name__} n° {child.id} child of {child.father.id} at level {child.level}")
                        logger.info(f"Explored {self.fractal.__name__}: {self.n_h}")

                        # Progress bar
                        prec_calls = self.loss_func.calls

                        # Run exploration, scores and evaluated solutions are saved using FDA_loss_func class
                        self.exploration[explor_idx].run(H=child, n_process=self.n_process)

                        # Save best found solution
                        if self.loss_func.new_best:
                            logger.info(f"Best solution found :{self.loss_func.best_score}")
                            self.best_ind_c = self.search_space.convert_to_continuous([self.loss_func.best_sol])[0]

                        child.score = self.heuristic(child, self.best_ind_c, self.loss_func.best_score)

                        logger.debug(f"Child {child.father.id}.{child.id}.{child.level} score: {child.score}")

                        # Add child to tree search
                        self.tree_search.add(child)

                        # Progress bar
                        self.pending_pb(self.loss_func.calls - prec_calls)
                        self.update_main_pb(self.loss_func.calls - prec_calls, explor=True, best=self.loss_func.new_best)
                        self.meta_pb.update(self.loss_func.calls - prec_calls)

                # Exploitation
                elif self.exploitation:

                    # Run exploitation, scores and evaluated solutions are saved using FDA_loss_func class
                    self.exploitation.loss_func = modified_loss_func
                    calls_left = np.min([self.exploi_calls, self.f_calls - self.loss_func.calls])

                    if calls_left > 0:

                        self.exploitation.f_calls = calls_left + self.loss_func.calls

                        logger.info(f" -> Exploitation {self.fractal.__name__} n° {child.id} child of {child.father.id} at level {child.level}")
                        logger.info(f"Explored {self.fractal.__name__}: {self.n_h}")

                        # Progress bar
                        prec_calls = self.loss_func.calls

                        # Run exploitation, scores and evaluated solutions are saved using FDA_loss_func class
                        self.exploitation.run(H=child, n_process=self.n_process)

                        if self.loss_func.new_best:
                            logger.info(f"Best solution found :{self.loss_func.best_score}")
                            self.best_ind_c = self.search_space.convert_to_continuous([self.loss_func.best_sol])[0]

                        logger.debug(f"Child {child.father.id}.{child.id}.{child.level} score: EXPLOITaTION")

                        # Progress bar
                        self.pending_pb(self.loss_func.calls - prec_calls)
                        self.update_main_pb(self.loss_func.calls - prec_calls, explor=False, best=self.loss_func.new_best)
                        self.meta_pb.update(self.loss_func.calls - prec_calls)

            i += 1

    def run(self, H=None, n_process=1):

        """run(H=None, n_process=1)

        Runs FDA.

        Parameters
        ----------
        H : Fractal, default=None
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

        if H != None:
            logger.warning(f"Do not give any fractal to FDA.run")

        self.build_bar(self.f_calls)

        for exp in self.exploration:
            exp.manager = self.manager

        logger.info("Starting")

        if self.exploitation:
            self.exploitation.manager = self.manager

        self.n_process = n_process

        self.n_h = 0

        stop = True

        # Select initial hypervolume (root) from the search tree
        stop, hypervolumes = self.tree_search.get_next()

        while stop and self.loss_func.calls < self.f_calls:

            for H in hypervolumes:
                H.create_children()

            self.evaluate(hypervolumes)

            stop, hypervolumes = self.tree_search.get_next()

        self.executed = True

        logger.info(f"Loss function calls: {self.loss_func.calls}")
        logger.info(f"Explored {self.fractal.__name__}: {self.n_h}")
        logger.info(f"Best score: {self.loss_func.best_score}")
        logger.info(f"Best solution: {self.loss_func.best_sol}")

        best_idx = np.argpartition(self.loss_func.all_scores, n_process)
        best = [self.loss_func.all_solutions[i] for i in best_idx[:n_process]]
        min = [self.loss_func.all_scores[i] for i in best_idx[:n_process]]

        self.close_bar()

        logger.info("Ending")

        return best, min

    def show(self, filepath="", save=False):

        """show(filename=None)

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        """

        super().show(filepath, save)
