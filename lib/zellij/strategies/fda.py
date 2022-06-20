# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-13T13:56:55+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from zellij.core.metaheuristic import Metaheuristic

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

    def __init__(
        self,
        search_space,
        f_calls,
        tree_search,
        exploration=None,
        exploitation=None,
        verbose=True,
        **kwargs,
    ):

        """__init__(loss_func, search_space, f_calls, exploration, exploitation, fractal, tree_search, heuristic, level=5, verbose=True, **kwargs)

        Initialize FDA class

        Parameters
        ----------
        search_space : Fractal
            Search space object containing bounds of the search space

        f_calls : int
            Maximum number of :ref:`lf` calls

        exploration : {Metaheuristic, list[Metaheuristic]}, default=None
            At each node of a fractal FDA applies an exploration algorithm to determine if this fractal is promising or not.
            If a list of metaheuristic is given, at each level FDA will use the metaheuristic at the index equel to the current level.
            If len(exploration) < level, the last metaheuristic will be used for following levels.

        exploitation : Metaheuristic, default=None
            At a leaf of the rooted fractal tree, FDA applies an exploitation algorithm, which ignores subspace bounds (not SearchSpace bounds),
            to refine the best solution found inside this fractal.

        tree_search : Tree_search
            BFS : Breadth first search
            DFS : Depth first search
            BS : Beam_search
            BestFS : Best first search
            CBFS : Cyclic best first search
            DBFS : Diverse best first search
            EGS : Epsilon greedy search

        level : int, default=5
            Fractal tree depth.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super(FDA, self).__init__(search_space, f_calls, verbose)

        # Exploration and exploitation function
        if exploration:
            if type(exploration) != list:
                self.exploration = [exploration]
            else:
                self.exploration = exploration
        else:
            self.exploration = False

        if exploitation:
            self.exploitation = exploitation
            self.exploi_calls = self.exploitation.f_calls
        else:
            self.exploitation = False

        #############
        # VARIABLES #
        #############

        # Save f_calls from metaheuristic, to adapt them during FDA.
        if self.exploitation:
            self.explor_calls = [i.f_calls for i in self.exploration]
        else:
            self.explor_calls = None

        self.tree_search = tree_search

        # Number of explored hypersphere
        self.n_h = 0

        self.executed = False

    # Evaluate a list of hypervolumes
    def evaluate(self, hypervolumes, n_process):

        """evaluate(hypervolumes, n_process)

        Evaluate a list of fractals using exploration and/or exploitation.

        Parameters
        ----------
        hypervolumes : list[Fractal]
            list of hypervolumes to evaluate with exploration and/or exploitation

        """

        # While there are hypervolumes to evaluate do...
        i = 0
        while (
            i < len(hypervolumes)
            and self.search_space.loss.calls < self.f_calls
        ):
            # Select parent hypervolume
            H = hypervolumes[i]
            H.create_children()
            i += 1
            j = 0

            # While there are children do...
            while (
                j < len(H.children)
                and self.search_space.loss.calls < self.f_calls
            ):

                # Select children of parent H
                child = H.children[j]
                j += 1

                # Count the number of explored hypervolume
                self.n_h += 1

                # Exploration
                if child.level != self.tree_search.max_depth:

                    if self.exploration:

                        # Compute the first index of the first solution
                        # which will be computed during exploration
                        start_idx = len(self.search_space.loss.all_solutions)

                        opti_idx = (
                            np.min([child.level, len(self.exploration)]) - 1
                        )
                        calls_left = np.min(
                            [
                                self.explor_calls[opti_idx],
                                self.f_calls - self.search_space.loss.calls,
                            ]
                        )

                        # If there is budget
                        if calls_left > 0:

                            self.exploration[opti_idx].search_space = child
                            self.exploration[opti_idx].f_calls = (
                                calls_left + self.search_space.loss.calls
                            )

                            logger.info(
                                f"""
                                Exploration {child.__class__.__name__}
                                n° {child.id}
                                child of {child.father.id}
                                at level {child.level}\n
                                # of explored fractals : {self.n_h}"""
                            )

                            # Progress bar
                            prec_calls = self.search_space.loss.calls

                            # Run exploration, scores and evaluated solutions are saved using FDA_loss_func class
                            self.exploration[opti_idx].run(n_process=n_process)
                            # Compute the last index of the last solution
                            # computed during exploration
                            last_idx = len(self.search_space.loss.all_solutions)

                            # Save best found solution
                            if self.search_space.loss.new_best:
                                logger.info(
                                    f"""
                                    Best solution found :
                                    {self.search_space.loss.best_score}"""
                                )

                            child.compute_score(slice(start_idx, last_idx))

                            logger.debug(
                                f"""
                                Child {child.father.id}.{child.id}.{child.level}
                                score: {child.score}
                                """
                            )

                            # Progress bar
                            self.pending_pb(
                                self.search_space.loss.calls - prec_calls
                            )
                            self.update_main_pb(
                                self.search_space.loss.calls - prec_calls,
                                explor=True,
                                best=self.search_space.loss.new_best,
                            )
                            self.meta_pb.update(
                                self.search_space.loss.calls - prec_calls
                            )

                    # Add child to tree search
                    self.tree_search.add(child)

                # Exploitation
                elif self.exploitation:

                    # Run exploitation, scores and evaluated solutions are saved using FDA_loss_func class
                    self.exploitation.loss_func = modified_loss_func
                    calls_left = np.min(
                        [
                            self.exploi_calls,
                            self.f_calls - self.search_space.loss.calls,
                        ]
                    )

                    if calls_left > 0:

                        self.exploitation.f_calls = (
                            calls_left + self.search_space.loss.calls
                        )

                        logger.info(
                            f"""
                            Exploration {child.__class__.__name__}
                            n° {child.id}
                            child of {child.father.id}
                            at level {child.level}\n
                            # of explored fractals : {self.n_h}"""
                        )

                        # Progress bar
                        prec_calls = self.search_space.loss.calls

                        # Run exploitation
                        self.exploitation.run(H=child, n_process=n_process)

                        if self.search_space.loss.new_best:
                            logger.info(
                                f"""Best solution found :
                                {self.search_space.loss.best_score}"""
                            )

                        logger.debug(
                            f"""Child {child.father.id}.{child.id}.{child.level}
                            score: EXPLOITATION"""
                        )

                        # Progress bar
                        self.pending_pb(
                            self.search_space.loss.calls - prec_calls
                        )
                        self.update_main_pb(
                            self.search_space.loss.calls - prec_calls,
                            explor=False,
                            best=self.search_space.loss.new_best,
                        )
                        self.meta_pb.update(
                            self.search_space.loss.calls - prec_calls
                        )

    def run(self, n_process=1):

        """run(n_process=1)

        Runs FDA.

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """

        self.build_bar(self.f_calls)

        if self.exploration:
            for exp in self.exploration:
                exp.manager = self.manager

        logger.info("Starting")

        if self.exploitation:
            self.exploitation.manager = self.manager

        self.n_h = 0

        stop = True

        # Select initial hypervolume (root) from the search tree
        stop, hypervolumes = self.tree_search.get_next()

        while stop and self.search_space.loss.calls < self.f_calls:

            self.evaluate(hypervolumes, n_process)
            stop, hypervolumes = self.tree_search.get_next()

        self.executed = True

        logger.info(f"Loss function calls: {self.search_space.loss.calls}")
        logger.info(
            f"Explored {self.search_space.__class__.__name__}: {self.n_h}"
        )
        logger.info(f"Best score: {self.search_space.loss.best_score}")
        logger.info(f"Best solution: {self.search_space.loss.best_sol}")

        self.close_bar()

        logger.info("Ending")

        return self.search_space.loss.get_best(n_process)

    def show(self, filepath="", save=False):

        """show(filename=None)

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filename : str, default=None
            If a filepath is given, the method will read the file and will try to plot contents.

        """

        super().show(filepath, save)
