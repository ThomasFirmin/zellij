# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:12+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic

import numpy as np
import copy

import logging

logger = logging.getLogger("zellij.DBA")


class DBA(Metaheuristic):

    """DBA

    Decomposition-Based-Algorithm (DBA) is made of 5 part:

        * **Geometry** : DBA uses hyper-spheres or hyper-cubes to decompose the search-space into smaller sub-spaces in a fractal way.
        * **Tree search**: Fractals are stored in a *k-ary rooted tree*. The tree search determines how to move inside this tree.
        * **Exploration** : To explore a fractal, DBA requires an exploration algorithm.
        * **Exploitation** : At the final fractal level (e.g. a leaf of the rooted tree) DBA performs an exploitation.
        * **Scoring method**: To score a fractal, DBA can use the best score found, the median, ...

    Attributes
    ----------

    search_space : Fractal
        :ref:`sp` defined as a  :ref:`frac`. Contains decision
        variables of the search space, converted to continuous and
        constrained to an Euclidean :ref:`frac`.

    f_calls : int
        Maximum number of :ref:`lf` calls

    exploration : {Metaheuristic, list[Metaheuristic]}, default=None
        Algorithm used to sample inside each subspaces.

    exploitation : Metaheuristic, default=None
        Intensification algorithm applied on a subspace at the last level
        of the partition tree.

    tree_search : Tree_search
        Tree search algorithm applied on the partition tree.

    verbose : boolean, default=True
        Algorithm verbosity


    Methods
    -------

    evaluate(hypervolumes)
        Evaluate a list of fractals using exploration and/or exploitation

    run(n_process=1)
        Runs DBA

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

        """__init__(search_space, f_calls, tree_search, exploration=None, exploitation=None, verbose=True, **kwargs)

        Initialize DBA class

        Parameters
        ----------
        search_space : Fractal
            :ref:`sp` defined as a  :ref:`frac`. Contains decision
            variables of the search space, converted to continuous and
            constrained to an EUclidean :ref:`frac`.

        f_calls : int
            Maximum number of :ref:`lf` calls

        exploration : {Metaheuristic, list[Metaheuristic]}, default=None
            Algorithm used to sample inside each subspaces.

        exploitation : Metaheuristic, default=None
            Intensification algorithm applied on a subspace at the last level
            of the partition tree.

        tree_search : Tree_search
            Tree search algorithm applied on the partition tree.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############

        super(DBA, self).__init__(search_space, f_calls, verbose)

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

        # Save f_calls from metaheuristic, to adapt them during DBA.
        if self.exploration:
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

                            # Run exploration, scores
                            self.exploration[opti_idx].run(
                                H=child, n_process=n_process
                            )
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

                            # score fractal
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

                    # Run exploitation, scores
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

        Runs DBA.

        Parameters
        ----------
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

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
        logger.info(f"Best solution: {self.search_space.loss.best_point}")

        self.close_bar()

        logger.info("Ending")

        return self.search_space.loss.get_best(n_process)
