# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T15:08:28+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from zellij.core.search_space import ContinuousSearchspace
from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.strategies.tools.chaos_map import Chaos_map, Henon

import numpy as np

import logging

logger = logging.getLogger("zellij.CO")


class CGS(ContinuousMetaheuristic):

    """Chaotic Global search

    CGS is an exploration :ref:`meta` using chaos to violently move in the :ref:`sp`.
    It is continuous optimization, so the :ref:`sp` is converted to continuous.
    To do so, it uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    search_space : Searchspace
        Search space object containing bounds of the search space
    level : int
        Chaotic level corresponds to the number of vectors of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See :ref:`cmap` object.
    verbose : boolean, default=True
        Algorithm verbosity.
    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous
    center : float
        List of floats containing the coordinates of the search space center converted to continuous
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous

    See Also

    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    Chaotic_optimization : CGS is used here to perform an exploration
    CLS : Chaotic Local Search
    CFS : Chaotic Fine Search

    Examples
    --------
    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.benchmarks import himmelblau
    >>> from zellij.strategies import CGS
    >>> from zellij.strategies.tools import Henon
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> stop = Threshold(lf, 'calls', 100)
    >>> zcgs = CGS(sp,Henon(5,sp.size))
    >>> exp = Experiment(zcgs, stop)
    >>> exp.run()
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")


    """

    def __init__(
        self,
        search_space,
        map,
        verbose=True,
    ):
        """__init__(search_space, map, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        map : Chaos_map
            Chaotic map used to sample points. See :ref:`cmap` object.
        verbose : boolean, default=True
            Algorithm verbosity.

        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, verbose)

        self.map = map
        self.level = self.map.vectors

        #############
        # VARIABLES #
        #############

        self.iteration = 0

        # Working attributes, saved to avoid useless computations.
        self.up_plus_lo = self.search_space.upper + self.search_space.lower
        self.up_m_lo = self.search_space.upper - self.search_space.lower
        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.search_space.lower

    def forward(self, X, Y):
        """forward(x, Y)
        Runs one step of CGS.

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

        logger.info("CGS starting")

        # For each level of chaos
        points = np.empty((0, self.search_space.size), dtype=float)

        logger.info("CGS computing chaotic points")

        for l in range(self.level):
            # Randomly select a parameter index of a solution
            d = np.random.randint(self.search_space.size)

            # Apply 3 transformations on the selected chaotic variables
            r_mul_y = np.multiply(self.up_m_lo, self.map.map[l])

            # xx = [np.add(self.center,r_mul_y), np.add(self.center,np.multiply(self.radius,np.multiply(2,y)-1)), np.subtract(self.search_space.upper,r_mul_y)]

            # for each transformation of the chaotic variable
            # for x in xx:
            #
            #     x_ = np.subtract(self.up_plus_lo,x)
            #     sym = np.matrix([x,x,x_,x_])
            #     sym[1,d] = x_[d]
            #     sym[3,d] = x[d]
            #     points = np.append(points,sym,axis=0)
            #     n_points += 4

            xx = [self.search_space.lower + r_mul_y, self.search_space.upper - r_mul_y]

            # for each transformation of the chaotic variable
            sym = np.array([xx[0], xx[1], xx[0], xx[1]])
            sym[2, d] = xx[1][d]
            sym[3, d] = xx[0][d]

            points = np.append(points, sym, axis=0)

        logger.info("CGS forward done")

        return points, {"algorithm": "CGS", "seed": self.map.seed}


class CLS(ContinuousMetaheuristic):

    """Chaotic Local Search

    CLS is an exploitation :ref:`meta` using chaos to wiggle points arround an initial solution.\
     It uses a rotating polygon to distribute those points, a progressive and mooving zoom on the best solution found, to refine it.
    It is continuous optimization, so the :ref:`sp` is converted to continuous.
    To do so, it uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    search_space : Searchspace
        Search space object containing bounds of the search space
    level : int
        Chaotic level corresponds to the number of vectors of the chaotic map
    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points)
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    verbose : boolean, default=True
        Algorithm verbosity
    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous
    center : float
        List of floats containing the coordinates of the search space center converted to continuous
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    Chaotic_optimization : CLS is used here to perform an exploitation
    CGS : Chaotic Global Search
    CFS : Chaotic Fine Search

    Examples
    --------
    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.benchmarks import himmelblau
    >>> from zellij.strategies import CLS
    >>> from zellij.strategies.tools import Henon
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> stop = Threshold(lf, 'calls', 100)
    >>> zcls = CLS(sp,8,Henon(5,sp.size))
    >>> x_start = [sp.random_point()]
    >>> _, y_start = lf(x_start)
    >>> exp = Experiment(zcls, stop)
    >>> exp.run(x_start, y_start)
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")

    """

    def __init__(
        self,
        search_space,
        polygon,
        map,
        verbose=True,
    ):
        """__init__(self,search_space,level,polygon,map,verbose=True)

        Initialize CLS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        level : int
            Chaotic level corresponds to the number of vectors of the chaotic map
        polygon : int
            Vertex number of the rotating polygon (has an influence on the number of evaluated points)
        map : Chaos_map
            Chaotic map used to sample points. See Chaos_map object.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, verbose)

        self.polygon = polygon
        self.map = map
        self.level = self.map.vectors

        self.up_plus_lo = self.search_space.upper + self.search_space.lower
        self.up_m_lo = self.search_space.upper - self.search_space.lower

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.search_space.lower

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    def forward(self, X, Y):
        """forward(X, Y)

        Runs one step of CLS.

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

        x_best = np.array(X[np.argmin(Y)])

        logger.info("CLS starting")

        # Initialization
        # Limits of the search space, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(
            self.search_space.upper - x_best, x_best - self.search_space.lower
        )

        logger.info("CLS computing chaotic points")

        points = np.empty((0, self.search_space.size), dtype=float)

        for l in range(self.level):
            red_rate = np.random.random()

            # Local search area radius
            Rl = self.radius * red_rate
            # Decomposition vector
            d = np.random.randint(self.search_space.size)

            # zoom speed
            gamma = 10 ** (-2 * red_rate * l) / (l + 1)

            # for each parameter of a solution, determine the improved radius
            xx = np.minimum(gamma * Rl, db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [
                np.multiply(self.map.map[l], xx),
                np.multiply(1 - self.map.map[l], xx),
            ]

            # For both chaotic variable
            for x in xv:
                xi = np.outer(self.H[1], x)
                xi[:, d] = x[d] * self.H[0]
                xt = x_best + xi

                points = np.append(points, xt, axis=0)

        logger.info("CLS forward ending")

        return points, {"algorithm": "CLS", "seed": self.map.seed}


class CFS(ContinuousMetaheuristic):

    """Chaotic Fine Search

    CFS is an exploitation :ref:`meta` using chaos to wiggle points arround an initial solution.\
     Contrary to CLS, CFS uses an exponential zoom on the best solution found, it works at a much smaller scale than the CLS.
    It is continuous optimization, so the :ref:`sp` is converted to continuous.
    To do so, it uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    search_space : Searchspace
        Search space object containing bounds of the search space
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points)
    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous
    center : float
        List of floats containing the coordinates of the search space center converted to continuous
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    Chaotic_optimization : CLS is used here to perform an exploitation
    CGS : Chaotic Global Search
    CLS : Chaotic Local Search

    Examples
    --------
    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.benchmarks import himmelblau
    >>> from zellij.strategies import CFS
    >>> from zellij.strategies.tools import Henon
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> stop = Threshold(lf, 'calls', 100)
    >>> zcfs = CFS(sp,8,Henon(5,sp.size))
    >>> x_start = [sp.random_point()]
    >>> _, y_start = lf(x_start)
    >>> exp = Experiment(zcfs, stop)
    >>> exp.run(x_start, y_start)
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")

    """

    def __init__(
        self,
        search_space,
        polygon,
        map,
        verbose=True,
    ):
        """__init__(self,search_space,polygon,map,verbose=True)

        Initialize CLS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        polygon : int
            Vertex number of the rotating polygon (has an influence on the number of evaluated points)
        map : Chaos_map
            Chaotic map used to sample points. See :ref:`cmap` object.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, verbose)

        self.polygon = polygon
        self.map = map
        self.level = self.map.vectors

        #############
        # VARIABLES #
        #############

        self.up_plus_lo = self.search_space.upper + self.search_space.lower
        self.up_m_lo = self.search_space.upper - self.search_space.lower

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.search_space.lower

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    def _stochastic_round(self, solution, k):
        s = np.array(solution)
        r = np.random.uniform(-1, 1, len(s))
        # perturbation on CFS zoom
        z = np.round(s.astype(float)) + (k % 2) * r

        return z

    def forward(self, X, Y):
        """forward(X, Y)

        Runs one step of CFS.

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

        logger.info("CLS starting")

        x_best = np.array(X[np.argmin(Y)])

        # Initialization
        # Limits of the search space, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(
            self.search_space.upper - x_best, x_best - self.search_space.lower
        )

        logger.info("CLS computing chaotic points")

        points = np.empty((0, self.search_space.size), dtype=float)

        for l in range(self.level):
            red_rate = np.random.random()

            # Local search area radius
            Rl = self.radius * red_rate
            # Decomposition vector
            d = np.random.randint(self.search_space.size)

            # Exponential Zoom factor on the search window
            pc = 10 ** (l + 1)

            # Compute the error/the perturbation applied to the solution
            error_g = np.absolute(
                x_best - (self._stochastic_round(pc * x_best, l) / pc)
            )

            # for each parameter of a solution determines the improved radius
            r_g = np.minimum((Rl * error_g) / (l**2 + 1), db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [
                np.multiply(r_g, self.map.map[l]),
                np.multiply(r_g, self.map.map[l]),
            ]

            # For both chaotic variable
            for x in xv:
                xi = np.outer(self.H[1], x)
                xi[:, d] = x[d] * self.H[0]
                xt = x_best + xi

                points = np.append(points, xt, axis=0)

        logger.info("CLS forward ending")

        return points, {"algorithm": "CFS", "seed": self.map.seed}


class Chaotic_optimization(ContinuousMetaheuristic):

    """Chaotic_optimization

    Chaotic optimization combines CGS, CLS and CFS.

    Attributes
    ----------

    chaos_map : {'henon', 'kent', 'tent', 'logistic', 'random', Chaos_map}
        If a string is given, the algorithm will select the corresponding map. The chaotic map is used to sample points.\
         If it is a map, it will directly use it. Be carefull, the map size must be sufficient according to the parametrization.

    exploration_ratio : float
        It will determine the number of calls to the loss function dedicated to exploration and exploitation, according to chaotic levels associated to CGS, CLS and CFS.

    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points) for CLS and CFS

    red_rate : float
        Reduction rate of the progressive zoom on the best solution found for CLS and CFS

    CGS_level : int
        Number of chaotic level associated to CGS

    CLS_level : int
        Number of chaotic level associated to CLS

    CFS_level : int
        Number of chaotic level associated to CFS

    verbose : boolean, default=True
        Algorithm verbosity

    Methods
    -------

    run(self, n_process=1)
        Runs Chaotic_optimization

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    CGS : Chaotic Global Search
    CLS : Chaotic Local Search
    CFS : Chaotic Fine Search

    Examples
    --------

    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.benchmarks import himmelblau
    >>> from zellij.strategies import CGS, CLS, CFS, Chaotic_optimization
    >>> from zellij.strategies.tools import Henon
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> stop = Threshold(lf, 'calls', 100)
    >>> zcgs = CGS(sp,Henon(5,sp.size))
    >>> zcls = CLS(sp,8,Henon(2,sp.size))
    >>> zcfs = CFS(sp,8,Henon(2,sp.size))
    >>> co = Chaotic_optimization(sp,zcgs,zcls,zcfs)
    >>> exp = Experiment(co, stop)
    >>> exp.run()
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")


    """

    def __init__(self, search_space, cgs, cls, cfs, inner=5, verbose=True):
        """__init__(search_space, cgs, cls, cfs, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        cgs : CGS
            CGS :ref:`meta`
        cls : CLS
            CLS :ref:`meta`
        cfs : CFS
            CFS :ref:`meta`
        inner : int, default=5
            Number of iterations of CLS and CFS.
        verbose : boolean, default=True
            Algorithm verbosity
        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, verbose)

        assert hasattr(search_space, "convert") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `convert`"""
        )

        # Initialize CGS/CLS/CFS
        self.cgs = cgs
        self.cls = cls
        self.cfs = cfs

        self.inner = inner

        #############
        # VARIABLES #
        #############

        # iteration counter for CGS
        self.outer_it = 0
        self.inner_it = -1

        # Switchers
        self.CGS_switch = True
        self.CLS_switch = False
        self.CFS_switch = False

        logging.info(str(self))

    def reset(self):
        """reset()

        Reset Chaotic_optimization to its initial values.

        """
        # iteration counter for CGS
        self.outer_it = 0
        self.inner_it = 0
        # Switchers
        self.CGS_switch = True
        self.CLS_switch = False
        self.CFS_switch = False
        # reset algos
        self.cgs.reset()
        self.cls.reset()
        self.cfs.reset()

    def _do_cgs(self, X, Y):
        self.cgs.map.sample(np.random.randint(0, 1000000))
        # Outer loop (exploration)
        logger.info("Chaotic optimization: Exploration phase")

        self.CGS_switch = False
        self.CLS_switch = True

        # If there is CGS
        if self.cgs:
            return self.cgs.forward(X, Y)

        # Else select random point for the exploitation
        else:
            logger.warning("Chaotic optimization: using random instead of CGS")

            return [np.random.random(self.search_space.size)], {"algorithm": "random"}

    def _do_cls(self, X, Y):
        self.cls.map.sample(np.random.randint(0, 1000000))

        self.CLS_switch = False
        self.CFS_switch = True

        return self.cls.forward(X, Y)

    def _do_cfs(self, X, Y):
        self.cfs.map.sample(np.random.randint(0, 1000000))

        self.CLS_switch = True
        self.CFS_switch = False

        return self.cfs.forward(X, Y)

    def forward(self, X, Y):
        """forward(H=None, n_process=1)

        Runs one step of BO.

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

        logger.info("Chaotic optimization starting")

        logger.debug(
            f"""
            CGS:{self.CGS_switch}, CLS:{self.CLS_switch}, CFS:{self.CFS_switch},
            outer:{self.outer_it}, inner:{self.inner_it}
            """
        )

        if self.CGS_switch:
            return self._do_cgs(X, Y)

        # Inner loop (exploitation)
        if self.inner_it < self.inner:
            self.inner_it += 1

            logger.info("Chaotic optimization: Exploitation phase")

            if self.cls and self.CLS_switch:
                return self._do_cls(X, Y)
            else:
                self.CLS_switch = False
                self.CFS_switch = True

            if self.cfs and self.CFS_switch:
                return self._do_cfs(X, Y)
            else:
                self.inner_it += 1
                self.CLS_switch = True
                self.CFS_switch = False

                if self.inner_it < self.inner:
                    return self._do_cfs(X, Y)
                else:
                    self.outer_it += 1
                    return self._do_cgs(X, Y)

        else:
            self.CGS_switch = True
            self.CLS_switch = False
            self.CFS_switch = False
            self.outer_it += 1
            self.inner_it = 0

            return self._do_cgs(X, Y)

        logger.info("Chaotic optimization forward ending")
