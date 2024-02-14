# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.search_space import ContinuousSearchspace
from zellij.core.metaheuristic import ContinuousMetaheuristic

from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import ContinuousSearchspace
    from zellij.strategies.tools.chaos_map import ChaosMap

import numpy as np

import logging

logger = logging.getLogger("zellij.CO")


class CGS(ContinuousMetaheuristic):

    """
    CGS is an exploration :ref:`meta` using chaos to violently move in the :ref:`sp`.
    It uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    search_space : ContinuousSearchspace
        Search space object containing bounds of the search space
    level : int
        Chaotic level corresponds to the number of vectors of the chaotic map
    map : ChaosMap
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
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.continuous import CGS
    >>> from zellij.strategies.tools import Henon


    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}


    >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
    >>> sp = ContinuousSearchspace(a)
    >>> cmap = Henon(100, sp.size)
    >>> opt = CGS(sp, cmap)
    >>> stop = Calls(himmelblau, 100)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.9146957930510133, -3.8090645073989187])=13.184852092364691
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 100

    """

    def __init__(
        self,
        search_space: ContinuousSearchspace,
        map: ChaosMap,
        verbose: bool = True,
    ):
        """
        Initialize CGS class

        Parameters
        ----------
        search_space : ContinuousSearchspace
            Search space object containing bounds of the search space
        map : Chaos_map
            Chaotic map used to sample points. See :ref:`cmap` object.
        verbose : boolean, default=True
            Algorithm verbosity.

        """

        super().__init__(search_space, verbose)
        ##############
        # PARAMETERS #
        ##############
        self.map = map
        self.level = self.map.nvectors

        #############
        # VARIABLES #
        #############
        self.iteration = 0

    @property
    def search_space(self) -> ContinuousSearchspace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: ContinuousSearchspace):
        self._search_space = value
        self.up_plus_lo = value.upper + value.lower
        self.up_m_lo = value.upper - value.lower
        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.search_space.lower

    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of CGS.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

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
            xx = [self.search_space.lower + r_mul_y, self.search_space.upper - r_mul_y]

            # for each transformation of the chaotic variable
            sym = np.array([xx[0], xx[1], xx[0], xx[1]])
            sym[2, d] = xx[1][d]
            sym[3, d] = xx[0][d]

            points = np.append(points, sym, axis=0)

        logger.info("CGS forward done")

        return points.tolist(), {"algorithm": "CGS", "seed": self.map.seed}


class CLS(ContinuousMetaheuristic):

    """
    CLS is an exploitation :ref:`meta` using chaos to wiggle points arround an initial solution.
    It uses a rotating polygon to distribute those points, a progressive and mooving zoom on the best solution found, to refine it.
    It uses a :ref:`cmap`, such as Henon or Kent map.
    CLS is a local search and needs a starting point.
    X and Y must not be None at the first forward.

    Attributes
    ----------
    search_space : ContinuousSearchspace
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
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.continuous import CGS
    >>> from zellij.strategies.tools import Henon


    >>> @Loss(objective=Minimizer("obj"), constraint=["c0", "c1"])
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     # coordinates must be <0
    ...     return {"obj": res, "c0": x[0], "c1": x[1]}


    >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
    >>> sp = ContinuousSearchspace(a)
    >>> cmap = Henon(100, sp.size)
    >>> opt = CGS(sp, cmap)
    >>> stop = Calls(himmelblau, 100)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.9146957930510133, -3.8090645073989187])=13.184852092364691
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 100

    """

    def __init__(
        self,
        search_space: ContinuousSearchspace,
        polygon: int,
        map: ChaosMap,
        verbose: bool = True,
    ):
        """
        Initialize CLS class

        Parameters
        ----------
        search_space : ContinuousSearchspace
            Search space object containing bounds of the search space
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
        self.level = self.map.nvectors

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    @property
    def search_space(self) -> ContinuousSearchspace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: ContinuousSearchspace):
        self._search_space = value
        self.up_plus_lo = value.upper + value.lower
        self.up_m_lo = value.upper - value.lower
        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - value.lower

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """
        Runs one step of CLS.
        CLS is a local search and needs a starting point.
        X and Y must not be None.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        Examples
        --------
        >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
        >>> from zellij.core import Experiment, Loss, Minimizer, Calls
        >>> from zellij.strategies.continuous import CLS
        >>> from zellij.strategies.tools import Henon


        >>> @Loss(objective=Minimizer("obj"))
        >>> def himmelblau(x):
        ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
        ...     return {"obj": res}


        >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
        >>> sp = ContinuousSearchspace(a)
        >>> first_point = sp.random_point(1)
        >>> cmap = Henon(100, sp.size)
        >>> opt = CLS(sp, 5, cmap)
        >>> stop = Calls(himmelblau, 1000)
        >>> exp = Experiment(opt, himmelblau, stop)
        >>> exp.run(X=first_point)
        >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
        f([-3.958194530360954, -3.4004586479597])=1.9708306283302715
        >>> print(f"Calls: {himmelblau.calls}")
        Calls: 1000
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

        return points.tolist(), {"algorithm": "CLS", "seed": self.map.seed}


class CFS(ContinuousMetaheuristic):

    """
    CFS is an exploitation :ref:`meta` using chaos to wiggle points arround an initial solution.\
    Contrary to CLS, CFS uses an exponential zoom on the best solution found, it works at a much smaller scale than the CLS.
    It uses a :ref:`cmap`, such as Henon or Kent map.
    CFS is a local search and needs a starting point.
    X and Y must not be None at the first forward.

    Attributes
    ----------
    search_space : ContinuousSearchspace
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
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.continuous import CFS
    >>> from zellij.strategies.tools import Henon


    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}


    >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
    >>> sp = ContinuousSearchspace(a)
    >>> first_point = sp.random_point(1)
    >>> cmap = Henon(100, sp.size)
    >>> opt = CFS(sp, 5, cmap)
    >>> stop = Calls(himmelblau, 1000)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run(X=first_point)
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([3.6504859250093644, -0.026884720842520162])=16.500552515448756
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 1000
        
    """

    def __init__(
        self,
        search_space: ContinuousSearchspace,
        polygon: int,
        map: ChaosMap,
        verbose: bool = True,
    ):
        """__init__

        Initialize CLS class

        Parameters
        ----------
        search_space : ContinuousSearchspace
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
        self.level = self.map.nvectors

        #############
        # VARIABLES #
        #############

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    @property
    def search_space(self) -> ContinuousSearchspace:
        return self._search_space

    @search_space.setter
    def search_space(self, value: ContinuousSearchspace):
        self._search_space = value
        self.up_plus_lo = value.upper + value.lower
        self.up_m_lo = value.upper - value.lower
        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - value.lower

    def _stochastic_round(self, solution: list, k: int):
        s = np.array(solution)
        r = np.random.uniform(-1, 1, len(s))
        # perturbation on CFS zoom
        z = np.round(s.astype(float)) + (k % 2) * r

        return z

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """forward

        Runs one step of CFS.
        CFS is a local search and needs a starting point.
        X and Y must not be None at the first forward.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

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

        return points.tolist(), {"algorithm": "CFS", "seed": self.map.seed}


class ChaoticOptimization(ContinuousMetaheuristic):

    """ChaoticOptimization

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
    >>> from zellij.core import ContinuousSearchspace, ArrayVar, FloatVar
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.continuous import ChaoticOptimization, CGS, CLS, CFS
    >>> from zellij.strategies.tools import Henon


    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}


    >>> a = ArrayVar(FloatVar("f1", -5, 5), FloatVar("i2", -5, 5))
    >>> sp = ContinuousSearchspace(a)
    >>> first_point = sp.random_point(1)

    >>> cmap1 = Henon(20, sp.size)
    >>> cmap2 = Henon(40, sp.size)
    >>> cmap3 = Henon(60, sp.size)

    >>> ocgs = CGS(sp, cmap1)
    >>> ocls = CLS(sp, 5, cmap2)
    >>> ocfs = CFS(sp, 5, cmap3)

    >>> opt = ChaoticOptimization(sp, ocgs, ocls, ocfs)
    >>> stop = Calls(himmelblau, 5000)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run(X=first_point)
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.7846701960436095, -3.2898514141766624])=0.002626222860246351
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 5000
    
    """

    def __init__(
        self,
        search_space: ContinuousSearchspace,
        cgs: CGS,
        cls: CLS,
        cfs: CFS,
        inner: int = 5,
        verbose: bool = True,
    ):
        """__init__(search_space, cgs, cls, cfs, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        search_space : ContinuousSearchspace
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

    def _do_cgs(self, X: list, Y: np.ndarray) -> Tuple[List[list], dict]:
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
            return np.random.random((1, self.search_space.size)).tolist(), {
                "algorithm": "random"
            }

    def _do_cls(self, X: list, Y: np.ndarray) -> Tuple[List[list], dict]:
        self.cls.map.sample(np.random.randint(0, 1000000))

        self.CLS_switch = False
        self.CFS_switch = True

        return self.cls.forward(X, Y)

    def _do_cfs(self, X: list, Y: np.ndarray) -> Tuple[List[list], dict]:
        self.cfs.map.sample(np.random.randint(0, 1000000))

        self.CLS_switch = True
        self.CFS_switch = False

        return self.cfs.forward(X, Y)

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray],
        constraint: Optional[np.ndarray] = None,
    ):
        """forward

        Runs one step of CO.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

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
