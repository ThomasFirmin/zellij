# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:06+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from zellij.core.search_space import ContinuousSearchspace
from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.tools.chaos_map import Chaos_map, Henon
import zellij.utils.progress_bar as pb

import numpy as np

import logging

logger = logging.getLogger("zellij.CO")


class CGS(Metaheuristic):

    """Chaotic Global search

    CGS is an exploration :ref:`meta` using chaos to violently move in the :ref:`sp`.
    It is continuous optimization, so the :ref:`sp` is converted to continuous.
    To do so, it uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    level : int
        Chaotic level corresponds to the number of iteration of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
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
    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core import FloatVar, ArrayVar
    >>> from zellij.strategies import CGS
    >>> from zellij.strategies.tools import Henon
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    ...
    ...          # Henon(map size, dimensions)
    >>> chaosmap = Henon(250,sp.size)
    ...                  # 4 points/iterations: 4x250=1000
    >>> cgs = CGS(sp, 1000, 250, chaosmap)
    >>> cgs.run()


    """

    def __init__(
        self,
        search_space,
        f_calls,
        level,
        map,
        verbose=True,
    ):
        """__init__(search_space, f_calls, level, map, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of :ref:`lf` calls
        level : int
            Chaotic level corresponds to the number of iteration of the chaotic map
        map : Chaos_map
            Chaotic map used to sample points. See :ref:`cmap` object.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, f_calls, verbose)

        assert hasattr(search_space, "to_continuous") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `to_continuous`"""
        )

        self.map = map.map
        self.level = level

        #############
        # VARIABLES #
        #############

        if isinstance(self.search_space, ContinuousSearchspace):
            self.bounds = np.array(
                [
                    [v.low_bound for v in self.search_space.values],
                    [v.up_bound for v in self.search_space.values],
                ],
                dtype=float,
            )
        else:
            self.bounds = np.array(
                [
                    [0.0] * self.search_space.size,
                    [1.0] * self.search_space.size,
                ],
                dtype=float,
            )

        # Working attributes, saved to avoid useless computations.
        self.up_plus_lo = self.bounds[1] + self.bounds[0]
        self.up_m_lo = self.bounds[1] - self.bounds[0]
        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.bounds[0]

    def run(self, shift=1, H=None, n_process=1):

        """run(shift=1, H=None, n_process=1)

        Parameters
        ----------
        shift : int, default=1
            Determines the starting point of the chaotic map.

        H : Fractal, default=None
            When used by :ref:`dba`, a fractal corresponding to the current subspace is given

        n_process : int, default=1
            Determines the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        logger.info("CGS starting")

        self.build_bar(self.level)

        self.k = shift

        # For each level of chaos
        shift_map = (self.k - 1) * self.level
        points = np.empty((0, self.search_space.size), dtype=float)

        n_points = self.search_space.loss.calls
        l = 0

        logger.info("CGS computing chaotic points")

        while l < self.level and n_points < self.f_calls:

            # Randomly select a parameter index of a solution
            d = np.random.randint(self.search_space.size)

            # Apply 3 transformations on the selected chaotic variables
            r_mul_y = np.multiply(self.up_m_lo, self.map[l + shift_map])

            # xx = [np.add(self.center,r_mul_y), np.add(self.center,np.multiply(self.radius,np.multiply(2,y)-1)), np.subtract(self.bounds[1],r_mul_y)]

            # for each transformation of the chaotic variable
            # for x in xx:
            #
            #     x_ = np.subtract(self.up_plus_lo,x)
            #     sym = np.matrix([x,x,x_,x_])
            #     sym[1,d] = x_[d]
            #     sym[3,d] = x[d]
            #     points = np.append(points,sym,axis=0)
            #     n_points += 4

            xx = [self.bounds[0] + r_mul_y, self.bounds[1] - r_mul_y]

            # for each transformation of the chaotic variable
            sym = np.array([xx[0], xx[1], xx[0], xx[1]])
            sym[2, d] = xx[1][d]
            sym[3, d] = xx[0][d]

            points = np.append(points, sym, axis=0)
            n_points += 4

            l += 1
            self.meta_pb.update()

        # Update progress bar
        self.pending_pb(len(points))

        logger.info("CGS evaluating chaotic points")
        if isinstance(self.search_space, ContinuousSearchspace):
            ys = self.search_space.loss(points, algorithm="CGS")
        else:
            ys = self.search_space.loss(
                self.search_space.to_continuous.reverse(points), algorithm="CGS"
            )

        # Update progress bar
        self.update_main_pb(
            len(points), explor=True, best=self.search_space.loss.new_best
        )

        self.close_bar()

        logger.info("CGS ending")

        return self.search_space.loss.get_best(n_process)


class CLS(Metaheuristic):

    """Chaotic Local Search

    CLS is an exploitation :ref:`meta` using chaos to wiggle points arround an initial solution.\
     It uses a rotating polygon to distribute those points, a progressive and mooving zoom on the best solution found, to refine it.
    It is continuous optimization, so the :ref:`sp` is converted to continuous.
    To do so, it uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    level : int
        Chaotic level: the number of iteration of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points)
    red_rate : float
        Reduction rate of the progressive zoom on the best solution found
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
    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core import FloatVar, ArrayVar
    >>> from zellij.strategies import CLS
    >>> from zellij.strategies.tools import Henon
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    ...
    ...          # Henon(map size, dimensions)
    >>> chaosmap = Henon(50,sp.size)
    ...                  # 2xpolygon points/iterations: 2x10x50=1000
    >>> cls = CLS(sp, 1000, 50, 10,chaosmap)
    >>> point = sp.random_point()
    >>> cls.run(point, lf([point])[0])

    """

    def __init__(
        self,
        search_space,
        f_calls,
        level,
        polygon,
        map,
        verbose=True,
    ):

        """__init__(self,search_space,f_calls,level,polygon,map,verbose=True)

        Initialize CLS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of :ref:`lf` calls
        level : int
            Chaotic level corresponds to the number of iteration of the chaotic map
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
        super().__init__(search_space, f_calls, verbose)

        assert hasattr(search_space, "to_continuous") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `to_continuous`"""
        )

        self.level = level
        self.polygon = polygon
        self.map = map.map
        self.red_rate = np.random.random()

        #############
        # VARIABLES #
        #############

        if isinstance(self.search_space, ContinuousSearchspace):
            self.bounds = np.array(
                [
                    [v.low_bound for v in self.search_space.values],
                    [v.up_bound for v in self.search_space.values],
                ],
                dtype=float,
            )
        else:
            self.bounds = np.array(
                [
                    [0.0] * self.search_space.size,
                    [1.0] * self.search_space.size,
                ],
                dtype=float,
            )

        self.up_plus_lo = self.bounds[1] + self.bounds[0]
        self.up_m_lo = self.bounds[1] - self.bounds[0]

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.bounds[0]

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    def run(
        self, X0=None, Y0=None, chaos_level=0, shift=1, H=None, n_process=1
    ):

        """run(X0=None, Y0=None, chaos_level=0, shift=1, H=None, n_process=1)

        Parameters
        ----------
        X0 : list[float], optional
            Initial solution. If None, a Fractal must be given (H!=None)
        Y0 : {int, float}, optional
            Score of the initial solution
        chaos_level : int, default=0
            Determines at which level of the chaos map, the algorithm starts
        shift : int, default=1
            Determines the starting point of the chaotic map.
        H : Fractal, optional
            When used by :ref:`dba`, a fractal corresponding to the current subspace is given
        n_process : int, default=1
            Determines the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        logger.info("CLS starting")

        self.build_bar(self.level)

        if X0:
            if isinstance(self.search_space, ContinuousSearchspace):
                self.X0 = np.array(X0)
            else:
                self.X0 = np.array(
                    self.search_space.to_continuous.convert([X0])[0]
                )
        elif H:
            self.X0 = H.center
        else:
            raise ValueError("No starting point given to CLS")

        if Y0:
            self.Y0 = Y0
        else:
            logger.info("CLS evaluating initial solution")
            if isinstance(self.search_space, ContinuousSearchspace):
                self.Y0 = self.search_space.loss([self.X0], algorithm="CLS")[0]
            else:
                self.Y0 = self.search_space.loss(
                    self.search_space.to_continuous.reverse([self.X0]),
                    algorithm="CLS",
                )[0]

        self.k = shift
        self.chaos_level = chaos_level

        # Initialization
        shift = self.chaos_level * (self.k - 1) * self.level
        # Limits of the search space, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.bounds[1] - self.X0, self.X0 - self.bounds[0])

        center_m_solution = self.center - self.X0
        points = np.empty((0, self.search_space.size), dtype=float)

        n_points = self.search_space.loss.calls
        l = 0

        logger.info("CLS computing chaotic points")
        # for each level of chaos
        while l < self.level and n_points < self.f_calls:

            self.red_rate = np.random.random()

            # Local search area radius
            Rl = self.radius * self.red_rate

            # Decomposition vector
            d = np.random.randint(self.search_space.size)

            # zoom speed
            gamma = 10 ** (-2 * self.red_rate * l) / (l + 1)

            # for each parameter of a solution, determine the improved radius
            xx = np.minimum(gamma * Rl, db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [
                np.multiply(self.map[shift + l], xx),
                np.multiply(1 - self.map[shift + l], xx),
            ]

            # For both chaotic variable
            for x in xv:
                xi = np.outer(self.H[1], x)
                xi[:, d] = x[d] * self.H[0]
                xt = self.X0 + xi

                points = np.append(points, xt, axis=0)
                n_points += self.polygon

            l += 1
            self.meta_pb.update()

        # Update progress bar
        self.pending_pb(len(points))

        logger.info("CLS evaluating chaotic points")
        if isinstance(self.search_space, ContinuousSearchspace):
            ys = self.search_space.loss(points, algorithm="CLS")
        else:
            ys = self.search_space.loss(
                self.search_space.to_continuous.reverse(points), algorithm="CLS"
            )

        # Update progress bar
        self.update_main_pb(
            len(points), explor=True, best=self.search_space.loss.new_best
        )

        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = ys[idx]

        self.close_bar()

        logger.info("CLS ending")

        return self.search_space.loss.get_best(n_process)


class CFS(Metaheuristic):

    """Chaotic Fine Search

    CFS is an exploitation :ref:`meta` using chaos to wiggle points arround an initial solution.\
     Contrary to CLS, CFS uses an exponential zoom on the best solution found, it works at a much smaller scale than the CLS.
    It is continuous optimization, so the :ref:`sp` is converted to continuous.
    To do so, it uses a :ref:`cmap`, such as Henon or Kent map.

    Attributes
    ----------
    level : int
        Chaotic level corresponds to the number of iteration of the chaotic map
    map : Chaos_map
        Chaotic map used to sample points. See Chaos_map object.
    polygon : int
        Vertex number of the rotating polygon (has an influence on the number of evaluated points)
    red_rate : float
        Reduction rate of the progressive zoom on the best solution found
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
    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core.variables import FloatVar, ArrayVar
    >>> from zellij.strategies import CFS
    >>> from zellij.strategies.tools import Henon
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    ...
    ...          # Henon(map size, dimensions)
    >>> chaosmap = Henon(50,sp.size)
    ...                  # 2xpolygon points/iterations: 2x10x50=1000
    >>> cfs = CFS(sp, 1000, 50, 10, chaosmap)
    >>> point = sp.random_point()
    >>> cfs.run(point, lf([point])[0])

    """

    def __init__(
        self,
        search_space,
        f_calls,
        level,
        polygon,
        map,
        verbose=True,
    ):

        """__init__(self,search_space,f_calls,level,polygon,map,verbose=True,converter=None)

        Initialize CLS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of :ref:`lf` calls
        level : int
            Chaotic level corresponds to the number of iteration of the chaotic map
        polygon : int
            Vertex number of the rotating polygon (has an influence on the number of evaluated points)
        map : Chaos_map
            Chaotic map used to sample points. See Chaos_Simulated Annealingmap object.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, f_calls, verbose)

        assert hasattr(search_space, "to_continuous") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `to_continuous`"""
        )

        self.level = level
        self.polygon = polygon
        self.map = map.map
        self.red_rate = np.random.random()

        #############
        # VARIABLES #
        #############

        if isinstance(self.search_space, ContinuousSearchspace):
            self.bounds = np.array(
                [
                    [v.low_bound for v in self.search_space.values],
                    [v.up_bound for v in self.search_space.values],
                ],
                dtype=float,
            )
        else:
            self.bounds = np.array(
                [
                    [0.0] * self.search_space.size,
                    [1.0] * self.search_space.size,
                ],
                dtype=float,
            )

        self.up_plus_lo = self.bounds[1] + self.bounds[0]
        self.up_m_lo = self.bounds[1] - self.bounds[0]

        self.center = np.multiply(0.5, self.up_plus_lo)
        self.radius = np.multiply(0.5, self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.bounds[0]

        trigo_val = 2 * np.pi / self.polygon
        self.H = [np.zeros(self.polygon), np.zeros(self.polygon)]

        for i in range(1, self.polygon + 1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i - 1] = np.cos(trigo_val * i)
            self.H[1][i - 1] = np.sin(trigo_val * i)

    def stochastic_round(self, solution, k):
        s = np.array(solution)
        r = np.random.uniform(-1, 1, len(s))
        # perturbation on CFS zoom
        z = np.round(s.astype(float)) + (k % 2) * r

        return z

    def run(
        self, X0=None, Y0=None, chaos_level=0, shift=1, H=None, n_process=1
    ):

        """run(X0=None, Y0=None, chaos_level=0, shift=1, H=None, n_process=1)

        Parameters
        ----------
        X0 : list[float], optional
            Initial solution. If None, a Fractal must be given (H!=None)
        Y0 : {int, float}, optional
            Score of the initial solution
        chaos_level : int, default=0
            Determines at which level of the chaos map, the algorithm starts
        shift : int, default=1
            Determines the starting point of the chaotic map.
        H : Fractal, optional
            When used by :ref:`dba`, a fractal corresponding to the current subspace is given
        n_process : int, default=1
            Determines the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        logger.info("CFS starting")

        self.build_bar(self.level)

        if X0:
            if isinstance(self.search_space, ContinuousSearchspace):
                self.X0 = np.array(X0)
            else:
                self.X0 = np.array(
                    self.search_space.to_continuous.convert([X0])[0]
                )

        elif H:
            self.X0 = H.center
        else:
            raise ValueError("No starting point given to CFS")

        if Y0:
            self.Y0 = Y0
        else:
            logger.info("CLS evaluating initial solution")
            if isinstance(self.search_space, ContinuousSearchspace):
                self.Y0 = self.search_space.loss([self.X0], algorithm="CFS")[0]
            else:
                self.Y0 = self.search_space.loss(
                    self.search_space.to_continuous.reverse([self.X0]),
                    algorithm="CFS",
                )[0]

        self.k = shift
        self.chaos_level = chaos_level

        shift = self.chaos_level * (self.k - 1) * self.level

        y = self.map[shift]
        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.bounds[1] - self.X0, self.X0 - self.bounds[0])

        r_g = np.zeros(self.search_space.size)

        # Randomly select the reduction rate
        # red_rate = random.random()*0.5

        xc = self.X0
        zc = self.Y0

        center_m_solution = self.center - self.X0
        points = np.empty((0, self.search_space.size), dtype=float)

        n_points = self.search_space.loss.calls
        l = 0

        logger.info("CFS computing chaotic points")

        # for each level of chaos
        while l < self.level and n_points < self.f_calls:

            # Local search area radius
            self.red_rate = np.random.random()
            Rl = self.radius * self.red_rate

            # Decomposition vector
            d = np.random.randint(self.search_space.size)

            # Exponential Zoom factor on the search window
            pc = 10 ** (l + 1)

            # Compute the error/the perturbation applied to the solution
            error_g = np.absolute(
                self.X0 - (self.stochastic_round(pc * self.X0, shift + l) / pc)
            )

            r = np.random.random()

            # for each parameter of a solution determines the improved radius
            r_g = np.minimum((Rl * error_g) / (l**2 + 1), db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(r_g, y), np.multiply(r_g, y)]

            # For both chaotic variable
            for x in xv:
                xi = np.outer(self.H[1], x)
                xi[:, d] = x[d] * self.H[0]
                xt = self.X0 + xi

                points = np.append(points, xt, axis=0)
                n_points += self.polygon

            l += 1
            self.meta_pb.update()

        # Update progress bar
        self.pending_pb(len(points))

        logger.info("CFS evaluating chaotic points")
        if isinstance(self.search_space, ContinuousSearchspace):
            ys = self.search_space.loss(points, algorithm="CFS")
        else:
            ys = self.search_space.loss(
                self.search_space.to_continuous.reverse(points), algorithm="CFS"
            )

        # Update progress bar
        self.update_main_pb(
            len(points), explor=True, best=self.search_space.loss.new_best
        )

        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        # best solution found
        best_sol = points[idx]
        best_scores = ys[idx]

        self.close_bar()

        logger.info("CFS ending")

        return self.search_space.loss.get_best(n_process)


class Chaotic_optimization(Metaheuristic):

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

    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core import FloatVar, ArrayVar
    >>> from zellij.strategies import Chaotic_optimization
    >>> from zellij.utils.benchmark import himmelblau
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> co = Chaotic_optimization(sp, 1000)
    >>> co.run()

    """

    def __init__(
        self,
        search_space,
        f_calls,
        chaos_map=Henon,
        exploration_ratio=0.30,
        levels=(32, 6, 2),
        polygon=4,
        red_rate=0.5,
        verbose=True,
    ):

        """__init__(search_space, f_calls,chaos_map="henon", exploration_ratio = 0.70, levels = (32,6,2), polygon=4, red_rate=0.5, verbose=True)

        Initialize CGS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        f_calls : int
            Maximum number of :ref:`lf` calls
        chaos_map : {'henon', 'kent', 'tent', 'logistic', 'random', Chaos_map}
            If a string is given, the algorithm will select the corresponding map. The chaotic map is used to sample points.\
             If it is a map, it will directly use it. Be carefull, the map size must be sufficient according to the parametrization.
        exploration_ratio : float, default=0.80
            Must be between 0 and 1.\
            It will determine the number of calls to the loss function dedicated to exploration and exploitation, according to chaotic levels associated to CGS, CLS and CFS.
        levels : (int, int, int)
            Used to determine the number of chaotic levels for respectively, CGS, CLS and CFS.
        polygon : int, default=4
            Vertex number of the rotating polygon (has an influence on the number of evaluated points) for CLS and CFS
        red_rate : float, default=0.5
            Reduction rate of the progressive zoom on the best solution found
        verbose : boolean, default=True
            Algorithm verbosity
        """

        ##############
        # PARAMETERS #
        ##############
        super().__init__(search_space, f_calls, verbose)

        assert hasattr(search_space, "to_continuous") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `to_continuous`"""
        )

        self.chaos_map = chaos_map
        self.exploration_ratio = exploration_ratio
        self.polygon = polygon
        self.red_rate = red_rate

        self.CGS_level = levels[0]
        self.CLS_level = levels[1]
        self.CFS_level = levels[2]

        #############
        # VARIABLES #
        #############

        if self.CGS_level > 0:
            if self.CLS_level != 0 or self.CFS_level != 0:
                self.iterations = np.ceil(
                    (self.f_calls * self.exploration_ratio)
                    / (4 * self.CGS_level)
                )
                self.inner_iterations = np.ceil(
                    (self.f_calls * (1 - self.exploration_ratio))
                    / (
                        (self.CLS_level + self.CFS_level)
                        * self.polygon
                        * self.iterations
                    )
                )
            else:
                self.iterations = np.ceil(self.f_calls / (4 * self.CGS_level))
                self.inner_iterations = 0
        else:
            raise ValueError("CGS level must be > 0")

        if type(chaos_map) == str:
            self.map_size = int(
                np.max(
                    [
                        self.iterations * self.CGS_level,
                        self.iterations
                        * self.inner_iterations
                        * self.CLS_level,
                        self.iterations
                        * self.inner_iterations
                        * self.CFS_level,
                    ]
                )
            )
        else:
            self.map_size = int(
                np.ceil(
                    np.max(
                        [
                            self.iterations * self.CGS_level,
                            self.iterations
                            * self.inner_iterations
                            * self.CLS_level,
                            self.iterations
                            * self.inner_iterations
                            * self.CFS_level,
                        ]
                    )
                    / len(self.chaos_map)
                )
            )

        self.map = self.chaos_map(self.map_size, self.search_space.size)

        logging.info(str(self))

    def run(self, H=None, n_process=1):

        """run(H=None, n_process=1)

        Runs the Chaotic_optimization

        Parameters
        ----------
        H : Fractal, default=None
            When used by :ref:`dba`, a fractal corresponding to the current subspace is given

        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points to the continuous format

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated to best_sol

        """

        logger.info("Chaotic optimization starting")

        # Progress bar
        self.build_bar(self.iterations * self.inner_iterations)

        # Initialize CGS/CLS/CFS
        cgs = CGS(
            self.search_space,
            self.f_calls,
            self.CGS_level,
            self.map,
            verbose=self.verbose,
        )
        cls = CLS(
            self.search_space,
            self.f_calls,
            self.CLS_level,
            self.polygon,
            self.map,
            verbose=self.verbose,
        )
        cfs = CFS(
            self.search_space,
            self.f_calls,
            self.CFS_level,
            self.polygon,
            self.map,
            verbose=self.verbose,
        )

        cgs.manager, cls.manager, cfs.manager = (
            self.manager,
            self.manager,
            self.manager,
        )

        # Initialize historic vector
        best_sol = np.array([])
        best_scores = np.array([])

        k = 1

        # Outer loop (exploration)
        while (
            k <= self.iterations and self.search_space.loss.calls < self.f_calls
        ):

            logger.info("Chaotic optimization: Exploration phase")

            # If there is CGS
            if self.CGS_level > 0:

                prec_calls = self.search_space.loss.calls
                self.pending_pb(self.CGS_level * 4)

                x_inter, loss_value = cgs.run(k)

                self.update_main_pb(
                    self.search_space.loss.calls - prec_calls,
                    explor=True,
                    best=self.search_space.loss.new_best,
                )

                # Store to return best solution found
                best_sol = np.append(best_sol, x_inter)
                best_scores = np.append(best_scores, loss_value)

            # Else select random point for the exploitation
            else:

                logger.warning(
                    "Chaotic optimization: using random instead of CGS"
                )

                x_inter = [np.random.random(self.search_space.size)]

                self.pending_pb(1)

                loss_value = self.search_space.loss(x_inter, algorithm="CO")

                self.update_main_pb(
                    1, explor=True, best=self.search_space.loss.new_best
                )

                # Store to return best solution found
                best_sol = np.append(x_inter)
                best_scores = np.append(loss_value)

            logger.debug(
                f"Iterations | Loss function calls | Best value from CGS"
            )
            logger.debug(
                f"{k} < {self.iterations} | {self.search_space.loss.calls} < {self.f_calls} | {loss_value}"
            )
            logger.debug(
                f"New best solution found {self.search_space.loss.new_best}"
            )

            inner = 0

            # Inner loop (exploitation)
            while (
                inner < self.inner_iterations
                and self.search_space.loss.calls < self.f_calls
            ):

                logger.info("Chaotic optimization: Exploitation phase")

                if self.CLS_level > 0:

                    prec_calls = self.search_space.loss.calls
                    self.pending_pb(self.CLS_level * self.polygon * 2)

                    x_inter, loss_value = cls.run(
                        x_inter[0], loss_value[0], inner, k
                    )

                    self.update_main_pb(
                        self.search_space.loss.calls - prec_calls,
                        explor=False,
                        best=self.search_space.loss.new_best,
                    )

                    # Store to return best solution found
                    best_sol = np.append(best_sol, x_inter)
                    best_scores = np.append(best_scores, loss_value)

                if self.CFS_level > 0:

                    prec_calls = self.search_space.loss.calls
                    self.pending_pb(self.CFS_level * self.polygon * 2)

                    x_inter, loss_value = cfs.run(
                        x_inter[0], loss_value[0], inner, k
                    )

                    self.update_main_pb(
                        self.search_space.loss.calls - prec_calls,
                        explor=False,
                        best=self.search_space.loss.new_best,
                    )

                    # Store to return best solution found
                    best_sol = np.append(best_sol, x_inter)
                    best_scores = np.append(best_scores, loss_value)

                logger.debug(
                    f"Iterations | Loss function calls | Best value from CGS"
                )
                logger.debug(
                    f"{k} < {self.iterations} | {self.search_space.loss.calls} < {self.f_calls} | {loss_value}"
                )
                logger.debug(
                    f"New best solution found {self.search_space.loss.new_best}"
                )

                inner += 1

                self.meta_pb.update()

            ind_min = np.argsort(best_scores)[0:n_process]
            best_scores = np.array(best_scores)[ind_min].tolist()
            best_sol = np.array(best_sol)[ind_min].tolist()

            k += 1

        self.close_bar()

        logger.info("Chaotic optimization ending")

        return self.search_space.loss.get_best(n_process)

    def __str__(self):
        return f"Max Loss function calls:{self.f_calls}\nDimensions:{self.search_space.size}\nExploration/Exploitation:{self.exploration_ratio}|{1-self.exploration_ratio}\nRegular polygon:{self.polygon}\nZoom:{self.red_rate}\nIterations:\n\tGlobal:{self.iterations}\n\tInner:{self.inner_iterations}\nChaos Levels:\n\tCGS:{self.CGS_level}\n\tCLS:{self.CLS_level}\n\tCFS:{self.CFS_level}\nMap size:{self.map_size}x{self.search_space.size}"
