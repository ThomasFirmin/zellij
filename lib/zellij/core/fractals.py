# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   ThomasFirmin
# @Last modified time: 2022-05-03T15:44:11+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import numpy as np
import abc
import copy

import time

from zellij.strategies.utils.spoke_dart import (
    randomMuller,
    Hyperplane,
    HalfLine,
)

import logging

logger = logging.getLogger("zellij.fractal")
logger.setLevel(logging.INFO)


class Fractal(object):
    """Fractal

    Fractal is an abstract class used in Fractal Decomposition. This class is used to build a partition tree of fractals. Each object contains a reference to its father, references to its children, its bounds,\
    and its heuristic value (the score) computed after each exploration. Fractals are simplified and continuous subspaces, builded thanks to the original SearchSpace object.

    Attributes
    ----------

    lo_bounds : list[float]
        Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

    up_bounds : list[float]
        Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

    id : int
        Identifier of a fractal. Combined to the id of itf parents, the id is unique.

    father : Fractal
        Reference to the parent of the current fractal.

    children : list[Fractal]
        References to all children of the current fractal.

    score : {float, int}
        Heuristic value associated to the fractal after an exploration

    level : int
        Current level of the fractal in the partition tree. See Tree_search.

    min_score : {float, int}
        Score associated to the best found solution inside the fractal

    best_sol : list[{float, int, str}]
        Best found solution inside the fractal in its mixed format.

    all_solutions : float
        Historic of all evaluated solutions inside the fractal.

    all_scores : float
        Historic of all evaluated scores inside the fractal..

    Methods
    -------

    add_point(score, solution)
        Adds a point to the fractal

    create_children()
        Abstract method which defines how fractal children should be created


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Hypercube : Inherited Fractal type
    Hypersphere : Inherited Fractal type
    """

    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
    ):

        """__init__(lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

        self.lo_bounds = np.array(lo_bounds)
        self.up_bounds = np.array(up_bounds)

        self.id = id
        self.father = father
        self.children = []
        self.score = score
        self.level = level

        self.min_score = float("inf")
        self.best_sol = None
        self.solutions = []
        self.all_scores = []

    def add_point(self, score, solution):

        """add_point(self,score, solution)

        This method adds a point associated to its evaluation by the loss function (f(solution)=score), to the historic of the fractal,\
         and determine if this point is the best one among all evaluated points inside the fractal.

        Parameters
        ----------
        score : {int, float}
            Score associated to the evaluated solution

        solution : list[{int, float, str}]
            It corresponds to a point in the mixed format inside the fractal and associated to its evaluation (score).

        """
        for sol, sco in zip(solution, score):
            self.solutions.append(sol)
            self.all_scores.append(sco)

            if sco < self.min_score:
                self.min_score = sco
                self.best_sol = sol

    @abc.abstractmethod
    def create_children(self):
        """create_children(self)

        Abstract method which will create children of the current Fractal object, according to certain rules (Hypercube, Hypersphere...)

        """
        pass


class Hypercube(Fractal):
    """Hypercube

    The hypercube is a basic hypervolume to decompose the SearchSpace. It's also one of the most computationally inefficient in high dimension.\
    The decomposition complexity of an Hypercube with equalsize Hypercubes, is equal to $2^d$, d is the dimension.\
    However building a single hypercube is low complexity task, and the space coverage is very good, 100% of the initial hypercube is covered by its children.

    Attributes
    ----------

    dim : int
        Number of dimensions

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hypercubes based on the current Hypercube.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypersphere : Another hypervolume, with different properties
    """

    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
    ):

        """__init__(lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

        super().__init__(
            lo_bounds, up_bounds, father, level, id, children, score
        )

        self.dim = len(self.up_bounds)

    def create_children(self):

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

        level = self.level + 1

        up_m_lo = self.up_bounds - self.lo_bounds
        radius = np.abs(up_m_lo / 2)
        bounds = [[self.lo_bounds, self.up_bounds]]

        # Hyperplan bisecting
        next_b = []
        for i in range(self.dim):
            next_b = []
            for b in bounds:

                # First part
                up = np.copy(b[1])
                up[i] = b[0][i] + radius[i]
                next_b.append([np.copy(b[0]), np.copy(up)])

                # Second part
                low = np.copy(b[0])
                low[i] = b[1][i] - radius[i]
                next_b.append([np.copy(low), np.copy(b[1])])

            bounds = copy.deepcopy(next_b)

        # Create Hypercube
        n_h = 0
        for b in bounds:
            h = Hypercube(b[0], b[1], self, level, n_h)
            self.children.append(h)
            n_h += 1

    def __repr__(self):
        if type(self.father) == str:
            id = "root"
        else:
            id = str(self.father.id)

        return (
            "ID: "
            + str(self.id)
            + " son of "
            + id
            + "\n"
            + "BOUNDS: "
            + str(self.lo_bounds)
            + "|"
            + str(self.up_bounds)
            + "\n"
        )


class Hypersphere2(Fractal):

    """Hypersphere

    The Hypersphere is a basic hypervolume to decompose the SearchSpace. It is one of the most computationally efficient, to decompose the SearchSpace.
    To decompose an hypersphere by equalsize hypersphere the complexity is equal to $2*d$, d is the dimension, moreover building an hypersphere is easy, it only needs a center and its radius.
    However the space coverage is poor, indeed the volume of an hypersphere tends to 0, when the dimension tends to infinity. To partially tackle this problem, an inflation rate allows bigger hypersphere,
    but this will create overlapping hypervolumes.

    Attributes
    ----------

    dim : int
        Number of dimensions

    inflation : float
        Inflation rate of hyperspheres. Be carefull a too large inflation can result to hypersphere with identical center and radius.

    center : list[float]
        List of floats containing the coordinates

    radius : list[float]
        List of floats containing the radius for each dimension (in case the initial SearchSpace is not an hypercube).

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hyperspheres based on the current Hypersphere.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties
    """

    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        inflation=1.75,
        children=[],
        score=None,
    ):

        """__init__(lo_bounds, up_bounds, father="root", level=0, id=0, inflation=1.75, children=[], score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal.
            If no child IS given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

        super().__init__(
            lo_bounds, up_bounds, father, level, id, children, score
        )

        self.dim = len(self.up_bounds)

        up_m_lo = self.up_bounds - self.lo_bounds
        center = self.lo_bounds + (up_m_lo) / 2
        radius = up_m_lo[0] / 2

        self.center = center
        self.radius = radius

        self.inflation = inflation

        self.radius = self.radius * self.inflation

    def create_children(self):

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

        level = self.level + 1

        r_p = self.radius / (1 + np.sqrt(2))

        n_h = 0
        for i in range(self.dim):

            n_h += 1
            center = np.copy(self.center)
            center[i] += ((-1) ** i) * (self.radius - r_p)

            lo = np.maximum(center - r_p, self.lo_bounds)
            up = np.minimum(center + r_p, self.up_bounds)

            h = Hypersphere(lo, up, self, level, n_h, inflation=self.inflation)
            self.children.append(h)

            n_h += 1
            center = np.copy(self.center)
            center[i] -= ((-1) ** i) * (self.radius - r_p)

            lo = np.maximum(center - r_p, self.lo_bounds)
            up = np.minimum(center + r_p, self.up_bounds)

            h = Hypersphere(lo, up, self, level, n_h, inflation=self.inflation)
            self.children.append(h)

    def __repr__(self):
        if type(self.father) == str:
            id = "root"
        else:
            id = str(self.father.id)

        return (
            "ID: "
            + str(self.id)
            + " son of "
            + id
            + " at level "
            + str(self.level)
            + "\n"
            + "BOUNDS: "
            + str(self.lo_bounds)
            + "|"
            + str(self.up_bounds)
            + "\n"
        )


class Hypersphere(Fractal):

    """Hypersphere

    The Hypersphere is a basic hypervolume to decompose the SearchSpace. It is one of the most computationally efficient.
    To decompose an hypersphere by equalsize hypersphere the complexity is equal to $2*d$, d is the dimension, moreover building an hypersphere is easy, it only needs a center and its radius.
    However the space coverage is poor, indeed the volume of an hypersphere tends to 0, when the dimension tends to infinity. To partially tackle this problem, an inflation rate allows bigger hypersphere,
    but this will create overlapping hypervolumes.

    Attributes
    ----------

    dim : int
        Number of dimensions

    inflation : float
        Inflation rate of hyperspheres. Be carefull a too large inflation can result to hypersphere with identical center and radius.

    center : list[float]
        List of floats containing the coordinates

    radius : list[float]
        List of floats containing the radius for each dimension (in case the initial SearchSpace is not an hypercube).

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hyperspheres based on the current Hypersphere.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties
    """

    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        radius=None,
        center=None,
        inflation=1.75,
        children=[],
        score=None,
    ):

        """__init__(father,lo_bounds,up_bounds,level,id,children=[],score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.
            /!\ Those bounds are clipped in case the hypersphere is outside the decision space

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.
            /!\ Those bounds are clipped in case the hypersphere is outside the decision space

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of its parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal.
            If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.
            If no score is given, it will be built after the execution of an exploration inside the fractal.

        """

        super().__init__(
            lo_bounds, up_bounds, father, level, id, children, score
        )

        if isinstance(father, str):
            self.original_up = self.lo_bounds
            self.original_down = self.up_bounds
        else:
            self.original_up = father.original_up
            self.original_down = father.original_down

        self.dim = len(self.up_bounds)

        if center is None or radius is None:
            up_m_lo = self.up_bounds - self.lo_bounds
            center = self.lo_bounds + (up_m_lo) / 2
            radius = up_m_lo[0] / 2

        self.center = center
        self.inflation = inflation
        self.radius = radius * self.inflation

    def create_children(self):

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

        level = self.level + 1

        r_p = self.radius / (1 + np.sqrt(2))

        n_h = 0
        for i in range(self.dim):

            logger.info(f"Building children n°{n_h}/{self.dim*2}")

            center = np.copy(self.center)
            center[i] += r_p
            center[i] = np.minimum(center[i], 1)

            lo = center - r_p
            lo[lo < 0] = 0
            lo[lo > 1] = 1

            up = center + r_p
            up[up < 0] = 0
            up[up > 1] = 1

            h = Hypersphere(
                lo,
                up,
                self,
                level,
                n_h,
                center=np.copy(center),
                radius=r_p,
                inflation=self.inflation,
            )

            self.children.append(h)
            n_h += 1

            logger.info(f"Building children n°{n_h}/{self.dim*2}")

            center = np.copy(self.center)
            center[i] -= r_p
            center[i] = np.maximum(center[i], 0)

            lo = center - r_p
            lo[lo < 0] = 0
            lo[lo > 1] = 1

            up = center + r_p
            up[up < 0] = 0
            up[up > 1] = 1

            h = Hypersphere(
                lo,
                up,
                self,
                level,
                n_h,
                center=np.copy(center),
                radius=r_p,
                inflation=self.inflation,
            )

            self.children.append(h)
            n_h += 1

    def __repr__(self):
        if type(self.father) == str:
            id = "root"
        else:
            id = str(self.father.id)

        return (
            "ID: "
            + str(self.id)
            + " son of "
            + id
            + " at level "
            + str(self.level)
            + "\n"
            + "BOUNDS: "
            + str(self.lo_bounds)
            + "|"
            + str(self.up_bounds)
            + "\n"
        )


class Section(Fractal):

    """Section

    Performs a n-Section of the search space. 3-Section: DIRECT, 2-Section: BIRECT

    Attributes
    ----------

    dim : int
        Number of dimensions

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hyper-rectangles based on the current Hyper-rectangle.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties
    """

    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
        n=2,
    ):

        """__init__(lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None, n=2)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

        super().__init__(
            lo_bounds, up_bounds, father, level, id, children, score
        )

        assert n > 1, logger.error(
            f"{n}-Section is not possible, n must be > 1"
        )

        self.dim = len(self.up_bounds)
        self.section = n

        up_m_lo = self.up_bounds - self.lo_bounds
        self.center = self.lo_bounds + (up_m_lo) / 2
        self.longest = np.argmax(up_m_lo)
        self.length = up_m_lo[self.longest]

    def create_children(self):

        level = self.level + 1

        n_h = 0

        new_val = self.length / self.section

        lo = np.copy(self.lo_bounds)
        up = np.copy(self.up_bounds)
        up[self.longest] = lo[self.longest] + new_val

        for i in range(self.section):

            h = Section(lo, up, self, level, n_h, n=self.section)

            self.children.append(h)

            lo = np.copy(h.lo_bounds)
            up = np.copy(h.up_bounds)
            lo[self.longest] += new_val
            up[self.longest] += new_val

            n_h += 1

    def __repr__(self):
        if type(self.father) == str:
            id = "root"
        else:
            id = str(self.father.id)

        return (
            "ID: "
            + str(self.id)
            + " son of "
            + id
            + " at level "
            + str(self.level)
            + "\n"
            + "BOUNDS: "
            + str(self.lo_bounds)
            + "|"
            + str(self.up_bounds)
            + "\n"
        )


class Voronoi(Fractal):
    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
        seed="random",
        n_seeds=None,
    ):

        """lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None, seed="random", spokes=2, n_seeds=5

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        seed : {random, [[float]]}
            If 'random' Voronoi centroid will be initialized randomly. Else: if a list of points is given, the DynamicVoronoi will use them.

        spokes : int, default=2
            Number of randoms spokes to draw during hyperplane sampling (SpokeDart)

        n_seeds : int, default=2
            Number of centroids at each decomposition.

        """

        super().__init__(
            lo_bounds, up_bounds, father, level, id, children, score
        )

        self.dim = len(self.up_bounds)
        if n_seeds is None:
            self.n_seeds = 2 * self.dim
        else:
            self.n_seeds = n_seeds

        self.next_seeds = []

        if isinstance(seed, str) and seed == "random":
            self.all_seeds = []
            self.next_seeds = list(
                np.random.random((self.n_seeds, self.dim))
                * (np.array(self.up_bounds) - np.array(self.lo_bounds))
                + np.array(self.lo_bounds)
            )
            self.seed = "root"
            self.hyperplanes = []

        else:
            self.all_seeds = self.father.all_seeds
            self.hyperplanes = []
            self.seed = seed
            self.center = self.seed

        self.xlist = np.zeros(2 * self.dim)

    @abc.abstractmethod
    def create_children(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    def randomSpoke(self, s):
        p = randomMuller(1, self.dim)[0]
        pfar = p + s

        return HalfLine(s, pfar)

    def dimSpoke(self, s, dim, r=1):
        p = np.zeros(self.dim)
        p[dim] = r
        pfar = p + s
        return HalfLine(s, pfar)

    def shiftBorder(self, s, p):
        l = HalfLine(s, p)
        return l.point(np.random.random())

    def clipBorder(self, line, upma, loma):

        self.xlist[: self.dim] = loma / (line.v * (1 + 1e-10))
        self.xlist[self.dim :] = upma / (line.v * (1 + 1e-10))

        x = np.nanmin(np.where(self.xlist > 0, self.xlist, np.inf))
        res = line.point(x)

        return res


class DynamicVoronoi(Voronoi):
    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
        seed="random",
        spokes=2,
        n_seeds=5,
    ):

        """lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None, seed="random", spokes=2, n_seeds=5

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        seed : {random, [[float]]}
            If 'random' Voronoi centroid will be initialized randomly. Else: if a list of points is given, the DynamicVoronoi will use them.

        spokes : int, default=2
            Number of randoms spokes to draw during hyperplane sampling (SpokeDart)

        n_seeds : int, default=2
            Number of centroids at each decomposition.

        """

        super().__init__(
            lo_bounds,
            up_bounds,
            father,
            level,
            id,
            children,
            score,
            seed=seed,
            n_seeds=n_seeds,
        )

        self.n_dim = 2 * self.dim
        self.spokes = spokes

        self.sampled_bounds = []

    def create_children(self):

        update_neighbors = False

        if not isinstance(self.father, str):
            # Current cell will be it's own children
            selfchild = DynamicVoronoi(
                self.lo_bounds,
                self.up_bounds,
                self,
                self.level + 1,
                self.id,
                seed=self.seed,
                spokes=self.spokes,
                n_seeds=self.n_seeds,
            )
            self.children.append(selfchild)
            # Replace previous cell by new cell
            self.all_seeds[self.id] = selfchild

        for i in self.next_seeds:
            child = DynamicVoronoi(
                self.lo_bounds,
                self.up_bounds,
                self,
                self.level + 1,
                len(self.all_seeds),
                seed=i,
                spokes=self.spokes,
                n_seeds=self.n_seeds,
            )
            self.children.append(child)
            self.all_seeds.append(child)

        for child in self.children:
            for cell in self.all_seeds:
                cell.sampled_bounds = []
                cell.next_seeds = []
                if child.id != cell.id:
                    try:
                        h = Hyperplane(self.children[i], self.children[j])
                        self.children[i].hyperplanes.append(h)
                        self.children[j].hyperplanes.append(h)
                    except AssertionError as e:
                        logger.warning(f"Hyperplane building aborted: {e}")

        for i, c in enumerate(self.all_seeds):
            logger.info(f"Building children n°{i}/{len(self.children)}")
            if len(c.hyperplanes) > 0:
                c.update()
            else:
                self.children.pop(i)

    def update(self):
        upma = self.up_bounds - self.seed
        loma = self.lo_bounds - self.seed

        cell_idx = [False] * len(self.hyperplanes)

        # Fixed points (2 for each dimension)
        for d in range(self.n_dim):

            l = self.dimSpoke(self.seed, d % self.dim)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)

                        cell_idx[j] = False

                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:

                pfar_clipped = self.clipBorder(l, upma, loma)
                self.sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)

                a = np.copy(inter[minidx])

                if cell_idx[minidx]:
                    self.sampled_hyperplanes.add(minidx)

                    if self.hyperplanes[minidx].cellX != self:
                        self.hyperplanes[minidx].cellX.sampled_bounds.append(a)

                    else:
                        self.hyperplanes[minidx].cellY.sampled_bounds.append(a)

                self.sampled_bounds.append(a)

        # Random points
        for d in range(self.spokes):

            l = self.randomSpoke(self.seed)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)

                        cell_idx[j] = False

                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:

                pfar_clipped = self.clipBorder(l, upma, loma)
                self.sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)

                a = np.copy(inter[minidx])

                if cell_idx[minidx]:
                    self.sampled_hyperplanes.add(minidx)

                    if self.hyperplanes[minidx].cellX != self:
                        self.hyperplanes[minidx].cellX.sampled_bounds.append(a)

                    else:
                        self.hyperplanes[minidx].cellY.sampled_bounds.append(a)

                self.sampled_bounds.append(a)

        dist = np.linalg.norm(np.array(self.sampled_bounds) - self.seed, axis=1)

        dist = np.nan_to_num(dist)
        sum = np.sum(dist)

        if sum != 0:
            p = dist / sum
            choosen = np.random.choice(
                list(range(len(self.sampled_bounds))),
                np.minimum(np.count_nonzero(p), self.n_seeds),
                replace=False,
                p=p,
            )

            for c in choosen:
                self.next_seeds.append(
                    self.shiftBorder(self.seed, self.sampled_bounds[c])
                )


class FixedVoronoi(Voronoi):
    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
        seed="random",
        spokes=2,
        n_seeds=5,
    ):

        """__init__(lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None, seed="random", spokes=2, n_seeds=5)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        seed : {random, [[float]]}
            If 'random' Voronoi centroid will be initialized randomly. Else: if a list of points is given, the DynamicVoronoi will use them.

        spokes : int, default=2
            Number of randoms spokes to draw during hyperplane sampling (SpokeDart)

        n_seeds : int, default=2
            Number of centroids at each decomposition.

        """

        super().__init__(
            lo_bounds,
            up_bounds,
            father,
            level,
            id,
            children,
            score,
            seed=seed,
            n_seeds=n_seeds,
        )

        self.dim = len(self.up_bounds)
        self.n_dim = 2 * self.dim
        self.spokes = spokes

        self.sampled_bounds = []

    def create_children(self):

        logger.info(f"Creating children of n°{self.id}")

        # if not isinstance(self.father, str):
        #     # Current cell will be it's own children
        #     selfchild = FixedVoronoi(self.lo_bounds, self.up_bounds, self, self.level + 1, self.id, seed=self.seed, spokes=self.spokes, n_seeds=self.n_seeds)
        #     selfchild.hyperplanes = self.hyperplanes[:]
        #     self.children.append(selfchild)
        #
        #     # Replace previous cell by new cell
        #     self.all_seeds.append(selfchild)

        for i in self.next_seeds:
            child = FixedVoronoi(
                self.lo_bounds,
                self.up_bounds,
                self,
                self.level + 1,
                len(self.all_seeds),
                seed=i,
                spokes=self.spokes,
                n_seeds=self.n_seeds,
            )
            child.hyperplanes = self.hyperplanes[:]
            self.children.append(child)
            self.all_seeds.append(child)

        for i in range(len(self.children) - 1):
            for j in range(i + 1, len(self.children)):
                try:
                    h = Hyperplane(self.children[i], self.children[j])
                    self.children[i].hyperplanes.append(h)
                    self.children[j].hyperplanes.append(h)
                except AssertionError as e:
                    logger.warning(f"Hyperplane building aborted: {e}")

        for i, c in enumerate(self.children):
            logger.info(f"Building children n°{i}/{len(self.children)}")
            if len(c.hyperplanes) > 0:
                c.update()
            else:
                self.children.pop(i)

    def update(self):

        upma = self.up_bounds - self.seed
        loma = self.lo_bounds - self.seed

        cell_idx = [False] * len(self.hyperplanes)

        # Fixed points (2 for each dimension)
        for d in range(self.n_dim):

            l = self.dimSpoke(self.seed, d % self.dim)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)

                        cell_idx[j] = False

                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:

                pfar_clipped = self.clipBorder(l, upma, loma)
                self.sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)

                a = np.copy(inter[minidx])

                if cell_idx[minidx]:

                    if self.hyperplanes[minidx].cellX != self:
                        self.hyperplanes[minidx].cellX.sampled_bounds.append(a)

                    else:
                        self.hyperplanes[minidx].cellY.sampled_bounds.append(a)

                self.sampled_bounds.append(a)

        # Random points
        for d in range(self.spokes):

            l = self.randomSpoke(self.seed)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)

                        cell_idx[j] = False

                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:

                pfar_clipped = self.clipBorder(l, upma, loma)
                self.sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)

                a = np.copy(inter[minidx])

                if cell_idx[minidx]:

                    if self.hyperplanes[minidx].cellX != self:
                        self.hyperplanes[minidx].cellX.sampled_bounds.append(a)

                    else:
                        self.hyperplanes[minidx].cellY.sampled_bounds.append(a)

                self.sampled_bounds.append(a)

        dist = np.linalg.norm(np.array(self.sampled_bounds) - self.seed, axis=1)

        dist = np.nan_to_num(dist)
        sum = np.sum(dist)

        if sum != 0:
            p = dist / sum
            choosen = np.random.choice(
                list(range(len(self.sampled_bounds))),
                np.minimum(np.count_nonzero(p), self.n_seeds),
                replace=False,
                p=p,
            )

            for c in choosen:
                self.next_seeds.append(
                    self.shiftBorder(self.seed, self.sampled_bounds[c])
                )


class LightFixedVoronoi(Voronoi):
    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
        seed="random",
        spokes=2,
        n_seeds=5,
    ):

        """lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None, seed="random", spokes=2, n_seeds=5

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        seed : {random, [[float]]}
            If 'random' Voronoi centroid will be initialized randomly. Else: if a list of points is given, the DynamicVoronoi will use them.

        spokes : int, default=2
            Number of randoms spokes to draw during hyperplane sampling (SpokeDart)

        n_seeds : int, default=2
            Number of centroids at each decomposition.

        """

        super().__init__(
            lo_bounds,
            up_bounds,
            father,
            level,
            id,
            children,
            score,
            seed=seed,
            n_seeds=n_seeds,
        )

        self.dim = len(self.up_bounds)
        self.n_dim = 2 * self.dim
        self.spokes = spokes

        self.sampled_bounds = []

    def create_children(self):

        logger.info(f"Creating children of n°{self.id}")

        # if not isinstance(self.father, str):
        #     # Current cell will be it's own children
        #     selfchild = LightFixedVoronoi(self.lo_bounds, self.up_bounds, self, self.level + 1, self.id, seed=self.seed, spokes=self.spokes, n_seeds=self.n_seeds)
        #     selfchild.hyperplanes = self.hyperplanes[:]
        #     self.children.append(selfchild)

        for i in self.next_seeds:
            child = LightFixedVoronoi(
                self.lo_bounds,
                self.up_bounds,
                self,
                self.level + 1,
                len(self.children),
                seed=i,
                spokes=self.spokes,
                n_seeds=self.n_seeds,
            )
            child.hyperplanes = self.hyperplanes[:]

            self.children.append(child)

        for i in range(len(self.children) - 1):
            for j in range(i + 1, len(self.children)):
                try:
                    h = Hyperplane(self.children[i], self.children[j])
                    self.children[i].hyperplanes.append(h)
                    self.children[j].hyperplanes.append(h)
                except AssertionError as e:
                    logger.warning(f"Hyperplane building aborted: {e}")

        for i, c in enumerate(self.children):
            logger.info(f"Building children n°{i}/{len(self.children)}")
            if len(c.hyperplanes) > 0:
                c.update()
            else:
                self.children.pop(i)

    def update(self):

        sampled_hyperplanes = set()

        upma = self.up_bounds - self.seed
        loma = self.lo_bounds - self.seed

        cell_idx = [False] * len(self.hyperplanes)

        # Fixed points (2 for each dimension)
        for d in range(self.n_dim):

            l = self.dimSpoke(self.seed, d % self.dim)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)
                        cell_idx[j] = False
                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:
                pfar_clipped = self.clipBorder(l, upma, loma)
                self.sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)

                a = np.copy(inter[minidx])

                if cell_idx[minidx]:
                    sampled_hyperplanes.add(minidx)

                self.sampled_bounds.append(a)

        # Random points
        for d in range(self.spokes):

            l = self.randomSpoke(self.seed)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)
                        cell_idx[j] = False
                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:
                pfar_clipped = self.clipBorder(l, upma, loma)
                self.sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)

                a = np.copy(inter[minidx])

                if cell_idx[minidx]:
                    sampled_hyperplanes.add(minidx)

                self.sampled_bounds.append(a)

        self.hyperplanes = [
            self.hyperplanes[idx] for idx in sampled_hyperplanes
        ]

        dist = np.linalg.norm(np.array(self.sampled_bounds) - self.seed, axis=1)

        dist = np.nan_to_num(dist)
        sum = np.sum(dist)

        if sum != 0:
            p = dist / sum
            choosen = np.random.choice(
                list(range(len(self.sampled_bounds))),
                np.minimum(np.count_nonzero(p), self.n_seeds),
                replace=False,
                p=p,
            )

            for c in choosen:
                self.next_seeds.append(
                    self.shiftBorder(self.seed, self.sampled_bounds[c])
                )


class BoxedVoronoi(Voronoi):
    def __init__(
        self,
        lo_bounds,
        up_bounds,
        father="root",
        level=0,
        id=0,
        children=[],
        score=None,
        seed="random",
        spokes=2,
        n_seeds=5,
    ):

        """lo_bounds, up_bounds, father="root", level=0, id=0, children=[], score=None, seed="random", spokes=2, n_seeds=5

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal, default='root'
            Reference to the parent of the current fractal.

        level : int, default=0
            Current level of the fractal in the partition tree. See Tree_search.

        id : int, default=0
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no child is given, children will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        seed : {random, [[float]]}
            If 'random' Voronoi centroid will be initialized randomly. Else: if a list of points is given, the DynamicVoronoi will use them.

        spokes : int, default=2
            Number of randoms spokes to draw during hyperplane sampling (SpokeDart)

        n_seeds : int, default=2
            Number of centroids at each decomposition.

        """

        super().__init__(
            lo_bounds,
            up_bounds,
            father,
            level,
            id,
            children,
            score,
            seed=seed,
            n_seeds=n_seeds,
        )

        self.dim = len(self.up_bounds)
        self.n_dim = 2 * self.dim
        self.spokes = spokes

    def create_children(self):

        logger.info(f"Creating children of n°{self.id}")

        self.next_seeds = np.random.uniform(
            self.lo_bounds, self.up_bounds, (self.n_seeds, self.dim)
        )

        for i in self.next_seeds:
            child = BoxedVoronoi(
                self.lo_bounds,
                self.up_bounds,
                self,
                self.level + 1,
                len(self.children),
                seed=i,
                spokes=self.spokes,
                n_seeds=self.n_seeds,
            )
            self.children.append(child)

        for i in range(len(self.children) - 1):
            for j in range(i + 1, len(self.children)):
                try:
                    h = Hyperplane(self.children[i], self.children[j])
                    self.children[i].hyperplanes.append(h)
                    self.children[j].hyperplanes.append(h)
                except AssertionError as e:
                    logger.warning(f"Hyperplane building aborted: {e}")

        for i, c in enumerate(self.children):
            logger.info(f"Building children n°{i}/{len(self.children)}")
            if len(c.hyperplanes) > 0:
                c.update()
            else:
                self.children.pop(i)

    def update(self):

        sampled_bounds = []

        upma = self.up_bounds - self.seed
        loma = self.lo_bounds - self.seed

        cell_idx = [False] * len(self.hyperplanes)

        # Fixed points (2 for each dimension)
        for d in range(self.n_dim):

            l = self.dimSpoke(self.seed, d % self.dim)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)

                        cell_idx[j] = False

                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:
                pfar_clipped = self.clipBorder(l, upma, loma)
                sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)
                a = np.copy(inter[minidx])
                sampled_bounds.append(a)

        # Random points
        for d in range(self.spokes):

            l = self.randomSpoke(self.seed)

            inter = np.empty((0, self.dim))

            for j, h in enumerate(self.hyperplanes):

                on, pfar = h.intersection(l)

                if on:
                    # Clip line to bounds
                    if np.any(pfar > self.up_bounds) or np.any(
                        pfar < self.lo_bounds
                    ):

                        pfar_clipped = self.clipBorder(l, upma, loma)
                        inter = np.append(inter, [pfar_clipped], axis=0)

                        cell_idx[j] = False

                    else:
                        inter = np.append(inter, [pfar], axis=0)

                        cell_idx[j] = True
                else:
                    cell_idx[j] = False

            if len(inter) == 0:
                pfar_clipped = self.clipBorder(l, upma, loma)
                sampled_bounds.append(np.copy(pfar_clipped))

            else:
                dist = np.linalg.norm(inter - l.A, axis=1)
                minidx = np.argmin(dist)
                a = np.copy(inter[minidx])
                sampled_bounds.append(a)

        self.lo_bounds = np.nanmin(sampled_bounds, axis=0)
        self.up_bounds = np.nanmax(sampled_bounds, axis=0)
        del self.hyperplanes
