# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-20T11:51:02+02:00
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
from zellij.core.search_space import Searchspace
from zellij.core.variables import FloatVar, Constant
from zellij.strategies.utils.direct_utils import SigmaInf

import logging

logger = logging.getLogger("zellij.fractal")


class Fractal(Searchspace):
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
    :ref:`lf` : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    :ref:`sp` : Initial search space used to build fractal.
    Hypercube : Inherited Fractal type
    Hypersphere : Inherited Fractal type
    """

    _instances = {}
    _instances_count = {}
    _god = None

    def __init__(
        self,
        values,
        loss,
        heuristic="prout",
        **kwargs,
    ):

        """__init__(self,values,loss,father="root",level=0,id=0,children=[],score=None,**kwargs)

        Parameters
        ----------
        heuristic : callable
            Determine using using current state of the algorithm, how to score the current fractal. Used informations are given to the function at the following order:
            - The current fractal
            - The best solution found so far (converted to continuous)
            - The best score found so far (computed with the loss function)
        """

        super(Fractal, self).__init__(values, loss, **kwargs)

        self.children = []
        self.score = float("inf")

    def __new__(cls, *args, **kwargs):
        obj = super(Fractal, cls).__new__(cls)
        if cls not in cls._instances:
            obj.id = 0
            obj.father = "root"
            obj.level = 0
            cls._god = obj
            cls._instances[cls] = obj
            cls._instances_count[cls] = 0
        else:
            cls._instances_count[cls] += 1
            obj.id = cls._instances_count[cls]
            obj.level = None
            obj.father = None

        return obj

    @abc.abstractmethod
    def create_children(self):
        """create_children(self)

        Abstract method which will create children of the current Fractal object, according to certain rules (Hypercube, Hypersphere...)

        """
        pass

    def compute_score(self, idx):
        self.score = self.heuristic(self.loss, idx)

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, f):
        assert f != self, f"Father of a Fractal cannot be `self`"
        self._father = f

    def subspace(self, low_bounds, up_bounds, **kwargs):
        new = super(Fractal, self).subspace(low_bounds, up_bounds, **kwargs)
        new.father = self
        new.level = self.level + 1
        return new


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
        values,
        loss,
        heuristic="prout",
        **kwargs,
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

        super(Hypercube, self).__init__(values, loss, **kwargs)

        continuous = True
        for v in self.values:
            if not (
                isinstance(v, FloatVar)
                or (isinstance(v, Constant) and isinstance(v.value, float))
            ):
                continuous = False

        if continuous:
            self.lo_bounds = np.zeros(self.size)
            self.up_bounds = np.zeros(self.size)
            for i, v in enumerate(self.values):
                if isinstance(v, FloatVar):
                    self.lo_bounds[i] = v.low_bound
                    self.up_bounds[i] = v.up_bound
                else:
                    self.lo_bounds[i] = v.value
                    self.up_bounds[i] = v.value

            self.to_convert = False

        elif all(hasattr(v, "to_continuous") for v in self.values):
            logger.warning(
                f"""Be carefull, for {self.__class__.__name__}  with
            mixed variables, the Searchspace wil be approximated by the unit
            hypercube. Upper and lower bounds will be between [0,1],
            the `to_continuous` conversion method must take this into account.
            For example, Minmax converter can be used."""
            )

            assert hasattr(
                self, "to_continuous"
            ), f"""When {self.__class__.__name__} as mixed variables,
            a `to_continuous` method must be implemented.
            Use the `to_continuous` kwargs when defining the :ref:`Searchspace`
            """

            self.lo_bounds = np.array([0.0] * self.size)
            self.up_bounds = np.array([1.0] * self.size)
            self.to_convert = True

        else:
            raise ValueError(
                f"""For {self.__class__.__name__}, all variables
            must be `FloatVar`, or all variables must have a `to_continuous`
            method added at the initialization of the variable.
            Got {self.values}.
            ex:\n>>> FloatVar("test",-5,5,to_continuous=...)."""
            )

    def create_children(self):

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

        up_m_lo = self.up_bounds - self.lo_bounds
        radius = np.abs(up_m_lo / 2)
        bounds = [[self.lo_bounds, self.up_bounds]]

        # Hyperplan bisecting
        next_b = []
        for i in range(self.size):
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
        if self.to_convert:
            for b in bounds:
                h = self.subspace(
                    self.to_continuous.reverse(b[0]),
                    self.to_continuous.reverse(b[1]),
                )
                self.children.append(h)
        else:
            for b in bounds:
                h = self.subspace(list(b[0]), list(b[1]))
                self.children.append(h)

    def __repr__(self):

        return (
            super(Hypercube, self).__repr__()
            + "\n"
            + "ID: "
            + str(self.id)
            + " son of "
            + f"{self.father.id}"
            + "\n"
            + "BOUNDS: "
            + str(self.lo_bounds)
            + "|"
            + str(self.up_bounds)
            + "\n"
        )


class Hypersphere(Fractal):

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
        values,
        loss,
        heuristic="prout",
        inflation=1.75,
        compute_bounds=False,
        **kwargs,
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

        super(Hypersphere, self).__init__(values, loss, **kwargs)

        self.inflation = inflation
        self.compute_bounds = compute_bounds

        continuous = True
        for v in self.values:
            if not (
                isinstance(v, FloatVar)
                or (isinstance(v, Constant) and isinstance(v.value, float))
            ):
                continuous = False

        if continuous:
            self.to_convert = False

            if self.compute_bounds or self.level == 0:
                self.lo_bounds = np.zeros(self.size)
                self.up_bounds = np.zeros(self.size)
                for i, v in enumerate(self.values):
                    if isinstance(v, FloatVar):
                        self.lo_bounds[i] = v.low_bound
                        self.up_bounds[i] = v.up_bound
                    else:
                        self.lo_bounds[i] = v.value
                        self.up_bounds[i] = v.value

        elif all(hasattr(v, "to_continuous") for v in self.values):
            logger.warning(
                f"""Be carefull, for {self.__class__.__name__}  with
            mixed variables, the Searchspace wil be approximated by the unit
            hypercube. Upper and lower bounds will be between [0,1],
            the `to_continuous` conversion method must take this into account.
            For example, Minmax converter can be used."""
            )

            assert hasattr(
                self, "to_continuous"
            ), f"""When {self.__class__.__name__} as mixed variables,
            a `to_continuous` method must be implemented.
            Use the `to_continuous` kwargs when defining the :ref:`Searchspace`
            """

            self.to_convert = True

            if self.compute_bounds or self.level == 0:
                self.lo_bounds = np.array([0.0] * self.size)
                self.up_bounds = np.array([1.0] * self.size)

        else:
            raise ValueError(
                f"""For {self.__class__.__name__}, all variables
            must be `FloatVar`, or all variables must have a `to_continuous`
            method added at the initialization of the variable.
            Got {self.values}.
            ex:\n>>> FloatVar("test",-5,5,to_continuous=...)."""
            )

        if self.level == 0:
            up_m_lo = self.up_bounds - self.lo_bounds
            self.center = self.lo_bounds + (up_m_lo) / 2
            self.radius = up_m_lo[0] / 2 * self.inflation
        else:
            self.center = None
            self.radius = None

    def create_children(self):

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

        r_p = self.radius / (1 + np.sqrt(2))

        for i in range(self.size):

            center = np.copy(self.center)
            center[i] += self.radius - r_p

            lo = np.maximum(center - r_p, self._god.lo_bounds)
            up = np.minimum(center + r_p, self._god.up_bounds)

            if self.to_convert:
                h = self.subspace(
                    self.to_continuous.reverse(lo, sub_values=True),
                    self.to_continuous.reverse(up, sub_values=True),
                )
            else:
                h = self.subspace(lo, up)
            h.center = np.minimum(
                np.maximum(center, self._god.lo_bounds), self._god.up_bounds
            )
            h.radius = r_p
            self.children.append(h)

            center = np.copy(self.center)
            center[i] -= self.radius - r_p

            lo = np.maximum(center - r_p, self._god.lo_bounds)
            up = np.minimum(center + r_p, self._god.up_bounds)

            if self.to_convert:
                h = self.subspace(
                    self.to_continuous.reverse(lo, sub_values=True),
                    self.to_continuous.reverse(up, sub_values=True),
                )
            else:
                h = self.subspace(lo, up)
            h.center = np.minimum(
                np.maximum(center, self._god.lo_bounds), self._god.up_bounds
            )
            h.radius = r_p
            self.children.append(h)

    def subspace(self, low_bounds, up_bounds):
        new = super(Hypersphere, self).subspace(low_bounds, up_bounds)
        new.inflation = self.inflation
        return new

    def __repr__(self):

        return (
            super(Hypersphere, self).__repr__()
            + "\n"
            + "ID: "
            + str(self.id)
            + " son of "
            + f"{self.father.id}"
            + "\n"
            + "Center: "
            + str(self.center)
            + "\n"
            + "Radius: "
            + str(self.radius)
            + "\n"
            + "Inflation: "
            + str(self.inflation)
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
        values,
        loss,
        heuristic="prout",
        section=2,
        **kwargs,
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

        super(Section, self).__init__(values, loss, **kwargs)

        assert section > 1, logger.error(
            f"{section}-Section is not possible, section must be > 1"
        )

        self.section = section

        continuous = True
        for v in self.values:
            if not (
                isinstance(v, FloatVar)
                or (isinstance(v, Constant) and isinstance(v.value, float))
            ):
                continuous = False

        if continuous:
            self.lo_bounds = np.zeros(self.size)
            self.up_bounds = np.zeros(self.size)
            for i, v in enumerate(self.values):
                if isinstance(v, FloatVar):
                    self.lo_bounds[i] = v.low_bound
                    self.up_bounds[i] = v.up_bound
                else:
                    self.lo_bounds[i] = v.value
                    self.up_bounds[i] = v.value

            self.to_convert = False

        elif all(hasattr(v, "to_continuous") for v in self.values):
            logger.warning(
                f"""Be carefull, for {self.__class__.__name__}  with
            mixed variables, the Searchspace wil be approximated by the unit
            hypercube. Upper and lower bounds will be between [0,1],
            the `to_continuous` conversion method must take this into account.
            For example, Minmax converter can be used."""
            )

            assert hasattr(
                self, "to_continuous"
            ), f"""When {self.__class__.__name__} as mixed variables,
            a `to_continuous` method must be implemented.
            Use the `to_continuous` kwargs when defining the :ref:`Searchspace`
            """

            self.lo_bounds = np.array([0.0] * self.size)
            self.up_bounds = np.array([1.0] * self.size)
            self.to_convert = True

        else:
            raise ValueError(
                f"""For {self.__class__.__name__}, all variables
            must be `FloatVar`, or all variables must have a `to_continuous`
            method added at the initialization of the variable.
            Got {self.values}.
            ex:\n>>> FloatVar("test",-5,5,to_continuous=...)."""
            )

        up_m_lo = self.up_bounds - self.lo_bounds
        self.longest = np.argmax(up_m_lo)
        self.width = up_m_lo[self.longest]
        self.center = up_m_lo / 2

    def create_children(self):

        new_val = self.width / self.section

        lo = np.copy(self.lo_bounds)
        up = np.copy(self.up_bounds)
        up[self.longest] = lo[self.longest] + new_val

        for i in range(self.section):

            if self.to_convert:
                h = self.subspace(
                    self.to_continuous.reverse(lo, sub_values=True),
                    self.to_continuous.reverse(up, sub_values=True),
                    section=self.section,
                )
            else:
                h = self.subspace(lo, up, section=self.section)

            self.children.append(h)

            lo = np.copy(h.lo_bounds)
            up = np.copy(h.up_bounds)
            lo[self.longest] += new_val
            up[self.longest] += new_val

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


class Direct(Fractal):

    """Direct

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
        values,
        loss,
        max_calls,
        heuristic="prout",
        force_convert=False,
        sigma=SigmaInf(),
        **kwargs,
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

        super(Direct, self).__init__(values, loss, **kwargs)

        self.force_convert = force_convert

        count_constant = 0

        continuous = True
        for v in self.values:
            if isinstance(v, Constant):
                count_constant += 1
                if not isinstance(v.value, float):
                    continuous = False
            elif not isinstance(v, FloatVar):
                continuous = False

        if count_constant == len(self.values):
            self.is_constant = True
        else:
            self.is_constant = False

        if continuous and not self.force_convert:
            self.lo_bounds = np.zeros(self.size)
            self.up_bounds = np.zeros(self.size)
            for i, v in enumerate(self.values):
                if isinstance(v, FloatVar):
                    self.lo_bounds[i] = v.low_bound
                    self.up_bounds[i] = v.up_bound
                else:
                    self.lo_bounds[i] = v.value
                    self.up_bounds[i] = v.value

            self.to_convert = False

        elif all(hasattr(v, "to_continuous") for v in self.values):
            # logger.warning(
            #     f"""
            # Be carefull, for {self.__class__.__name__}  with
            # mixed variables, the Searchspace wil be approximated by the unit
            # hypercube. Upper and lower bounds will be between [0,1],
            # the `to_continuous` conversion method must take this into account.
            # For example, Minmax converter can be used.
            # """
            # )

            assert hasattr(
                self, "to_continuous"
            ), f"""
            When {self.__class__.__name__} as mixed variables,
            a `to_continuous` method must be implemented.
            Use the `to_continuous` kwargs when defining the :ref:`Searchspace`
            """
            lo = np.zeros(self.size)
            up = np.zeros(self.size)
            for i, v in enumerate(self.values):
                if isinstance(v, FloatVar):
                    lo[i] = v.low_bound
                    up[i] = v.up_bound
                else:
                    lo[i] = v.value
                    up[i] = v.value
            self.lo_bounds = np.array(
                self.to_continuous.convert(lo, sub_values=True)
            )
            self.up_bounds = np.array(
                self.to_continuous.convert(up, sub_values=True)
            )
            self.to_convert = True

        else:
            raise ValueError(
                f"""
            For {self.__class__.__name__}, all variables
            must be `FloatVar`, or all variables must have a `to_continuous`
            method added at the initialization of the variable.
            Got {self.values}.
            ex:\n>>> FloatVar("test",-5,5,to_continuous=...).
            """
            )

        up_m_lo = self.up_bounds - self.lo_bounds
        self.longest = np.argmax(up_m_lo)
        self.width = up_m_lo[self.longest]
        self.center = (self.lo_bounds + self.up_bounds) * 0.5

        if self.level == 0:
            if self.to_convert:
                self.score = self.loss(
                    self.to_continuous.reverse([self.center], sub_values=True)
                )[0]
            else:
                self.score = self.loss([self.center])[0]

            self.length = 1.0
        else:
            self.length = None

        self.set_i = np.where(up_m_lo == up_m_lo[self.longest])[0]

        assert max_calls > 3, logger.error(
            f"{max_calls} must be greater than 3"
        )
        self.max_calls = max_calls

        self.stage = 0
        self.sigma = sigma

    def create_children(self):

        section_length = self.width / 3
        dim = 0
        points = np.empty((0, self.size), dtype=float)
        # While there is dimensions of equal length or remaining calls to loss
        while (
            dim < len(self.set_i)
            and self.loss.calls + dim * 2 <= self.max_calls
        ):
            new_p = np.repeat([self.center], 2, axis=0)
            new_p[0][self.set_i[dim]] -= section_length
            new_p[1][self.set_i[dim]] += section_length
            points = np.append(points, new_p, axis=0)
            dim += 1

        if len(points) > 0:
            if self.to_convert:
                scores = self.loss(
                    self.to_continuous.reverse(points, sub_values=True)
                )
            else:
                scores = self.loss(points)

            scores = np.reshape(scores, (-1, 2))
            scores_dim = scores.min(axis=1)
            argsort = np.argsort(scores_dim)

            current_section = self
            for stage, arg in enumerate(argsort):

                lo = np.copy(current_section.lo_bounds)
                up = np.copy(current_section.up_bounds)
                up[self.set_i[arg]] = lo[self.set_i[arg]] + section_length
                up = np.minimum(up, self._god.up_bounds)
                children = []
                # Build sections
                for i in range(3):

                    if self.to_convert:

                        h = current_section.subspace(
                            current_section.to_continuous.reverse(
                                lo, sub_values=True
                            ),
                            current_section.to_continuous.reverse(
                                up, sub_values=True
                            ),
                            max_calls=self.max_calls,
                            force_convert=self.force_convert,
                            sigma=self.sigma,
                        )
                    else:

                        h = current_section.subspace(
                            lo,
                            up,
                            max_calls=self.max_calls,
                            force_convert=self.force_convert,
                            sigma=self.sigma,
                        )

                    if not h.is_constant:
                        h.father = self
                        h.stage = stage
                        h.length = self.sigma(h)
                        h.level = int(
                            (
                                np.log(
                                    self._god.up_bounds[h.longest]
                                    - self._god.lo_bounds[h.longest]
                                )
                                - np.log(h.width)
                            )
                            / np.log(3)
                        )
                        children.append(h)

                    lo[self.set_i[arg]] += section_length
                    up[self.set_i[arg]] += section_length

                if children:
                    children[0].score, children[2].score = scores[arg]
                    self.children.append(children[0])
                    self.children.append(children[2])

                    children[1].score = self.score
                    current_section = children[1]

            if current_section != self:
                self.children.append(current_section)

    def __repr__(self):
        if type(self.father) == str:
            id = "root"
        else:
            id = str(self.father.id)

        return (
            super(Direct, self).__repr__()
            + "\n"
            + "ID: "
            + str(self.id)
            + " son of "
            + f"{id}"
            + "\n"
            + "BOUNDS: "
            + str(self.lo_bounds)
            + "|"
            + str(self.up_bounds)
            + "\n"
        )


class Soo(Fractal):

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
        values,
        loss,
        max_calls,
        heuristic="prout",
        section=3,
        force_convert=False,
        **kwargs,
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

        super(Soo, self).__init__(values, loss, **kwargs)

        assert (
            section > 1
        ), f"""
        Cannot divide a hypercube into {section} parts. Section must be >1
        """
        self.section = section

        self.force_convert = force_convert

        count_constant = 0

        continuous = True
        for v in self.values:
            if isinstance(v, Constant):
                count_constant += 1
                if not isinstance(v.value, float):
                    continuous = False
            elif not isinstance(v, FloatVar):
                continuous = False

        if count_constant == len(self.values):
            self.is_constant = True
        else:
            self.is_constant = False

        if continuous and not self.force_convert:
            self.lo_bounds = np.zeros(self.size)
            self.up_bounds = np.zeros(self.size)
            for i, v in enumerate(self.values):
                if isinstance(v, FloatVar):
                    self.lo_bounds[i] = v.low_bound
                    self.up_bounds[i] = v.up_bound
                else:
                    self.lo_bounds[i] = v.value
                    self.up_bounds[i] = v.value

            self.to_convert = False

        elif all(hasattr(v, "to_continuous") for v in self.values):
            # logger.warning(
            #     f"""
            # Be carefull, for {self.__class__.__name__}  with
            # mixed variables, the Searchspace wil be approximated by the unit
            # hypercube. Upper and lower bounds will be between [0,1],
            # the `to_continuous` conversion method must take this into account.
            # For example, Minmax converter can be used.
            # """
            # )

            assert hasattr(
                self, "to_continuous"
            ), f"""
            When {self.__class__.__name__} as mixed variables,
            a `to_continuous` method must be implemented.
            Use the `to_continuous` kwargs when defining the :ref:`Searchspace`
            """
            lo = np.zeros(self.size)
            up = np.zeros(self.size)
            for i, v in enumerate(self.values):
                if isinstance(v, FloatVar):
                    lo[i] = v.low_bound
                    up[i] = v.up_bound
                else:
                    lo[i] = v.value
                    up[i] = v.value
            self.lo_bounds = np.array(
                self.to_continuous.convert(lo, sub_values=True)
            )
            self.up_bounds = np.array(
                self.to_continuous.convert(up, sub_values=True)
            )
            self.to_convert = True

        else:
            raise ValueError(
                f"""
            For {self.__class__.__name__}, all variables
            must be `FloatVar`, or all variables must have a `to_continuous`
            method added at the initialization of the variable.
            Got {self.values}.
            ex:\n>>> FloatVar("test",-5,5,to_continuous=...).
            """
            )

        up_m_lo = self.up_bounds - self.lo_bounds
        self.longest = np.argmax(up_m_lo)
        self.width = up_m_lo[self.longest]
        self.center = (self.lo_bounds + self.up_bounds) * 0.5

        if self.level == 0:
            if self.to_convert:
                self.score = self.loss(
                    self.to_continuous.reverse([self.center], sub_values=True)
                )[0]
            else:
                self.score = self.loss([self.center])[0]

        assert max_calls > 3, logger.error(
            f"{max_calls} must be greater than 3"
        )
        self.max_calls = max_calls

    def create_children(self):

        new_val = self.width / self.section

        lo = np.copy(self.lo_bounds)
        up = np.copy(self.up_bounds)
        up[self.longest] = lo[self.longest] + new_val
        up = np.minimum(up, self._god.up_bounds)

        children = []
        i = 0
        while i < self.section and self.loss.calls < self.max_calls:
            i += 1
            if self.to_convert:
                h = self.subspace(
                    self.to_continuous.reverse(lo, sub_values=True),
                    self.to_continuous.reverse(up, sub_values=True),
                    max_calls=self.max_calls,
                    section=self.section,
                    force_convert=self.force_convert,
                )
            else:
                h = self.subspace(
                    lo,
                    up,
                    max_calls=self.max_calls,
                    section=self.section,
                    force_convert=self.force_convert,
                )

            if not h.is_constant:
                children.append(h)

            lo[self.longest] += new_val
            up[self.longest] += new_val

        if self.section % 2 == 0:
            centers = [child.center for child in children]
            if self.to_convert:
                scores = self.loss(
                    self.to_continuous.reverse(centers, sub_values=True)
                )
            else:
                scores = self.loss(centers)

            for child, s in zip(children, scores):
                child.score = s
        else:
            mid = self.section // 2
            centers = [
                child.center for i, child in enumerate(children) if i != mid
            ]
            if self.to_convert:
                scores = self.loss(
                    self.to_continuous.reverse(centers, sub_values=True)
                )
            else:
                scores = self.loss(centers)

            p = 0
            for idx, child in enumerate(children):
                if idx == mid:
                    p = 1
                    child.score = self.score
                else:
                    child.score = scores[idx - p]

        self.children = children

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
