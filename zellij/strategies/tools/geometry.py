# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.search_space import Fractal, MixedFractal

from typing import Optional, Callable, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.variables import ArrayVar, PermutationVar
    from zellij.strategies.tools.measurements import Measurement

from itertools import product
import numpy as np

import logging

logger = logging.getLogger("zellij.geometry")


class Hypercube(Fractal):
    """Hypercube

    The hypercube is a basic hypervolume used to decompose the :ref:`sp`.
    It is also one of the most computationally inefficient in high dimension.
    The partition complexity of an Hypercube is equal to :math:`O(2^d)`.

    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypersphere : Another hypervolume, with different properties

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> print(sp, sp.lower, sp.upper)
    Hypercube(0,-1,0) [0. 0.] [1. 1.]
    >>> print(sp.get_id())
    (0, -1, 0)
    >>> p = sp.random_point()
    >>> print(p)
    [0.24742878610332475, 0.24380936977381884]
    >>> p = sp.random_point(2)
    >>> d = sp.distance(p[0], p[1])
    >>> print(f"{d:.2f}")
    0.73
    >>> children = sp.create_children()
    >>> for c in children:
    ...     print(f"id:{c.get_id()}:[{c.lower},{c.upper}]")
    id:(1, 0, 0):[[0. 0.],[0.5 0.5]]
    id:(1, 0, 1):[[0.  0.5],[0.5 1. ]]
    id:(1, 0, 2):[[0.5 0. ],[1.  0.5]]
    id:(1, 0, 3):[[0.5 0.5],[1. 1.]]
    """

    def __init__(
        self,
        variables: ArrayVar,
        measurement: Optional[Measurement] = None,
    ):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar` and all must have converter.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        """
        super(Hypercube, self).__init__(variables, measurement)

    def create_children(self):
        """create_children(self)

        Partition function.
        """

        children = super().create_children(2**self.size)

        center = (self.lower + self.upper) / 2
        vertices = np.array(list(product(*zip(self.lower, self.upper))))
        lower = np.minimum(vertices, center)
        upper = np.maximum(vertices, center)

        for child, l, u in zip(children, lower, upper):
            child.lower = l
            child.upper = u
            child._update_measure()

        return children

    def _modify(self, upper, lower, level, father, f_id, c_id, score, measure):
        super()._modify(level, father, f_id, c_id, score, measure)
        self.upper, self.lower = upper, lower

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update({"upper": self.upper, "lower": self.lower})
        return infos


class Hypersphere(Fractal):
    """Hypersphere

    The Hypersphere is a basic hypervolume used to decompose the :ref:`sp`.
    The partition complexity is equal to :math:`2d`, but it does not fully covers
    the initial :ref:`sp`.

    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties

    Examples
    --------
    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypersphere(a)
    >>> print(sp, sp.center, sp.radius)
    Hypersphere(0,-1,0) [0.5 0.5] 0.5
    >>> print(sp.get_id())
    (0, -1, 0)
    >>> p = sp.random_point()
    >>> print(p)
    [0.5598151051101674, 0.5136607275129557]
    >>> p = sp.random_point(2)
    >>> d = sp.distance(p[0], p[1])
    >>> print(f"{d:.2f}")
    0.59
    >>> children = sp.create_children()
    >>> for c in children:
    ...     print(f"id:{c.get_id()}:[{c.center},{c.radius:.2f}]")
    id:(1, 0, 0):[[0.79289322 0.5       ],0.21]
    id:(1, 0, 1):[[0.5        0.79289322],0.21]
    id:(1, 0, 2):[[0.20710678 0.5       ],0.21]
    id:(1, 0, 3):[[0.5        0.20710678],0.21]

    """

    def __init__(self, variables, measurement: Optional[Measurement] = None):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar` and all must have converter.
        measurement : Measurement, optional
            Defines the measure of a fractal.

        """

        super(Hypersphere, self).__init__(variables, measurement)
        self.center = np.full(self.size, 0.5)
        self.radius = 0.5

    # Return a random point of the search space
    def random_point(self, size: Optional[int] = None) -> Union[list, List[list]]:
        if size:
            points = np.random.normal(size=(size, self.size))
            norm = np.linalg.norm(points, axis=1)[:, None]
            radii = np.random.random((size, 1)) ** (1 / self.size)
            points = self.radius * (radii * points / norm) + self.center
        else:
            points = np.random.normal(size=self.size)
            norm = np.linalg.norm(points)
            radii = np.random.random() ** (1 / self.size)
            points = self.radius * (radii * points / norm) + self.center
        return points.tolist()

    def create_children(self):
        """create_children(self)

        Partition function
        """

        children = super().create_children(2 * self.size)

        r_p = self.radius / (1 + np.sqrt(2))

        centers = np.tile(self.center, (len(children), 1))
        centers[: len(self.center) :].ravel()[:: len(self.center) + 1] += (
            self.radius - r_p
        )
        centers[len(self.center) : :].ravel()[:: len(self.center) + 1] -= (
            self.radius - r_p
        )

        for center, child in zip(centers, children):
            child.center = center
            child.radius = r_p
            child._update_measure()

        return children

    def _modify(
        self,
        center: np.ndarray,
        radius: float,
        level: int,
        father: int,
        f_id: int,
        c_id: int,
        score: float,
        measure: float,
    ):
        super()._modify(level, father, f_id, c_id, score, measure)
        self.center, self.radius = center, radius

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update({"center": self.center, "radius": self.radius})
        return infos


class Section(Fractal):

    """Section

    Performs a n-Section of the search space.

    Attributes
    ----------
    section : int
        Defines in how many equal sections the space should be partitioned.
    lower : list[float]
        Lower bounds of the section
    upper : list[float]
        Upper bounds of the section
    is_middle : boolean
        When section is an odd number.
        If the section is at the middle of the partition, then True.

    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Section

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Section(a, section=3)
    >>> print(sp, sp.lower, sp.upper, sp.section)
    Section(0,-1,0) [0. 0.] [1. 1.] 3
    >>> print(sp.get_id())
    (0, -1, 0)
    >>> p = sp.random_point()
    >>> print(p)
    [0.3977808911066778, 0.9684061309949581]
    >>> p = sp.random_point(2)
    >>> d = sp.distance(p[0], p[1])
    >>> print(f"{d:.2f}")
    0.41
    >>> children = sp.create_children()
    >>> for c in children:
    ...     print(f"id:{c.get_id()}:[{c.lower},{c.upper}]")
    id:(1, 0, 0):[[0. 0.],[0.33333333 1.        ]]
    id:(1, 0, 1):[[0.33333333 0.        ],[0.66666667 1.        ]]
    id:(1, 0, 2):[[0.66666667 0.        ],[1. 1.]]
    """

    def __init__(
        self,
        variables: ArrayVar,
        measurement: Optional[Measurement] = None,
        section: int = 2,
    ):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar` and all must have converter.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        section : int, default=2
            Defines in how many equal sections the space should be decompose.

        """

        super(Section, self).__init__(variables, measurement)
        self.lower = np.zeros(self.size)
        self.upper = np.ones(self.size)
        self.section = section

        self.is_middle = False

    @property
    def section(self) -> int:
        return self._section

    @section.setter
    def section(self, value: int):
        if value <= 1:
            raise InitializationError(
                f"{value}-Section is not possible, section must be > 1"
            )
        else:
            self._section = value

    def create_children(self):
        """create_children()

        Partition function.
        """

        children = super().create_children(self.section)

        up_m_lo = self.upper - self.lower
        longest = np.argmax(up_m_lo)
        width = up_m_lo[longest]

        step = width / self.section

        lowers = np.tile(self.lower, (self.section, 1))
        uppers = np.tile(self.upper, (self.section, 1))

        add = np.arange(0, self.section + 1) * step
        uppers[:, longest] = lowers[:, longest] + add[1:]
        lowers[:, longest] = lowers[:, longest] + add[:-1]

        are_middle = [False] * self.section
        if self.section % 2 != 0:
            are_middle[int(self.section / 2)] = True

        for l, u, child, mid in zip(lowers, uppers, children, are_middle):
            child.lower = l
            child.upper = u
            child.section = self.section
            child.is_middle = mid
            child._update_measure()

        return children

    def _modify(
        self, upper, lower, level, is_middle, father, f_id, c_id, score, measure
    ):
        super()._modify(level, father, f_id, c_id, score, measure)
        self.upper, self.lower = upper, lower
        self.is_middle = is_middle

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update(
            {"upper": self.upper, "lower": self.lower, "is_middle": self.is_middle}
        )
        return infos


class Direct(Fractal):

    """Direct

    DIRECT geometry. Direct cannot be used with :code:`DBA` and :code:`ABDA`.
    See `Direct optimization algorithm user guide <https://repository.lib.ncsu.edu/handle/1840.4/484>`_.
    Modify :code:`upper` last, to update all attributes.
    This version is not the most optimal one.

    Attributes
    ----------
    longest : list[int]
        Index of the dimensions with the longest side of the space.
    width : float
        Value of the longest side of the space.
    center : list[float]
        Center of the space.

    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties
    """

    def __init__(self, variables: ArrayVar, measurement: Optional[Measurement] = None):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar` and all must have converter.
        measurement : Measurement, optional
            Defines the measure of a fractal.

        """

        super(Direct, self).__init__(variables, measurement)
        self.width = 1.0
        self.set_i = list(range(0, self.size))

    def _update_attributes(self):
        up_m_lo = self.upper - self.lower
        longest = np.argmax(up_m_lo)
        self.width = up_m_lo[longest]
        self.set_i = np.where(up_m_lo == up_m_lo[longest])[0]
        self._update_measure()

        self.level = int(-np.log(self.width) / np.log(3))

    def create_children(self):
        """create_children()

        Partition function

        """
        section_length = self.width / 3
        if section_length > 1e-13:
            if self.level == 0:
                scores = np.reshape(self.losses[1:], (-1, 2))
                self.score = self.losses[0]
            else:
                scores = np.reshape(self.losses, (-1, 2))

            children = super().create_children((len(scores) - 1) * 2 + 3)

            scores_dim = scores.min(axis=1)
            argsort = np.argsort(scores_dim)

            current_low = np.copy(self.lower)
            current_up = np.copy(self.upper)

            for idx, arg in enumerate(argsort[:-1]):
                dim = self.set_i[arg]
                c_idx = idx * 2

                # 1st trisection
                children[c_idx].lower = np.copy(current_low)
                children[c_idx].upper = np.copy(current_up)
                children[c_idx].upper[dim] -= 2 * section_length
                children[c_idx].score = scores[arg][0]
                children[c_idx]._update_attributes()

                # 3rd trisection
                children[c_idx + 1].lower = np.copy(current_low)
                children[c_idx + 1].upper = np.copy(current_up)
                children[c_idx + 1].lower[dim] += 2 * section_length
                children[c_idx + 1].score = scores[arg][1]
                children[c_idx + 1]._update_attributes()

                # Middle
                current_low[dim] += section_length
                current_up[dim] -= section_length

            arg = argsort[-1]
            dim = self.set_i[arg]

            children[-3].lower = np.copy(current_low)
            children[-3].upper = np.copy(current_up)
            children[-3].upper[dim] -= 2 * section_length
            children[-3].score = scores[arg][0]
            children[-3]._update_attributes()

            children[-2].lower = np.copy(current_low)
            children[-2].upper = np.copy(current_up)
            children[-2].lower[dim] += section_length
            children[-2].upper[dim] -= section_length
            children[-2].score = self.score
            children[-2]._update_attributes()

            children[-1].lower = np.copy(current_low)
            children[-1].upper = np.copy(current_up)
            children[-1].lower[dim] += 2 * section_length
            children[-1].score = scores[arg][1]
            children[-1]._update_attributes()
        else:
            children = []

        return children

    def _modify(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        width: float,
        set_i: list,
        level: int,
        father: int,
        f_id: int,
        c_id: int,
        score: float,
        measure: float,
    ):
        super()._modify(level, father, f_id, c_id, score, measure)
        self.upper, self.lower = upper, lower
        self.width = width
        self.set_i = set_i

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update(
            {
                "upper": self.upper,
                "lower": self.lower,
                "width": self.width,
                "set_i": self.set_i,
            }
        )
        return infos


class LatinHypercube(Hypercube):
    """LatinHypercube

    Subspaces are built according to a Latin Hypercube Sampling.


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
        variables: ArrayVar,
        measurement: Optional[Measurement] = None,
        ndist: int = 1,
        grid_size: int = 2,
        orthogonal: bool = False,
    ):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar` and all must have converter.
        grid_size : {int, callable}, default=1
            Size of the grid for LHS. :code:`self.size * grid_size` hypercubes will
            be sampled. Given value must be :code:`grid_size > 1`.
            Can be a callable of the form Callable[[int], int].
            A callable taking an int (level) and returning an int (grid).
        ndist : {int, callable}, default=1
            Number of sampled distributions.
            Can be a callable of the form Callable[[int], int].
            A callable taking an int (level) and returning an int (number of distribution).
        orthogonal : boolean, default=False
            If True, performs an orthoganal LHS.
        symmetric : boolean, default=True
            Apply a symmetrization on the LatinHypercube distribution.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        """

        super(LatinHypercube, self).__init__(variables, measurement)

        self.grid_size = grid_size
        self.ndist = ndist

        self.orthogonal = orthogonal

    @property
    def grid_size(self) -> Callable:
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: Union[int, Callable]):
        if callable(value):
            self._grid_size = value
        elif value <= 1:
            raise InitializationError(f"grid_size must be >1, got {value}")
        else:
            self._grid_size = lambda level: value

    @property
    def ndist(self) -> Callable:
        return self._ndist

    @ndist.setter
    def ndist(self, value: Union[int, Callable]):
        if callable(value):
            self._ndist = value
        elif value < 1:
            raise InitializationError(f"ndist must be >0, got {value}")
        else:
            self._ndist = lambda level: value

    def create_children(self):
        """create_children(self)

        Partition function.

        attention collision grille paire
        """

        ndist = self.ndist(self.level)
        grid_size = self.grid_size(self.level)

        children = Fractal.create_children(
            self,
            ndist * grid_size,
            orthogonal=self.orthogonal,
            grid_size=self.grid_size,
            ndist=self.ndist,
        )

        for nd in range(ndist):
            A = np.zeros((self.size, grid_size), dtype=int)

            R = np.zeros((grid_size + 1, self.size))
            r = (self.upper - self.lower) / grid_size

            for k in range(self.size):
                A[k] = np.arange(0, grid_size, dtype=int)
                np.random.shuffle(A[k])

            A = A.T

            for k in range(grid_size):
                for i in range(self.size):
                    R[k, i] = (
                        self.lower[i]
                        + (self.upper[i] - self.lower[i]) * (k) / grid_size
                    )

            Bd = np.zeros((grid_size, self.size))
            sidx = nd * grid_size
            eidx = sidx + grid_size
            for k in range(grid_size):
                for i in range(self.size):
                    Bd[k, i] = R[A[k, i], i]
            for child, l in zip(children[sidx:eidx], Bd):
                child.lower = l
                child.upper = l + r
                child._update_measure()

        return children

    def _modify(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        level: int,
        father: int,
        f_id: int,
        c_id: int,
        score: float,
        measure: float,
    ):
        Fractal._modify(self, level, father, f_id, c_id, score, measure)
        self.upper, self.lower = upper, lower

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update({"upper": self.upper, "lower": self.lower})
        return infos


class PermFractal(MixedFractal):

    """PermFractal

    Concrete Fractal for permutations

    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Section

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Section(a, section=3)
    >>> print(sp, sp.lower, sp.upper, sp.section)
    Section(0,-1,0) [0. 0.] [1. 1.] 3
    >>> print(sp.get_id())
    (0, -1, 0)
    >>> p = sp.random_point()
    >>> print(p)
    [0.3977808911066778, 0.9684061309949581]
    >>> p = sp.random_point(2)
    >>> d = sp.distance(p[0], p[1])
    >>> print(f"{d:.2f}")
    0.41
    >>> children = sp.create_children()
    >>> for c in children:
    ...     print(f"id:{c.get_id()}:[{c.lower},{c.upper}]")
    id:(1, 0, 0):[[0. 0.],[0.33333333 1.        ]]
    id:(1, 0, 1):[[0.33333333 0.        ],[0.66666667 1.        ]]
    id:(1, 0, 2):[[0.66666667 0.        ],[1. 1.]]
    """

    def __init__(
        self,
        variables: PermutationVar,
        measurement: Optional[Measurement] = None,
    ):
        """__init__

        Parameters
        ----------
        variables : PermutationVar
            Determines the bounds of the search space.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        """

        super().__init__(variables, measurement)
        self.base = np.arange(self.variables.n, dtype=int)
        self.fixed_idx = 1

    @property
    def variables(self) -> PermutationVar:
        return self._variables

    @variables.setter
    def variables(self, value: PermutationVar):
        if value:
            self._variables = value
        else:
            raise InitializationError(
                f"In PermFractal, variables must be defined within an PermutationVar. Got {type(value)}"
            )

    def create_children(self):
        """create_children()

        Partition function.
        """

        fixed_idx = self.fixed_idx
        k = len(self.variables) - fixed_idx
        new_fixed_idx = fixed_idx + 1

        if k > 0:
            children = super().create_children(k)
            for idx, child in enumerate(children):
                child.base = self.base.copy()
                child.base[fixed_idx:] = np.roll(child.base[fixed_idx:], -idx)
                child.fixed_idx = new_fixed_idx
                child.score = self.score

            return children
        else:
            logger.warning(
                f"In PermFractal, cannot create children. Maximum depth reached. The theoretical depth = {self.variables.n})."
            )
            return []

    def random_point(self, size: Optional[int] = None) -> Union[list, List[list]]:
        if size:
            res = np.tile(self.base, (size, 1))
            for v in res:
                v[self.fixed_idx :] = np.random.permutation(v[self.fixed_idx :])
            return res.tolist()
        else:
            return np.random.permutation(self.base[self.fixed_idx :]).tolist()

    def _modify(self, fixed_v, level, father, f_id, c_id, score, measure):
        super()._modify(level, father, f_id, c_id, score, measure)
        self.base = np.arange(self.variables.n)
        self.base[: len(fixed_v)] = self.base[fixed_v]
        self.base[fixed_v] = np.arange(len(fixed_v))
        self.fixed_idx = len(fixed_v)

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update({"fixed_v": self.base[: self.fixed_idx]})
        return infos
