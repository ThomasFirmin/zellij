# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.search_space import Fractal, MixedFractal

from typing import Optional, Callable, Union, List, TYPE_CHECKING

from zellij.core.variables import ArrayVar
from zellij.strategies.tools.measurements import Measurement

if TYPE_CHECKING:
    from zellij.core.variables import ArrayVar, PermutationVar
    from zellij.strategies.tools.measurements import Measurement

from itertools import product
from scipy.stats import qmc
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
        level: int = 0,
        score: float = float("inf"),
        save_points=False,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
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
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.
        """
        super(Hypercube, self).__init__(
            variables,
            measurement=measurement,
            level=level,
            score=score,
            save_points=save_points,
            lower=lower,
            upper=upper,
        )

    def create_children(self):
        """create_children(self)

        Partition function.
        """

        center = (self.lower + self.upper) / 2
        vertices = np.array(list(product(*zip(self.lower, self.upper))))
        lower = np.minimum(vertices, center)
        upper = np.maximum(vertices, center)

        children = [
            super(Hypercube, self).create_child(lower=l, upper=u)
            for l, u in zip(lower, upper)
        ]
        return children

    def _modify(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        level: int,
        score: float,
        measure: float,
        best_loss: float,
        best_loss_parent: float,
    ):
        super()._modify(level, score, measure, best_loss, best_loss_parent)
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

    def __init__(
        self,
        variables,
        measurement: Optional[Measurement] = None,
        level: int = 0,
        score: float = float("inf"),
        save_points=False,
        center: Optional[np.ndarray] = None,
        radius: Optional[np.ndarray] = None,
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
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.

        """

        super(Hypersphere, self).__init__(
            variables, measurement, level=level, score=score, save_points=save_points
        )
        if center:
            self.center = center
        else:
            self.center = np.full(self.size, 0.5)

        if radius:
            self.radius = radius
        else:
            self.radius = 0.5

    # Return a random point from the search space
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

        r_p = self.radius / (1 + np.sqrt(2))

        centers = np.tile(self.center, (2 * self.size, 1))
        centers[: len(self.center) :].ravel()[:: len(self.center) + 1] += (
            self.radius - r_p
        )
        centers[len(self.center) : :].ravel()[:: len(self.center) + 1] -= (
            self.radius - r_p
        )

        children = [
            super(Hypersphere, self).create_child(center=c, radius=r_p)
            for c in zip(centers)
        ]

        return children

    def _modify(
        self,
        center: np.ndarray,
        radius: float,
        level: int,
        score: float,
        measure: float,
        best_loss: float,
        best_loss_parent: float,
    ):
        super()._modify(level, score, measure, best_loss, best_loss_parent)
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
        level: int = 0,
        score: float = float("inf"),
        save_points=False,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
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
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.
        """

        super(Section, self).__init__(
            variables,
            measurement,
            level=level,
            score=score,
            save_points=save_points,
            lower=lower,
            upper=upper,
        )
        self.section = section
        self.is_middle = False
        self.evaluated = False

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

        longest = self.level % self.size
        width = self.upper[longest] - self.lower[longest]

        step = width / self.section

        lowers = np.tile(self.lower, (self.section, 1))
        uppers = np.tile(self.upper, (self.section, 1))

        add = np.arange(0, self.section + 1) * step
        uppers[:, longest] = lowers[:, longest] + add[1:]
        lowers[:, longest] = lowers[:, longest] + add[:-1]

        children = [
            super(Section, self).create_child(lower=l, upper=u, section=self.section)
            for l, u in zip(lowers, uppers)
        ]

        if self.section % 2 != 0:
            children[int(self.section / 2)].is_middle = True

        return children

    def _modify(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        level: int,
        is_middle: bool,
        score: float,
        measure: float,
        best_loss: float,
        best_loss_parent: float,
    ):
        super()._modify(level, score, measure, best_loss, best_loss_parent)
        self.upper, self.lower = upper, lower
        self.is_middle = is_middle

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update(
            {"upper": self.upper, "lower": self.lower, "is_middle": self.is_middle}
        )
        return infos


class NMSOSection(Fractal):

    def __init__(
        self,
        variables: ArrayVar,
        measurement: Optional[Measurement] = None,
        section: int = 2,
        level: int = 0,
        score: float = float("inf"),
        save_points=False,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        df: float = -float("inf"),
        dx: float = 0,
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
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.
        """

        super(NMSOSection, self).__init__(
            variables,
            measurement,
            level=level,
            score=score,
            save_points=save_points,
            lower=lower,
            upper=upper,
        )
        self.section = section

        self.left = False
        self.middle = False
        self.right = False
        self.is_middle = False
        self.df = df
        self.dx = dx
        self.visited = float("inf")
        self.evaluated = False

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

        longest = self.level % self.size
        width = self.upper[longest] - self.lower[longest]

        step = width / self.section

        lowers = np.tile(self.lower, (self.section, 1))
        uppers = np.tile(self.upper, (self.section, 1))

        add = np.arange(0, self.section + 1) * step
        uppers[:, longest] = lowers[:, longest] + add[1:]
        lowers[:, longest] = lowers[:, longest] + add[:-1]

        middle = int(np.floor(self.section / 2))
        left = max(0, middle - 1)
        right = middle + 1

        children = [
            super(NMSOSection, self).create_child(
                lower=l, upper=u, section=self.section, dx=step, df=self.df
            )
            for l, u in zip(lowers, uppers)
        ]

        if self.section % 2 != 0:
            children[middle].is_middle = True
        children[middle].middle = True
        children[left].left = True
        children[right].right = True

        return children

    def _modify(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        level: int,
        left: bool,
        middle: bool,
        is_middle: bool,
        right: bool,
        df: float,
        dx: float,
        score: float,
        measure: float,
        best_loss: float,
        best_loss_parent: float,
    ):
        super()._modify(level, score, measure, best_loss, best_loss_parent)
        self.upper, self.lower = upper, lower
        self.left = left
        self.middle = middle
        self.is_middle = is_middle
        self.right = right
        self.df = df
        self.dx = dx

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update(
            {
                "upper": self.upper,
                "lower": self.lower,
                "left": self.left,
                "middle": self.middle,
                "is_middle": self.is_middle,
                "right": self.right,
                "df": self.df,
                "d": self.dx,
            }
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

    def __init__(
        self,
        variables: ArrayVar,
        measurement: Optional[Measurement] = None,
        level: int = 0,
        score: float = float("inf"),
        save_points=True,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
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
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.
        """

        super(Direct, self).__init__(
            variables,
            measurement,
            level=level,
            score=score,
            save_points=save_points,
            lower=lower,
            upper=upper,
        )
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
        if section_length > 1e-16:
            if self.level == 0:
                scores = np.reshape(self.losses[1:], (-1, 2))
            else:
                scores = np.reshape(self.losses, (-1, 2))

            children = []

            scores_dim = scores.min(axis=1)
            argsort = np.argsort(scores_dim)

            current_low = np.copy(self.lower)
            current_up = np.copy(self.upper)

            for arg in argsort:
                dim = self.set_i[arg]
                child1 = super(Direct, self).create_child(
                    lower=np.copy(current_low),
                    upper=np.copy(current_up),
                )
                child1.upper[dim] -= 2 * section_length
                child1.score = scores[arg][0]
                child1._update_attributes()

                child2 = super(Direct, self).create_child(
                    lower=np.copy(current_low),
                    upper=np.copy(current_up),
                )
                child2.lower[dim] += 2 * section_length
                child2.score = scores[arg][1]
                child2._update_attributes()

                children.append(child1)
                children.append(child2)

                # Middle
                current_low[dim] += section_length
                current_up[dim] -= section_length

            child3 = super(Direct, self).create_child(
                lower=np.copy(current_low),
                upper=np.copy(current_up),
            )
            child3.score = self.score
            child3._update_attributes()
            children.append(child3)

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
        score: float,
        measure: float,
        best_loss: float,
        best_loss_parent: float,
    ):
        super()._modify(level, score, measure, best_loss, best_loss_parent)
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


class LatinHypercubeUCB(Fractal):
    """LatinHypercubeUCB

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
        max_depth: int,
        measurement: Optional[Measurement] = None,
        grid_size: int = 2,
        level: int = 0,
        score: float = float("inf"),
        save_points=False,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        length: float = 1,
        strength=1,
        descending=True,
        sampler=None,
    ):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar` and all must have converter.
        grid_size : int, default=1
            Size of the grid for LHS. :code:`self.size * grid_size` hypercubes will
            be sampled. Given value must be :code:`grid_size > 1`.
            Can be a callable of the form Callable[[int], int].
            A callable taking an int (level) and returning an int (grid).
        symmetric : boolean, default=True
            Apply a symmetrization on the LatinHypercube distribution.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.
        """

        super(LatinHypercubeUCB, self).__init__(
            variables,
            measurement,
            level=level,
            score=score,
            save_points=save_points,
            lower=lower,
            upper=upper,
        )

        self.max_depth = max_depth
        self.strength = strength
        self.grid_size = grid_size
        self.length = length

        self.mean = float("inf")
        self.var = float("inf")

        self.best_score_parent = float("inf")

        if sampler is None:
            self.sampler = qmc.LatinHypercube(
                d=self.size, scramble=False, strength=self.strength
            )
        else:
            self.sampler = sampler

        self.descending = descending

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: int):
        if value <= 1:
            raise InitializationError(f"grid_size must be >1, got {value}")
        else:
            self._grid_size = value

    def create_children(self):
        """create_children(self)

        Partition function.
        """
        if self.descending:
            children = self._create_descending()
        else:
            children = self._create_ascending()

        return children

    def _create_descending(self):

        length = self.length / self.grid_size

        sample = self.sampler.random(n=self.grid_size)
        sample = qmc.scale(sample, self.lower, self.upper)

        l = length / 2
        lowers = np.clip(sample - l, 0, 1)
        uppers = np.clip(sample + l, 0, 1)

        descending = (self.level + 1) < (self.max_depth - 1)

        children = [
            super(LatinHypercubeUCB, self).create_child(
                length=length,
                lower=l,
                upper=u,
                max_depth=self.max_depth,
                grid_size=self.grid_size,
                descending=descending,
                strength=self.strength,
                sampler=self.sampler,
            )
            for l, u in zip(lowers, uppers)
        ]

        return children

    def _create_ascending(self):
        previous_level = self.level - 2
        next_level = self.level - 1

        if previous_level > 0:

            descending = next_level == 1

            previous_length = 1 / self.grid_size**previous_level

            center = (self.upper + self.lower) / 2
            l = previous_length / 2
            lower = np.clip(center - l, 0, 1)
            upper = np.clip(center + l, 0, 1)

            sample = self.sampler.random(n=self.grid_size)
            sample = qmc.scale(sample, lower, upper)

            next_length = previous_length / self.grid_size
            l = next_length / 2
            lowers = np.clip(sample - l, 0, 1)
            uppers = np.clip(sample + l, 0, 1)

            children = [
                super(LatinHypercubeUCB, self).create_child(
                    length=next_length,
                    lower=l,
                    upper=u,
                    max_depth=self.max_depth,
                    grid_size=self.grid_size,
                    descending=descending,
                    level=next_level,
                    sampler=self.sampler,
                )
                for l, u in zip(lowers, uppers)
            ]

            return children
        else:
            return []

    def _modify(
        self,
        upper: np.ndarray,
        lower: np.ndarray,
        level: int,
        score: float,
        measure: float,
        best_loss: float,
        best_loss_parent: float,
    ):
        super()._modify(level, score, measure, best_loss, best_loss_parent)
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
        level: int = 0,
        score: float = float("inf"),
        save_points=False,
    ):
        """__init__

        Parameters
        ----------
        variables : PermutationVar
            Determines the bounds of the search space.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        level : int, optional
            Set the default level of a fractal.
        score : float, optional
            Set the default score of a fractal.
        """

        super().__init__(
            variables, measurement, level=level, score=score, save_points=save_points
        )
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

    def _modify(self, fixed_v, level, score, measure):
        super()._modify(level, score, measure)
        self.base = np.arange(self.variables.n)
        self.base[: len(fixed_v)] = self.base[fixed_v]
        self.base[fixed_v] = np.arange(len(fixed_v))
        self.fixed_idx = len(fixed_v)

    def _essential_info(self):
        infos = super()._essential_info()
        infos.update({"fixed_v": self.base[: self.fixed_idx]})
        return infos
