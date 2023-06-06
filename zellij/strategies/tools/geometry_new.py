# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T12:43:39+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
import abc
from itertools import product
from abc import ABC, abstractmethod

from zellij.core.search_space import Fractal
from zellij.strategies.tools.measurements import SigmaInf

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
    """

    def __init__(
        self,
        variables,
        loss,
        measure=None,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        measure : Measurement, default=None
            Defines the measure of a fractal.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * converter : Converter
                * Will be called when converting a solution to another space is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)
        """

        super(Hypercube, self).__init__(variables, loss, measure, **kwargs)

        self.lower = np.zeros(self.size)
        self.upper = np.ones(self.size)

    def create_children(self):
        """create_children(self)

        Partition function.
        """

        children = super().create_children(2**self.size, **self._all_addons)

        center = (self.upper - self.lower) / 2
        vertices = np.array(list(product(*zip(self.lower, self.upper))))
        lower = np.minimum(vertices, center)
        upper = np.maximum(vertices, center)

        for child, l, u in zip(children, lower, upper):
            child.lower = l
            child.upper = u

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
    """

    def __init__(self, variables, loss, measure=None, **kwargs):
        """__init__(self,variables,loss,father="root",level=0,id=0,children=[],score=None,**kwargs)

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        measure : Measurement, default=None
            Defines the measure of a fractal.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * converter : Converter
                * Will be called when converting a solution to another space is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)
        """

        super(Hypersphere, self).__init__(variables, loss, measure, **kwargs)

        self.center = np.full(self.size, 0.5)
        self.radius = 0.5

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

        return children

    def _modify(self, center, radius, level, father, f_id, c_id, score, measure):
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
    """

    def __init__(
        self,
        variables,
        loss,
        measure=None,
        section=2,
        **kwargs,
    ):
        """__init__(self,variables,loss,father="root",level=0,id=0,children=[],score=None,**kwargs)

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
            of `FloatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        measure : Measurement, default=None
            Defines the measure of a fractal.

        section : int, default=2
            Defines in how many equal sections the space should be decompose.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * converter : Converter
                * Will be called when converting a solution to another space is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)

        """

        super(Section, self).__init__(variables, loss, measure, **kwargs)

        assert section > 1, logger.error(
            f"{section}-Section is not possible, section must be > 1"
        )

        self.lower = np.zeros(self.size)
        self.upper = np.ones(self.size)
        self.section = section

        self.is_middle = False

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
