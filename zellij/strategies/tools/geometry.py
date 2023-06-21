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
            child._update_measure()

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
    sigma : Direct_size
        Sigma function. Determines a measurement of the size of a subspace.

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

    def __init__(self, variables, loss, measure=SigmaInf, **kwargs):
        """__init__(values, loss, max_calls, scoring=Min(), force_convert=False, sigma=SigmaInf(), **kwargs,)

        Parameters
        ----------
        values : Variables
            Defines the decision variables. See :ref:`var`.
        loss : LossFunc
            Defines the loss function. See :ref:`lf`.
        heuristic : Heuristic, default=Min()
            Function that defines how promising a space is according to sampled
            points. It is similar to the acquisition function in BO.
        force_convert : bool, default=False
            Force the convertion of all :ref:`var`, even continuous ones.
            It allows for example, to consider the unit hypercube, instead of
            the defined space.
        sigma : Direct_size, default, Sigma2()
            Sigma function. Determines a measurement of the size of a subspace.

        """

        super(Direct, self).__init__(variables, loss, measure, **kwargs)

        self.lower = np.zeros(self.size)
        self.upper = np.ones(self.size)
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
        if section_length > 1e-15:
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
        upper,
        lower,
        width,
        set_i,
        level,
        father,
        f_id,
        c_id,
        score,
        measure,
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
