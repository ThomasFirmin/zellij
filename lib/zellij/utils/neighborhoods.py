# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-06T12:07:22+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:54:24+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.addons import VarNeighborhood, Neighborhood
from zellij.core.variables import (
    FloatVar,
    IntVar,
    CatVar,
    Constant,
    ArrayVar,
)
import numpy as np
import copy

import logging

logger = logging.getLogger("zellij.neighborhoods")


class ArrayInterval(VarNeighborhood):
    """ArrayInterval

    :ref:`spadd`, used to determine the neighbor of an ArrayVar.
    neighbor kwarg must be implemented for all :ref:`var` of the ArrayVar.

    Parameters
    ----------
    variable : ArrayVar, default=None
        Targeted :ref:`var`.
    neighborhood : list, default=None
        Not yet implemented

    Attributes
    ----------
    neighborhood

    """

    def __init__(self, variable=None, neighborhood=None):
        super(ArrayInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        variables = np.random.choice(self.target.values, size=size)
        res = []

        for v in variables:
            inter = copy.deepcopy(value)
            inter[v._idx] = v.neighbor(value[v._idx])
            res.append(inter)

        return res

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood:
            for var, neig in zip(self.target.values, neighborhood):
                var.neighborhood = neig

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, ArrayVar) or variable == None, logger.error(
            f"Target object must be an `ArrayVar` for {self.__class__.__name__},\
             got {variable}"
        )

        self._target = variable

        if variable != None:
            assert all(
                hasattr(v, "neighbor") for v in self.target.values
            ), logger.error(
                f"To use `ArrayInterval`, values in `ArrayVar` must have a `neighbor` method. Use `neighbor` kwarg when defining a variable"
            )


class BlockInterval(VarNeighborhood):
    """BlockInterval

    :ref:`spadd`, used to determine the neighbor of an BlockInterval.
    neighbor kwarg must be implemented for all :ref:`var` of the BlockInterval.

    Not yet implemented...

    """

    def __call__(self, value, size=1):
        raise NotImplementedError(
            f"{self.__class__.__name__}\
        neighborhood is not yet implemented"
        )


class DynamicBlockInterval(VarNeighborhood):
    """BlockInterval

    :ref:`spadd`, used to determine the neighbor of an BlockInterval.
    neighbor kwarg must be implemented for all :ref:`var` of the BlockInterval.

    Not yet implemented...

    """

    def __call__(self, value, size=1):
        raise NotImplementedError(
            f"{self.__class__.__name__}\
        neighborhood is not yet implemented"
        )


class FloatInterval(VarNeighborhood):
    """FloatInterval

    :ref:`varadd`, used to determine the neighbor of a FloatVar.
    Draw a random point in :math:`x \pm neighborhood`.

    Parameters
    ----------
    variable : FloatVar, default=None
        Targeted :ref:`var`.
    neighborhood : float, default=None
        :math:`x \pm neighborhood`

    Attributes
    ----------
    neighborhood

    """

    def __call__(self, value, size=1):
        upper = np.min([value + self.neighborhood, self.target.up_bound])
        lower = np.max([value - self.neighborhood, self.target.low_bound])

        if size > 1:
            res = []
            for _ in range(size):
                v = np.random.uniform(lower, upper)
                while v == value:
                    v = np.random.uniform(lower, upper)
                res.append(float(v))
            return res
        else:
            v = np.random.uniform(lower, upper)
            while v == value:
                v = np.random.uniform(lower, upper)

            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert isinstance(neighborhood, int) or isinstance(
            neighborhood, float
        ), logger.error(
            f"`neighborhood` must be a float or an int, for `FloatInterval`,\
            got{neighborhood}"
        )

        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, FloatVar) or variable == None, logger.error(
            f"Target object must be a `FloatVar` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class IntInterval(VarNeighborhood):
    """IntInterval

    :ref:`varadd`, used to determine the neighbor of an IntVar.
    Draw a random point in :math:`x \pm neighborhood`.

    Parameters
    ----------
    variable : IntVar, default=None
        Targeted :ref:`var`.
    neighborhood : int, default=None
        :math:`x \pm neighborhood`

    Attributes
    ----------
    neighborhood

    """

    def __call__(self, value, size=1):

        upper = np.min([value + self.neighborhood + 1, self.target.up_bound])
        lower = np.max([value - self.neighborhood, self.target.low_bound])

        if size > 1:
            res = []
            for _ in range(size):
                v = np.random.randint(lower, upper)
                while v == value:
                    v = np.random.randint(lower, upper)
                res.append(int(v))
            return res
        else:
            v = np.random.randint(lower, upper)
            while v == value:
                v = np.random.randint(lower, upper)

            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert isinstance(neighborhood, int) or isinstance(
            neighborhood, float
        ), logger.error(
            f"`neighborhood` must be an int, for `IntInterval`,\
            got{neighborhood}"
        )

        self._neighborhood = neighborhood

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, IntVar) or variable == None, logger.error(
            f"Target object must be a `IntInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class CatInterval(VarNeighborhood):
    """CatInterval

    :ref:`varadd`, used to determine the neighbor of a CatVar.
    Draw a random feature in CatVar.

    Parameters
    ----------
    variable : FlaotVar, default=None
        Targeted :ref:`var`.
    neighborhood : int, default=None
        Undefined, for CatVar it draws a random feature.

    Attributes
    ----------
    neighborhood

    """

    def __init__(self, variable=None, neighborhood=None):
        super(CatInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        if size > 1:
            res = []
            for _ in range(size):
                v = self.target.random()
                while v == value:
                    v = self.target.random()
                res.append(v)
            return res
        else:
            v = self.target.random()
            while v == value:
                v = self.target.random()
            return v

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood != None:
            logger.warning(
                f"`neighborhood`= {neighborhood} is useless for \
            {self.__class__.__name__}, it will be replaced by None"
            )

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, CatVar) or variable == None, logger.error(
            f"Target object must be a `CatInterval` for {self.__class__.__name__},\
             got {variable}"
        )
        self._target = variable


class ConstantInterval(VarNeighborhood):
    """ConstantInterval

    :ref:`varadd`, used to determine the neighbor of a Constant.
    Do nothing. Return the constant.

    Parameters
    ----------
    variable : Constant, default=None
        Targeted :ref:`var`.
    neighborhood : int, default=None
        Not implemented.

    Attributes
    ----------
    neighborhood

    """

    def __init__(self, variable=None, neighborhood=None):
        super(ConstantInterval, self).__init__(variable)
        self.neighborhood = neighborhood

    def __call__(self, value, size=1):
        logger.warning("Calling `neighbor` of a constant is useless")
        if size > 1:
            return [self.target.value for _ in range(size)]
        else:
            return self.target.value

    @VarNeighborhood.neighborhood.setter
    def neighborhood(self, neighborhood=None):
        if neighborhood != None:
            logger.warning(
                f"`neighborhood`= {neighborhood} is useless for \
            {self.__class__.__name__}, it will be replaced by None"
            )

        self._neighborhood = None

    @VarNeighborhood.target.setter
    def target(self, variable):

        assert isinstance(variable, Constant) or variable == None, logger.error(
            f"Target object must be a `ConstantInterval` for {self.__class__.__name__}\
            , got {variable}"
        )
        self._target = variable


class Intervals(Neighborhood):
    """Intervals

    :ref:`spadd`, used to determine the neighbor of a given point.
    All :ref:`var` of the :ref:`sp` must have the neighbor addon implemented.

    Parameters
    ----------
    variable : :ref:`sp`, default=None
        Targeted :ref:`sp`.
    neighborhood : list, default=None
        If a list of the shape of the values from the :ref:`sp`.
        Modify the neighborhood attribute of all :ref:`varadd` of type
        VarNeighborhood, for each :ref:`var`.


    Attributes
    ----------
    neighborhood

    """

    def __init__(self, search_space=None, neighborhood=None):
        super(Intervals, self).__init__(search_space, neighborhood)

    @Neighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        if neighborhood:
            for var, neig in zip(self.target.values, neighborhood):
                var.neighbor.neighborhood = neig

        self._neighborhood = None

    @Neighborhood.target.setter
    def target(self, object):
        self._target = object
        if object:
            assert hasattr(self.target.values, "neighbor"), logger.error(
                f"To use `Intervals`, values in Searchspace must have a `neighbor` method. Use `neighbor` kwarg when defining a variable"
            )

    def __call__(self, point, size=1):

        """__call__(point, size=1)

        Draw a neighbor of a solution, according to the :ref:`var` neighbor
        function.

        Parameters
        ----------

        point : list
            Initial point.
        size : int, default=1
            Draw <size> neighbors of <point>.

        Returns
        -------

        out : list
            List of neighbors of <point>.

        """
        attribute = self.target.random_attribute(size=size, exclude=Constant)

        points = []

        for att in attribute:

            inter = copy.deepcopy(point)
            inter[att._idx] = att.neighbor(point[att._idx])
            points.append(inter)

        return points
