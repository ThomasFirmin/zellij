# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.errors import InitializationError, DimensionalityError
from zellij.core.addons import (
    IntNeighborhood,
    FloatNeighborhood,
    CatNeighborhood,
    ArrayNeighborhood,
    PermutationNeighborhood,
)

from typing import Optional, Union, List

import numpy as np

import logging

logger = logging.getLogger("zellij.neighborhoods")


class IntInterval(IntNeighborhood):
    """IntInterval

    :ref:`varadd`, used to determine the neighbor of an :code:`IntVar`.
    Draw a random point in :math:`x \pm neighborhood`.

    Parameters
    ----------
    neighborhood : int, default=None
        :math:`x \pm neighborhood`

    Attributes
    ----------
    neighborhood

    Examples
    --------
    >>> from zellij.core import IntVar
    >>> from zellij.utils import IntInterval

    >>> a = IntVar("i1", 0, 1000, neighborhood=IntInterval(10))
    >>> p = a.random()
    >>> n = a.neighborhood(p)
    >>> print(f"{p} is close to {n}")
    582 is close to 580
    """

    @property
    def neighborhood(self) -> int:
        return self._neighborhood

    @neighborhood.setter
    def neighborhood(self, neighborhood: int):
        if isinstance(neighborhood, int) and neighborhood > 0:
            self._neighborhood = neighborhood
        else:
            raise InitializationError(
                f"`neighborhood` must be a positive int, got {neighborhood}"
            )

    def __call__(self, value: int, size: Optional[int] = None) -> Union[int, List[int]]:
        """__call__

        Parameters
        ----------
        value : object
            Feature of :code:`CatVar`
        size : Optional[int], optional
            If size is None, then returns a single value. Else, returns a list of values.

        Returns
        -------
        v : {int, list[int]}
            Return a single feature or list of feature.
        """

        upper = np.min([value + self.neighborhood + 1, self.target.upper])
        lower = np.max([value - self.neighborhood, self.target.lower])
        if size:
            res = [0] * size
            for i in range(size):
                v = np.random.randint(lower, upper)
                while v == value:
                    v = np.random.randint(lower, upper)
                res[i] = v
            return res
        else:
            v = np.random.randint(lower, upper)
            while v == value:
                v = np.random.randint(lower, upper)

            return int(v)


class FloatInterval(FloatNeighborhood):
    """FloatInterval

    :ref:`varadd`, used to determine the neighbor of a FloatVar.
    Draw a random point in :math:`x \pm neighborhood`.

    Parameters
    ----------
    neighborhood : float
        :math:`x \pm neighborhood`

    Attributes
    ----------
    neighborhood

    Examples
    --------
    >>> from zellij.core import FloatVar
    >>> from zellij.utils import FloatInterval

    >>> a = FloatVar("f1", 0, 1000, neighborhood=FloatInterval(10))
    >>> p = a.random()
    >>> n = a.neighborhood(p)
    >>> print(f"{p:.2f} is close to {n:.2f}")
    133.47 is close to 141.24

    """

    @property
    def neighborhood(self) -> Union[float, int]:
        return self._neighborhood

    @neighborhood.setter
    def neighborhood(self, neighborhood: Union[float, int]):
        if isinstance(neighborhood, (int, float)) and neighborhood > 0:
            self._neighborhood = neighborhood
        else:
            raise InitializationError(
                f"`neighborhood` must be a positive float or int, got {neighborhood}"
            )

    def __call__(
        self, value: Union[float, int], size: Optional[int] = None
    ) -> Union[float, List[float]]:
        """__call__

        Parameters
        ----------
        value : object
            Feature of :code:`CatVar`
        size : Optional[int], optional
            If size is None, then returns a single value. Else, returns a list of values.

        Returns
        -------
        v : {float, list[float]}
            Return a single feature or list of feature.
        """
        upper = np.min([value + self.neighborhood, self.target.upper])
        lower = np.max([value - self.neighborhood, self.target.lower])

        if size:
            res = [0.0] * size
            for i in range(size):
                v = np.random.uniform(lower, upper)
                while v == value:
                    v = np.random.uniform(lower, upper)
                res[i] = float(v)
            return res
        else:
            v = np.random.uniform(lower, upper)
            while v == value:
                v = np.random.uniform(lower, upper)

            return float(v)


class CatRandom(CatNeighborhood):
    """CatRandom

    :ref:`varadd`, used to determine the neighbor of a CatVar.
    Draw a random feature in CatVar.

    Parameters
    ----------
    neighborhood : int, optional
        Undefined, for CatVar it draws a random feature.

    Attributes
    ----------
    neighborhood

    Examples
    --------
    >>> from zellij.core import CatVar
    >>> from zellij.utils import CatRandom

    >>> a = CatVar("c1", ["a","b","c","d"], neighborhood=CatRandom())
    >>> p = a.random()
    >>> n = a.neighborhood(p)
    >>> print(f"{p} is close to {n}")
    a is close to d

    """

    @property
    def neighborhood(self):
        return self._neighborhood

    @neighborhood.setter
    def neighborhood(self, neighborhood=None):
        self._neighborhood = neighborhood

    def __call__(
        self, value, size: Optional[int] = None
    ) -> Union[object, List[object]]:
        """__call__

        Parameters
        ----------
        value : object
            Feature of :code:`CatVar`
        size : Optional[int], optional
            If size is None, then returns a single value. Else, returns a list of values.

        Returns
        -------
        v : {object, list[object]}
            Return a single feature or list of feature.
        """
        if size:
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


class ArrayDefaultN(ArrayNeighborhood):
    """ArrayDefaultN

    :ref:`varadd`, used to determine the neighbor of a ArrayVar.
    Draw a random feature in ArrayVar.

    Parameters
    ----------
    neighborhood : int, optional
        Undefined, for ArrayVar.

    Attributes
    ----------
    neighborhood

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, IntVar, CatVar
    >>> from zellij.utils import ArrayDefaultN, FloatInterval, IntInterval, CatRandom

    >>> a = ArrayVar(
    ...     IntVar("i1", 0, 100, neighborhood=IntInterval(10)),
    ...     FloatVar("f1", -100, 0, neighborhood=FloatInterval(10)),
    ...     CatVar("c1", ["Hello", 87, 2.56], neighborhood=CatRandom()),
    ...     neighborhood=ArrayDefaultN(),
    ... )

    >>> p = a.random()
    >>> n = a.neighborhood(p)
    >>> print(f"{p}\nis close to\n{n}")
    [86, -38.23310264757463, 2.56]
    is close to
    [96, -42.305187546793135, 87]

    """

    @property
    def neighborhood(self):
        return self._neighborhood

    @neighborhood.setter
    def neighborhood(self, neighborhood):
        self._neighborhood = neighborhood

    def __call__(
        self, value: list, size: Optional[int] = None
    ) -> Union[list, List[list]]:
        """__call__

        Parameters
        ----------
        value : List
            List made of values from :code:`variables` from :code:`ArrayVar`.
        size : Optional[int], optional
            If size is None, then returns a single value. Else, returns a list of values.

        Returns
        -------
        v : {object, list[object]}
            Return a single feature or list of feature.
        """
        if len(value) == len(self.target):
            res = []
            if size:
                for _ in range(size):
                    inter = []
                    for var, v in zip(self.target.variables, value):
                        inter.append(var.neighborhood(v))  # type: ignore
                    res.append(inter)
            else:
                for var, v in zip(self.target.variables, value):
                    res.append(var.neighborhood(v))  # type: ignore
            return res
        else:
            raise DimensionalityError(
                f"In ArrayDefaultN, a value does not have the same length as the dimensionality of the target."
            )


class PermutationRandom(PermutationNeighborhood):
    """PermutationRandom

    :ref:`varadd`, used to determine the neighbor of a PermutationVar.
    Draw a random neighbor in PermutationVar.
    |!| Does not ensure that two succesive neighbors are different.

    Parameters
    ----------
    neighborhood : int, optional
        Undefined, for PermutationRandom it draws a random feature.

    Attributes
    ----------
    neighborhood

    Examples
    --------
    >>> from zellij.core import CatVar
    >>> from zellij.utils import CatRandom

    >>> a = CatVar("c1", ["a","b","c","d"], neighborhood=CatRandom())
    >>> p = a.random()
    >>> n = a.neighborhood(p)
    >>> print(f"{p} is close to {n}")
    a is close to d

    """

    @property
    def neighborhood(self):
        return self._neighborhood

    @neighborhood.setter
    def neighborhood(self, neighborhood=None):
        self._neighborhood = neighborhood

    def __call__(
        self, value, size: Optional[int] = None
    ) -> Union[object, List[object]]:
        """__call__

        Parameters
        ----------
        value : list[int]
            A permutation. A list of integer, list[int].
        size : Optional[int], optional
            If size is None, then returns a single value. Else, returns a list of values.

        Returns
        -------
        v : {object, list[object]}
            Return a single feature or list of feature.
        """

        if size:
            elem = np.arange(size)[:, None]
            res = np.tile(value, (size, 1))
            idx = np.array(
                [
                    np.random.choice(len(value), size=2, replace=False)
                    for _ in range(size)
                ]
            )
            idxT = idx.copy()[:, [1, 0]]
            res[elem, idx] = res[elem, idxT]
            return res.tolist()
        else:
            res = np.array(value[:])
            idx = np.random.choice(len(value), size=(1, 2), replace=False)
            idxT = idx.copy()[:, [1, 0]]
            res[idx] = res[idxT]
            return res.tolist()
