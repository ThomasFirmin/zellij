# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from zellij.core.errors import DimensionalityError
from zellij.core.addons import Distance
from zellij.core.variables import FloatVar, IntVar, CatVar
from scipy.spatial import distance

from typing import List, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.variables import FloatVar, IntVar

import numpy as np
import logging

logger = logging.getLogger("zellij.distances")


class Euclidean(Distance):
    """Euclidean distance

    Compute the Euclidean distance between two points.
    More info on `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean>`__

    See also
    --------
    :ref:`dist`: Distance addon
    :ref:`sp`: Searchspace

    """

    def __call__(
        self, point_a: List[Union[float, int]], point_b: List[Union[float, int]]
    ) -> float:
        return distance.euclidean(point_a, point_b, self.weights)


class Manhattan(Distance):
    """Manhattan distance

    Compute the Manhattan distance between two points.
    More info on `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html#scipy.spatial.distance.cityblock>`__

    See also
    --------
    :ref:`dist`: Distance addon
    :ref:`sp`: Searchspace

    """

    def __call__(
        self, point_a: List[Union[float, int]], point_b: List[Union[float, int]]
    ) -> float:
        return distance.cityblock(point_a, point_b, self.weights)


######################
# MIXED
######################


class _Float_int_dist:
    def __init__(self, var: Union[FloatVar, IntVar]):
        self.var = var

    def __call__(self, x, y):
        return np.abs(x - y) / (self.var.upper - self.var.lower)


def cat_dist(x, y):
    return 1 if x == y else 0


def constant_dist(x, y):
    return 0


class Mixed(Distance):
    """Mixed distance

    Compute a distance between two mixed points, using following equations:

    .. math::

        \\smash{
        \\begin{cases}
        \\delta_{i,j}^{(n)}=\\frac{|x_{i,n}-x_{j,n}|}{max_h(x_{h,n})-min_h(x_{h,n})}, \\quad \\text{if: $x_{h,n}$ is continuous or discrete}\\\\
        \\begin{cases}
        \\delta_{i,j}^{(n)}=0, \\quad \\text{if, $x_{i,n}=x_{j,n}$}\\\\
        \\delta_{i,j}^{(n)}=1, \\quad \\text{otherwise}
        \\end{cases} , \\quad \\text{if: $x_{h,n}$ is categorical}\\\\
        d(x_i,x_j)=\\frac{\\sum_{n=1}^{p}(\\delta_{i,j}^{(n)})^2}{\\sum_{n=1}^{p}\\delta_{i,j}^{(n)}}\\\\
        \\end{cases}}

    See also
    --------
    :ref:`dist`: Distance addon
    :ref:`sp`: Searchspace

    """

    def __init__(self, weights: Optional[List[float]] = None):
        super(Mixed, self).__init__(weights)
        self.operations = []

    def __call__(self, point_a: list, point_b: list) -> float:
        if len(point_a) != len(point_b):
            raise DimensionalityError(
                f"Dimensionality of both point must be equal. Got {len(point_a)}=={len(point_b)}"
            )
        num = 0
        denum = 0

        for x, y, op, w in zip(point_a, point_b, self.operations, self.weights):
            res = op(x, y) * w
            num += res**2
            denum += res
        if denum == 0:
            if num == 0:
                return 0
            else:
                return float("inf")
        else:
            return num / denum

    @Distance.target.setter
    def target(self, value):
        self._target = value
        if self._target:
            if self.weights:
                if len(self.weights) != self._target.size:
                    raise DimensionalityError(
                        f"len(weights) must be equal to len(variables) in `ArrayVar` of :ref:`Searchspace`"
                    )
            else:
                self.weights = [1] * self._target.size

            self.operations = []

            for v in self._target.variables:  # type: ignore
                if isinstance(v, FloatVar) or isinstance(v, IntVar):
                    self.operations.append(_Float_int_dist(v))
                elif isinstance(v, CatVar):
                    self.operations.append(cat_dist)
