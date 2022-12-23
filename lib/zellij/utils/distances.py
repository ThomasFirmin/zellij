# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-24T14:52:56+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:38:54+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.addons import Distance
from zellij.core.variables import FloatVar, IntVar, CatVar, Constant
from scipy.spatial import distance
import numpy as np
import logging

logger = logging.getLogger("zellij.distances")


class Euclidean(Distance):
    """Euclidean distance

    Compute the Euclidean distance between two points.
    More info on `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean>`__

    Example
    -------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.utils.distances import Euclidean
    >>> from zellij.core.loss_func import Loss
    >>> from zellij.utils.benchmark import himmelblau
    >>> from zellij.core.search_space import ContinuousSearchspace
    >>> lf = Loss()(himmelblau)
    >>> a = ArrayVar(FloatVar("float_1", 0,1),
    ...              FloatVar("float_2", 0,1))
    >>> sp = ContinuousSearchspace(a,lf, distance=Euclidean())
    >>> p1,p2 = sp.random_point(), sp.random_point()
    >>> print(p1)
    [0.8922761649920034, 0.12709277668616326]
    >>> print(p2)
    [0.7730279148456985, 0.14715728189857524]
    >>> sp.distance(p1,p2)
    0.12092447863180801

    See also
    --------
    :ref:`dist`: Distance addons

    """

    def __call__(self, point_a, point_b):
        return distance.euclidean(point_a, point_b, self.weights)


class Manhattan(Distance):
    """Manhattan distance

    Compute the Manhattan distance between two points.
    More info on `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html#scipy.spatial.distance.cityblock>`__

    Example
    -------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.utils.distances import Manhattan
    >>> from zellij.core.loss_func import Loss
    >>> from zellij.utils.benchmark import himmelblau
    >>> from zellij.core.search_space import ContinuousSearchspace
    >>> lf = Loss()(himmelblau)
    >>> a = ArrayVar(FloatVar("float_1", 0,1),
    ...              FloatVar("float_2", 0,1))
    >>> sp = ContinuousSearchspace(a,lf, distance=Manhattan())
    >>> p1,p2 = sp.random_point(), sp.random_point()
    >>> print(p1)
    [0.12946481931952147, 0.31940702810480137]
    >>> print(p2)
    [0.32347527913737095, 0.9356077155539462]
    >>> sp.distance(p1,p2)
    0.8102111472669943

    See also
    --------
    :ref:`dist`: Distance addons

    """

    def __call__(self, point_a, point_b):
        return distance.cityblock(point_a, point_b, self.weights)


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


    Example
    -------
    >>> from zellij.core.variables import ArrayVar, IntVar, FloatVar, CatVar
    >>> from zellij.utils.distances import Mixed
    >>> from zellij.core.loss_func import Loss
    >>> from zellij.utils.benchmark import himmelblau
    >>> from zellij.core.search_space import MixedSearchspace
    >>> a = ArrayVar(IntVar("int_1", 0,8),
    >>>              IntVar("int_2", 4,45),
    >>>              FloatVar("float_1", 2,12),
    >>>              CatVar("cat_1", ["Hello", 87, 2.56]))
    >>> lf = Loss()(himmelblau)
    >>> sp = MixedSearchspace(a,lf, distance=Mixed())
    >>> p1,p2 = sp.random_point(), sp.random_point()
    >>> print(p1)
    [5, 34, 4.8808143412719485, 87]
    >>> print(p2)
    [3, 42, 2.8196595134477738, 'Hello']
    >>> sp.distance(p1,p2)
    0.5990169287736146

    See also
    --------
    :ref:`dist`: Distance addons

    """

    def __init__(self, search_space=None, weights=None):
        self.weights = weights
        self.operations = []

        super(Mixed, self).__init__(search_space)

    def __call__(self, point_a, point_b):
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
    def target(self, object):
        from zellij.core.search_space import MixedSearchspace

        self._target = object

        assert (
            isinstance(self._target, MixedSearchspace) or object == None
        ), logger.error(
            f"Target must be of type `MixedSearchspace`, got {object}"
        )
        if self._target:
            if self.weights:
                assert len(self.weights) == self._target.size, logger.error(
                    f"len(weights) must be equal to len(values) in `ArrayVar` of :ref:`Searchspace`"
                )
            else:
                self.weights = [1] * self._target.size

            self.operations = []

            for v in self._target.values:
                if isinstance(v, FloatVar) or isinstance(v, IntVar):
                    up, lo = v.up_bound, v.low_bound
                    self.operations.append(
                        lambda x, y: np.abs(x - y) / (up - lo)
                    )
                elif isinstance(v, CatVar):
                    self.operations.append(lambda x, y: 1 if x == y else 0)
                elif isinstance(v, Constant):
                    self.operations.append(lambda x, y: 0)
