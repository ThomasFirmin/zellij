# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)
from __future__ import annotations
from abc import abstractmethod
from zellij.core.errors import InitializationError, InputError
from zellij.core.variables import (
    ArrayVar,
    FloatVar,
    IntVar,
    PermutationVar,
    IterableVar,
)
from zellij.utils.distances import Euclidean, Mixed, Manhattan

from typing import (
    Optional,
    Tuple,
    List,
    Sequence,
    Union,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from zellij.strategies.tools.measurements import Measurement
    from zellij.core.addons import Distance

    FR = TypeVar("FR", bound="Fractal")
    MFR = TypeVar("MFR", bound="MixedFractal")

import numpy as np
import os
import pickle
from abc import ABCMeta


import logging

logger = logging.getLogger("zellij.space")


class MetaSearchspace(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._check_addons(instance.variables)
        return instance


class Searchspace(metaclass=MetaSearchspace):
    """Searchspace

    Searchspace is an essential class for Zellij. Define your search space with this object.

    Attributes
    ----------
    variables : IterableVar
        Determines the decision space. See `MixedSearchspace`, `ContinuousSearchspace`,
        `DiscreteSearchspace` for more info.

    See Also
    --------
    LossFunc : Parent class for a loss function.
    """

    def __init__(self, variables: IterableVar):
        """__init__

        Parameters
        ----------
        variables : IterableVar
            Determines the decision space. See :code:`MixedSearchspace`, :code:`ContinuousSearchspace`,
            :code:`DiscreteSearchspace` for more info.
        """
        #############
        # VARIABLES #
        #############
        # if true solutions must be converted before being past to loss func.
        self._do_convert = False
        self._do_neighborhood = False
        self.size = len(variables)

        ##############
        # PARAMETERS #
        ##############

        self.variables = variables

    @property
    def variables(self) -> IterableVar:
        return self._variables

    @variables.setter
    def variables(self, value: IterableVar):
        if isinstance(value, IterableVar):
            self._variables = value
        else:
            raise InitializationError(
                f"In Searchspace, variables must be defined within an ArrayVar. Got {type(value)}"
            )

    def _check_addons(self, value):
        if value.converter:
            self._do_convert = True
        else:
            self._do_convert = False
        if value.neighborhood:
            self._do_neighborhood = True
        else:
            self._do_neighborhood = False

    # Return a random point of the search space
    def random_point(self, size: Optional[int] = None) -> list:
        """random_point

        Return a random point from the search space

        Parameters
        ----------
        size : int, optional
            Draw <size> points. If None returns a single point.

        Returns
        -------
        points : {list, list[list]
            List of <point>.
        """

        return self.variables.random(size)

    def convert(self, X: List[list]) -> List[list]:
        if isinstance(X[0], list):
            if self.variables.converter:
                res = []
                for x in X:
                    res.append(self.variables.converter.convert(x))
                return res
            else:
                raise InitializationError(
                    f"In {type(self).__name__} `convert` was called while no converter is set."
                )
        else:
            raise InputError(
                f"In {type(self).__name__} `convert` X must be a list of solutions, list of list. Got {X}."
            )

    def reverse(self, X: list) -> list:
        if isinstance(X, list):
            if self.variables.converter:
                res = []
                for x in X:
                    res.append(self.variables.converter.reverse(x))
                return res
            else:
                raise InitializationError(
                    f"In {type(self).__name__} `reverse` was called while no converter is set."
                )
        else:
            raise InputError(
                f"In {type(self).__name__} `reverse` X must be a list of solutions, list of list. Got {X}."
            )

    def neighborhood(self, X: List[list], size: Optional[int] = None) -> List[list]:
        if self.variables.neighborhood:
            res = []
            for x in X:
                res.extend(self.variables.neighborhood(x, size=size))
            return res
        else:
            raise InitializationError(
                f"In {type(self).__name__} `neighborhood` was called while no converter is set."
            )

    def save(self, path: str):
        pickle.dump(self, open(os.path.join(path, "searchspace.p"), "wb"))

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self.variables)


class MixedSearchspace(Searchspace):
    """MixedSearchspace

    :code:`MixedSearchspace` is a search space made for HyperParameter Optimization (HPO).
    The decision space can be made of various `Variable` types.

    Attributes
    ----------
    variables : IterableVar
        Determines the bounds of the search space.
        For `ContinuousSearchspace` the `variables` must be an `IterableVar`
        of `FloatVar`, `IntVar`, `CatVar`.
    distance : Distance, optional
        Distance object defining the distance between two point within this :ref:`sp`.
        By default :code:`Mixed()`.

    Methods
    -------
    random_point(self,size=1)
        Return random points from the search space
    subspace(self,lower,upper)
        Build a sub space according to the actual Searchspace using two vectors
        containing lower and upper bounds of the subspace.

    See Also
    --------
    LossFunc : Parent class for a loss function.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, IntVar, FloatVar, CatVar
    >>> from zellij.core import MixedSearchspace
    >>> a = ArrayVar(
    ...             IntVar("i1", 0, 8),
    ...             IntVar("i2", 30, 40),
    ...             FloatVar("f1", 10, 20),
    ...             CatVar("c1", ["Hello", 87, 2.56]),
    ...         )
    >>> sp = MixedSearchspace(a)
    >>> p = sp.random_point()
    >>> print(p)
    [7, 32, 19.70695001272685, 2.56]
    >>> p = sp.random_point(2)
    >>> print(p)
    [[7, 37, 18.734133066154932, 87],
    [3, 35, 12.003121963930703, 2.56]]
    >>> sp.distance(p[0],p[1])
    0.5411583306216192
    """

    # Initialize the search space
    def __init__(self, variables: IterableVar, distance: Optional[Distance] = None):
        """__init__

        Parameters
        ----------
        variables : IterableVar
            Determines the bounds of the search space.
        """

        ##############
        # ASSERTIONS #
        ##############
        super(MixedSearchspace, self).__init__(variables)
        self.distance = distance


class ContinuousSearchspace(Searchspace):
    """ContinuousSearchspace

    :code:`ContinuousSearchspace` is a search space made for continuous optimization.
    The decision space is made of `FloatVar` or all variables must have a `converter`
    :ref:`addons`.

    Attributes
    ----------
    variables : ArrayVar
        Determines the bounds of the search space.
        For :code:`ContinuousSearchspace` the :code:`variables` must be an :code:`ArrayVar`
        of :code:`FloatVar`.
    distance : Distance, optional
        Distance object defining the distance between two point within this :ref:`sp`.
        By default :code:`Euclidean()`.

    Methods
    -------
    random_point(self,size=None)
        Return random points from the search space

    subspace(self,lower,upper)
        Build a sub space according to the actual Searchspace using two vectors
        containing lower and upper bounds of the subspace.

    See Also
    --------
    LossFunc : Parent class for a loss function.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.core import ContinuousSearchspace
    >>> a = ArrayVar(FloatVar("f1", 0, 10),FloatVar("f2", -10, 0))
    >>> sp = ContinuousSearchspace(a)
    >>> p = sp.random_point()
    >>> print(p)
    [7.8040745696654845, -4.360651502175973]
    >>> p = sp.random_point(2)
    >>> print(p)
    [[2.409486588746743, -0.19724026255489946],
    [2.962067847388803, -5.178636429752261]]
    >>> sp.distance(p[0],p[1])
    5.011951099319607

    """

    # Initialize the search space
    def __init__(self, variables: ArrayVar, distance: Optional[Distance] = None):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
        distance : Distance, optional
            Distance object defining the distance between two point within this :ref:`sp`.
        """
        super(ContinuousSearchspace, self).__init__(variables)
        self.distance = distance

    @Searchspace.variables.setter
    def variables(self, value: ArrayVar):
        if isinstance(value, ArrayVar):
            cont_condition = all(isinstance(v, FloatVar) for v in value)
            conv_condition = (
                all(v.converter is not None for v in value) and value.converter
            )

            if cont_condition or conv_condition:
                self._variables = value
                self.lower = np.zeros(len(value), dtype=float)
                self.upper = np.ones(len(value), dtype=float)

                if value.converter and conv_condition:
                    self._do_convert = True
                else:
                    self._do_convert = False
                    for idx, v in enumerate(value):
                        self.lower[idx] = v.lower  # type: ignore
                        self.upper[idx] = v.upper  # type: ignore

                if self._variables.neighborhood:
                    self._do_neighborhood = True
                else:
                    self._do_neighborhood = False
            else:
                raise InitializationError(
                    f"In {type(self).__name__}, variables must be FloatVar defined within an ArrayVar, or have a converter. Got {type(value)}"
                )
        else:
            raise InitializationError(
                f"In {type(self).__name__}, variables must be FloatVar defined within an ArrayVar, or have a converter. Got {type(value)}"
            )

    @property
    def distance(self) -> Distance:
        return self._distance

    @distance.setter
    def distance(self, value: Optional[Distance]):
        if value:
            self._distance = value
        else:
            self._distance = Euclidean()
        self._distance.target = self

    # Return a random point of the search space
    def random_point(self, size: Optional[int] = None) -> Union[list, List[list]]:
        if size:
            points = super().random_point(size=size)
            if self._do_convert:
                points = self.convert(points)
        else:
            points = super().random_point(size=size)
            if self._do_convert:
                points = self.convert([points])[0]
        return points


class UnitSearchspace(ContinuousSearchspace):
    @Searchspace.variables.setter
    def variables(self, value: ArrayVar):
        if isinstance(value, ArrayVar):
            # Converter condition
            conv_condition = (
                all(v.converter is not None for v in value) and value.converter
            )

            # Unit cube condition
            unit_cond = True
            cv = 0  # current value
            while cv < self.size and unit_cond:
                v = value[cv]

                if isinstance(v, FloatVar):
                    if v.lower != 0 or v.upper != 1:
                        unit_cond = False
                else:
                    unit_cond = False

                cv += 1

            if unit_cond or conv_condition:
                self._variables = value
                self.lower = np.zeros(self.size)
                self.upper = np.ones(self.size)

                if conv_condition:
                    self._do_convert = True
                else:
                    self._do_convert = False

                if value.neighborhood:
                    self._neighborhood = True
                else:
                    self._neighborhood = False
            else:
                raise InitializationError(
                    f"In {type(self).__name__}, variables must be FloatVar within [0,1] defined within an ArrayVar, or have a converter."
                )
        else:
            raise InitializationError(
                f"In {type(self).__name__}, variables must be FloatVar within [0,1] defined within an ArrayVar, or have a converter."
            )


class DiscreteSearchspace(Searchspace):
    """DiscreteSearchspace

    :code:`DiscreteSearchspace` is a search space made for continuous optimization.
    The decision space is made of :code:`IntVar`.

    Attributes
    ----------
    variables : ArrayVar
        Determines the bounds of the search space.
    distance : Distance, optional
        Distance object defining the distance between two point within this :ref:`sp`.
        By default :code:`Manhattan()`.

    Methods
    -------
    random_point(self,size=1)
        Return random points from the search space
    subspace(self,lower,upper)
        Build a sub space according to the actual Searchspace using two vectors
        containing lower and upper bounds of the subspace.

    See Also
    --------
    LossFunc : Parent class for a loss function.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, IntVar
    >>> from zellij.core import DiscreteSearchspace
    >>> a = ArrayVar(IntVar("i1", 0, 10),IntVar("i2", -10, 0))
    >>> sp = DiscreteSearchspace(a)
    >>> p = sp.random_point()
    >>> print(p)
    [3, -2]
    >>> p = sp.random_point(2)
    >>> print(p)
    [[1, -4], [7, -7]]
    >>> sp.distance(p[0],p[1])
    6

    """

    # Initialize the search space
    def __init__(self, variables: ArrayVar, distance: Optional[Distance] = None):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
            For :code:`DiscreteSearchspace` the :code:`variables` must be an :code:`ArrayVar`
            of :code:`IntVar`.
        """
        super(DiscreteSearchspace, self).__init__(variables)
        self.distance = distance

    @Searchspace.variables.setter
    def variables(self, value: ArrayVar):
        if isinstance(value, ArrayVar):
            cont_condition = all(isinstance(v, IntVar) for v in value)
            conv_condition = all(v.converter is not None for v in value)

            if cont_condition or conv_condition:
                self._variables = value
                self.lower = np.zeros(len(value), dtype=int)
                self.upper = np.ones(len(value), dtype=int)

                if self._variables.converter and conv_condition:
                    self._do_convert = True
                else:
                    self._do_convert = False
                    for idx, v in enumerate(value):
                        self.lower[idx] = v.lower  # type: ignore
                        self.upper[idx] = v.upper  # type: ignore

                if self._variables.neighborhood:
                    self._do_neighborhood = True
                else:
                    self._do_neighborhood = False
            else:
                raise InitializationError(
                    f"In {type(self).__name__}, variables must be IntVar defined within an ArrayVar, or have a converter. Got {type(value)}"
                )
        else:
            raise InitializationError(
                f"In {type(self).__name__}, variables must be IntVar defined within an ArrayVar, or have a converter. Got {type(value)}"
            )

    @property
    def distance(self) -> Distance:
        return self._distance

    @distance.setter
    def distance(self, value: Optional[Distance]):
        if value:
            self._distance = value
        else:
            self._distance = Manhattan()
        self._distance.target = self

    # Return a random point of the search space
    def random_point(self, size: Optional[int] = None) -> Union[list, List[list]]:
        if size:
            points = super().random_point(size=size)
            if self._do_convert:
                points = self.convert(points)
        else:
            points = super().random_point(size=size)
            if self._do_convert:
                points = self.convert([points])[0]
        return points


class MetaFrac(MetaSearchspace):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._update_measure()
        return instance


class BaseFractal(Searchspace, metaclass=MetaFrac):
    """BaseFractal

    BaseFractal is an abstract describing what an fractal object is.
    This class is used to build a new kind of search space.

    Fractals can be compared between eachothers, using:
    :code:`__lt__`, :code:`__le__`, :code:`__eq__`,
    :code:`__ge__`, :code:`__gt__`, :code:`__ne__` operators.

    Attributes
    ----------
    level : int
        Current level of the fractal in the partition tree. See Tree_search.
    father : int
        Fractal id of the parent of the fractal.
    f_id : int
        ID of the fractal at the current level.
    c_id : int
        ID of the child among the children of the parent.
    score : float
        Score of the fractal. By default the score of the fractal is equal
        to the score of its parent (inheritance), so it can be locally
        used and modified by the :ref:`scoring`.
    solutions : list[list[float]]
        List of solutions computed within the fractal
    measure : float, default=NaN
        Measure of the fractal, obtained by a :code:`Measurement`.

    See Also
    --------
    :ref:`lf` : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    :ref:`sp` : Initial search space used to build fractal.
    Hypercube : Inherited Fractal type
    Hypersphere : Inherited Fractal type
    """

    def __init__(
        self,
        variables: IterableVar,
        measurement: Optional[Measurement] = None,
    ):
        """__init__

        Parameters
        ----------
        variables : IterableVar
            Determines the bounds of the search space.
        measurement : Measurement, optional
            Defines the measure of a fractal.
        """
        super(BaseFractal, self).__init__(variables)
        self._compute_measure = measurement

        #############
        # VARIABLES #
        #############

        self.measure = float("nan")

        self.level = 0
        self.father = -1  # father id, -1 = no father
        self.f_id = 0  # fractal id at a given level
        self.c_id = 0  # Children id
        self.score = float("inf")

        self.solutions = []
        self.losses = np.array([], dtype=float)
        self.secondary_loss = None
        self.constraint_val = None

    def _update_measure(self):
        if self._compute_measure:
            self.measure = self._compute_measure(self)
        else:
            self.measure = float("nan")

    def add_solutions(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ):
        """add_solutions

        Add computed solutions to the fractal.

        Parameters
        ----------
            X : list[list[float]]
                List of computed solutions.
            Y : list[float]
                List of loss values associated to X.
        """
        self.solutions.extend(X)
        self.losses = np.append(self.losses, Y)
        if secondary:
            if self.secondary_loss:
                self.secondary_loss = np.vstack((self.secondary_loss, secondary))
            else:
                self.secondary_loss = secondary

        if constraint:
            if self.constraint_val:
                self.constraint_val = np.vstack((self.constraint_val, constraint))
            else:
                self.constraint_val = secondary

    def get_id(self):
        return (self.level, self.father, self.c_id)

    def _compute_f_id(self, k: int) -> Tuple[int, int]:
        """_compute_f_id
        Compute range of IDs according to partition size k
        """
        base = self.f_id * k
        return base, base + k

    @abstractmethod
    def create_children(self) -> Sequence[BaseFractal]:
        pass

    def _modify(
        self,
        level: int,
        father: int,
        f_id: int,
        c_id: int,
        score: float,
        measure: float,
    ):
        """_modify

        Modify the fractal according to given info. used for sending fractals in
        distributed environment.

        """
        self.level = level
        self.father = father
        self.f_id = f_id
        self.c_id = c_id
        self.score = score
        self.measure = measure
        self.solutions = []
        self.losses = []

    def _essential_info(self) -> dict:
        return {
            "level": self.level,
            "father": self.father,
            "f_id": self.f_id,
            "c_id": self.c_id,
            "score": self.score,
            "measure": self.measure,
        }

    def __repr__(self) -> str:
        """_essential_info
        Essential information to create the fractal. Used in _modify
        """
        return f"{type(self).__name__}({self.level},{self.father},{self.c_id})"

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __eq__(self, other):
        return self.score == other.score

    def __ge__(self, other):
        return self.score > other.score

    def __gt__(self, other):
        return self.score >= other.score

    def __ne__(self, other):
        return self.score != other.score


class Fractal(BaseFractal, UnitSearchspace):
    """Fractal

    Fractal is an abstract class used in DBA.
    This class is used to build a new kind of search space.
    It is a :code:`Searchspace`, but it works with
    the unit hypercube. Bounds: [[0,...,0], [1,...,1]].

    Fractals are constrained continuous subspaces.

    Fractals can be compared between eachothers, using:
    :code:`__lt__`, :code:`__le__`, :code:`__eq__`,
    :code:`__ge__`, :code:`__gt__`, :code:`__ne__` operators.

    Attributes
    ----------
    level : int
        Current level of the fractal in the partition tree. See Tree_search.
    father : int
        Fractal id of the parent of the fractal.
    f_id : int
        ID of the fractal at the current level.
    c_id : int
        ID of the child among the children of the parent.
    score : float
        Score of the fractal. By default the score of the fractal is equal
        to the score of its parent (inheritance), so it can be locally
        used and modified by the :ref:`scoring`.
    solutions : list[list[float]]
        List of solutions computed within the fractal
    variables : list[float]
        List of objective variables.
    lower : list[0.0,...,0.0]
        Lower bounds of the unit hypercube.
    upper : list[1.0,...,1.0]
        Upper bounds of the unit hypercube.
    measure : float, default=NaN
        Measure of the fractal, obtained by a :code:`Measurement`.

    See Also
    --------
    :ref:`lf` : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    :ref:`sp` : Initial search space used to build fractal.
    Hypercube : Inherited Fractal type
    Hypersphere : Inherited Fractal type
    """

    def __init__(
        self,
        variables: ArrayVar,
        measurement: Optional[Measurement] = None,
        distance: Optional[Distance] = None,
    ):
        """__init__

        Parameters
        ----------
        variables : ArrayVar
            Determines the bounds of the search space.
        distance : Distance, optional
            Distance object defining the distance between two point within this :ref:`sp`.
            By default :code:`Euclidean()`.
        measurement : Measurement, optional
            Defines the measure of a fractal.

        """
        self._do_convert = False
        super(Fractal, self).__init__(variables=variables, measurement=measurement)
        self.distance = distance

    def create_children(self: FR, k: int, *args, **kwargs) -> Sequence[FR]:
        """create_children(self)

        Defines the partition function.
        Determines how children of the current space should be created.

        The child will inherit the parent's score.

        Attributes
        ----------
        k : int
            Partition size
        """

        children = [
            type(self)(
                variables=self.variables,  # type: ignore
                measurement=self._compute_measure,
                *args,
                **kwargs,
            )
            for _ in range(k)
        ]
        low_id, up_id = self._compute_f_id(k)
        for c_id, f_id in enumerate(range(low_id, up_id)):
            children[c_id].level = self.level + 1
            children[c_id].father = self.f_id
            children[c_id].c_id = c_id
            children[c_id].f_id = f_id
            children[c_id].score = self.score
            children[c_id]._update_measure()

        return children

    @property
    def distance(self) -> Distance:
        return self._distance

    @distance.setter
    def distance(self, value: Optional[Distance]):
        if value:
            self._distance = value
        else:
            self._distance = Euclidean()
        self._distance.target = self


class MixedFractal(BaseFractal, MixedSearchspace):
    """MixedFractal

    MixedFractal is an abstract where variables can be of Mixed types.

    MixedFractals can be compared between eachothers, using:
    :code:`__lt__`, :code:`__le__`, :code:`__eq__`,
    :code:`__ge__`, :code:`__gt__`, :code:`__ne__` operators.

    Attributes
    ----------
    level : int
        Current level of the fractal in the partition tree. See Tree_search.
    father : int
        Fractal id of the parent of the fractal.
    f_id : int
        ID of the fractal at the current level.
    c_id : int
        ID of the child among the children of the parent.
    score : float
        Score of the fractal. By default the score of the fractal is equal
        to the score of its parent (inheritance), so it can be locally
        used and modified by the :ref:`scoring`.
    solutions : list[list[float]]
        List of solutions computed within the fractal
    measure : float, default=NaN
        Measure of the fractal, obtained by a :code:`Measurement`.

    See Also
    --------
    :ref:`lf` : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal partition tree.
    :ref:`sp` : Initial search space used to build fractal.
    Hypercube : Inherited Fractal type
    Hypersphere : Inherited Fractal type
    """

    def __init__(
        self, variables: IterableVar, measurement: Optional[Measurement] = None
    ):
        """__init__

        Parameters
        ----------
        variables : IterableVar
            Determines the bounds of the search space.
        distance : Distance, optional
            Distance object defining the distance between two point within this :ref:`sp`.
            By default :code:`Euclidean()`.
        measurement : Measurement, optional
            Defines the measure of a fractal.

        """
        self._do_convert = False
        super(MixedFractal, self).__init__(variables=variables, measurement=measurement)

    def create_children(self: MFR, k: int, *args, **kwargs) -> Sequence[MFR]:
        """create_children(self)

        Defines the partition function.
        Determines how children of the current space should be created.

        The child will inherit the parent's score.

        Attributes
        ----------
        k : int
            Partition size
        """

        children = [
            type(self)(
                variables=self.variables,
                measurement=self._compute_measure,
                *args,
                **kwargs,
            )
            for _ in range(k)
        ]

        low_id, up_id = self._compute_f_id(k)
        for c_id, f_id in enumerate(range(low_id, up_id)):
            children[c_id].level = self.level + 1
            children[c_id].father = self.f_id
            children[c_id].c_id = c_id
            children[c_id].f_id = f_id
            children[c_id].score = self.score
            children[c_id]._update_measure()

        return children
