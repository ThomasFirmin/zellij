# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-26T15:33:50+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.variables import Variable, ArrayVar, FloatVar, IntVar
from zellij.core.loss_func import LossFunc
from zellij.utils.distances import Distance, Mixed, Euclidean
from zellij.core.addons import SearchspaceAddon, Converter
import numpy as np
import copy
import os
import pickle
from abc import ABC, abstractmethod


import logging

logger = logging.getLogger("zellij.space")


class Searchspace(ABC):
    """Searchspace

    Searchspace is an essential class for Zellij. Define your search space with this object.

    Attributes
    ----------

    variables : Variable
        Determines the decision space. See `MixedSearchspace`, `ContinuousSearchspace`,
        `DiscreteSearchspace` for more info.

    loss : LossFunc
        Callable of type `LossFunc`. See :ref:`lf` for more information.
        `loss` will be used by the :ref:`sp` object and by optimization

    See Also
    --------
    LossFunc : Parent class for a loss function.
    """

    def __init__(self, variables, loss, **kwargs):
        """__init__(variables, loss, **kwargs)

        Parameters
        ----------

        variables : ArrayVar
            Determines the decision space. See :code:`MixedSearchspace`, :code:`ContinuousSearchspace`,
            :code:`DiscreteSearchspace` for more info.

        loss : LossFunc
            Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
            :code:`loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

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

        assert isinstance(loss, LossFunc), logger.error(
            f"`loss` must be a `LossFunc`, got {loss}"
        )

        ##############
        # PARAMETERS #
        ##############

        self.variables = variables

        self.loss = loss
        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############

        self.size = len(self.variables)

        self._all_addons = kwargs

        self._add_addons(**kwargs)

        # if true solutions must be converted before being past to loss func.
        self._convert_sol = False

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    def _add_addons(self, **kwargs):
        for k in kwargs:
            assert isinstance(kwargs[k], SearchspaceAddon), logger.error(
                f"Kwargs must be of type `SearchspaceAddon`, got {k}:{kwargs[k]}"
            )

            if kwargs[k]:
                setattr(self, k, copy.copy(kwargs[k]))
                addon = getattr(self, k)
                addon.target = self
            else:
                setattr(self, k, kwargs[k])
                addon = getattr(self, k)
                addon.target = self

            if isinstance(kwargs[k], Converter):
                self._convert_sol = True

    # Return 1 or size=n random attribute from the search space, can exclude one attribute
    def random_attribute(self, size=1, replace=True, exclude=None):
        """random_attribute(size=1,replace=True, exclude=None)

        Draw random features from the search space.

        Parameters
        ----------
        size : int, default=1
            Select randomly <size> features.
        replace : boolean, default=True
            Select randomly <size> features with replacement if True.
            See numpy.random.choice
        exclude : Variable or list[Variable] or type or list[type] or int or list[int], default=None
            Exclude one or several `Variable` to be drawn.
            Can also exclude types.For example one can exclude all Constant type.
            Can also exclude variables according to their index or value.

        Returns
        -------

        out : numpy.array(dtype=int)
            Array of index, corresponding to the selected Variable in `variables`.

        Examples
        --------
        >>> rand_att = sp.random_attribute(3)
        >>> print(rand_att)
        array([FloatVar(float_1, [2;12]), CatVar(cat_1, ['Hello', 87, 2.56]),
         FloatVar(float_1, [2;12])], dtype=object)
        """

        if exclude:
            index = []
            if isinstance(exclude, int):
                index = [exclude]
            elif isinstance(exclude, Variable):
                index = [exclude._idx]  # type: ignore
            elif isinstance(exclude, type):
                index = []
                for elem in self.variables.variables:
                    if isinstance(elem, exclude):
                        index.append(elem._idx)
            elif isinstance(exclude, list) or isinstance(exclude, tuple):
                if all(isinstance(elem, int) for elem in exclude):
                    index = exclude
                elif all(isinstance(elem, Variable) for elem in exclude):
                    index = []
                    for elem in exclude:
                        index.append(elem._idx)
                elif all(isinstance(elem, type) for elem in exclude):
                    index = []
                    for elem in self.variables.variables:
                        if isinstance(elem, tuple(exclude)):
                            index.append(elem._idx)

            p = np.full(self.size, 1 / (self.size - len(index)))
            p[index] = 0
        else:
            p = np.full(self.size, 1 / self.size)

        return np.random.choice(
            self.variables.variables, size=size, replace=replace, p=p
        )

    # Return a random point of the search space
    def random_point(self, size=1):
        """random_point(size=1)

        Return a random point from the search space

        Parameters
        ----------

        size : int, default=1
            Draw <size> points.

        Returns
        -------

        points : list[list[{int, float, str}]]
            List of <point>.

        Examples
        --------

        >>> rand_pts = sp.random_point(3)
        >>> print(f"Random points: {rand_pts}")
        Random points: [[-3.830114043118622, 9, 'sigmoid'],
        ...             [3.065902630698311, 3, 'sigmoid'],
        ...             [-0.6839762230289024, 10, 'relu']]

        """

        return self.variables.random(size)

    def subspace(self, lower, upper, **kwargs):
        """subspace(self, lower, upper)

        Build a sub space according to the actual Searchspace using two vectors containing lower and upper bounds of the subspace.
        Can change type to Constant if necessary

        Parameters
        ----------

        lower : list
            Lower bounds of the subspace. See `Variable` for more info.
        upper : boolean, default=False
            Upper bounds of the subspace. See `Variable` for more info.

        Returns
        -------

        out : Searchspace
            Return a subspace of the actual Searchspace.

        Examples
        --------

        """

        sub = self.variables.subset(lower, upper)
        sp = type(self)(sub, self.loss, **kwargs, **self.kwargs)

        return sp

    def save(self, path):
        pickle.dump(self, open(os.path.join(path, "searchspace.p"), "wb"))

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.variables)


class MixedSearchspace(Searchspace):

    """MixedSearchspace

    :code:`MixedSearchspace` is a search space made for HyperParameter Optimization (HPO).
    The decision space can be made of various `Variable` types.

    Attributes
    ----------

    variables : ArrayVar
        Determines the bounds of the search space.
        For `ContinuousSearchspace` the `variables` must be an `ArrayVar`
        of `FloatVar`, `IntVar`, `CatVar`.
        The :ref:`sp` will then manipulate this array.

    loss : LossFunc
        Callable of type `LossFunc`. See :ref:`lf` for more information.
        `loss` will be used by the :ref:`sp` object and by optimization


    Methods
    -------
    random_attribute(self,size=1,replace=True, exclude=None)
        Draw random features from the search space.
        Return the selected `Variable`

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

    """

    # Initialize the search space
    def __init__(self, variables, loss, **kwargs):
        """__init__(variables, loss, **kwargs)

        Parameters
        ----------

        variables : ArrayVar
            Determines the bounds of the search space.
            For :code:`ContinuousSearchspace` the :code:`variables` must be an :code:`ArrayVar`
            of :code:`FloatVar`, :code:`IntVar`, :code:`CatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
            :code:`loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * er : Converter
                * Will be called when converting a solution to another space is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)


        """

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(variables, ArrayVar), logger.error(
            f"`variables` must be be an `ArrayVar`, got {variables}"
        )

        self.distance = kwargs.pop("distance", Mixed())
        super(MixedSearchspace, self).__init__(variables, loss, **kwargs)
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self

    @Searchspace.loss.setter
    def loss(self, value):
        self._loss = value
        if self._loss and not self.loss.labels:
            self.loss.labels = [v.label for v in self.variables]


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
        The :ref:`sp` will then manipulate this array.

    loss : LossFunc
        Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
        :code:`loss` will be used by the :ref:`sp` object and by optimization


    Methods
    -------
    random_attribute(self,size=1,replace=True, exclude=None)
        Draw random features from the search space.
        Return the selected `Variable`

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
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.core.loss_func import Loss
    >>> from zellij.utils.benchmark import himmelblau
    >>> from zellij.core.search_space import ContinuousSearchspace
    >>> lf = Loss()(himmelblau)
    >>> a = ArrayVar(FloatVar("float_1", 0,1),
    ...              FloatVar("float_2", 0,1))
    >>> sp = ContinuousSearchspace(a,lf)
    >>> p1,p2 = sp.random_point(), sp.random_point()
    >>> print(p1)
    [0.8922761649920034, 0.12709277668616326]
    >>> print(p2)
    [0.7730279148456985, 0.14715728189857524]

    """

    # Initialize the search space
    def __init__(self, variables, loss, **kwargs):
        """__init__(variables, loss, **kwargs)

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
        super(ContinuousSearchspace, self).__init__(variables, loss, **kwargs)

        ##############
        # ASSERTIONS #
        ##############

        cont_condition = all(isinstance(v, FloatVar) for v in variables)
        conv_condition = all(hasattr(v, "converter") for v in variables)

        assert isinstance(variables, ArrayVar) and (
            cont_condition or conv_condition
        ), logger.error(
            f"`variables` must be be an `ArrayVar` of `FloatVar`, got {variables}"
        )

        self.lower = np.zeros(self.size)
        self.upper = np.ones(self.size)

        if cont_condition:
            for idx, v in enumerate(variables):
                self.lower[idx] = v.lower
                self.upper[idx] = v.upper

        self.distance = kwargs.pop("distance", Euclidean(self))
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self

    @Searchspace.loss.setter
    def loss(self, value):
        self._loss = value
        if self._loss and not self.loss.labels:
            self.loss.labels = [v.label for v in self.variables]


class DiscreteSearchspace(Searchspace):

    """DiscreteSearchspace

    :code:`DiscreteSearchspace` is a search space made for continuous optimization.
    The decision space is made of :code:`IntVar`.

    Attributes
    ----------

    variables : ArrayVar
        Determines the bounds of the search space.
        For :code:`DiscreteSearchspace` the :code:`variables` must be an :code:`ArrayVar`
        of :code:`IntVar`.
        The :ref:`sp` will then manipulate this array.

    loss : LossFunc
        Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
        :code:`loss` will be used by the :ref:`sp` object and by optimization


    Methods
    -------
    random_attribute(self,size=1,replace=True, exclude=None)
        Draw random features from the search space.
        Return the selected `Variable`

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
    >>> from zellij.core.loss_func import Loss
    >>> from zellij.utils.benchmark import himmelblau
    >>> from zellij.core.search_space import DiscreteSearchspace
    >>> a = ArrayVar(IntVar("int_1", 0,8),
    >>>              IntVar("int_2", 4,45))
    >>> lf = Loss()(himmelblau)
    >>> sp = DiscreteSearchspace(a,lf)
    >>> p1,p2 = sp.random_point(), sp.random_point()
    >>> print(p1)
    [5, 34]
    >>> print(p2)
    [3, 42]

    """

    # Initialize the search space
    def __init__(self, variables, loss, **kwargs):
        """__init__(variables, loss, **kwargs)

        Parameters
        ----------

        variables : ArrayVar
            Determines the bounds of the search space.
            For :code:`DiscreteSearchspace` the :code:`variables` must be an :code:`ArrayVar`
            of :code:`IntVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
            :code:`loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

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
        super(DiscreteSearchspace, self).__init__(variables, loss, **kwargs)

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(variables, ArrayVar) and all(
            isinstance(v, IntVar) for v in variables
        ), logger.error(
            f"`variables` must be be an `ArrayVar` of `FloatVar`, got {variables}"
        )

        self.distance = kwargs.pop("distance", Euclidean(self))
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self

    @Searchspace.loss.setter
    def loss(self, value):
        self._loss = value
        if self._loss and not self.loss.labels:
            self.loss.labels = [v.label for v in self.variables]


class BaseFractal(Searchspace):
    """Fractal

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

    variables : list[float]
        List of objective variables.

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
        variables,
        loss,
        measure=None,
        **kwargs,
    ):
        """__init__(variables, loss, measure=None, **kwargs)

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
        super(BaseFractal, self).__init__(variables, loss, **kwargs)

        self._compute_mesure = measure
        self.measure = float("nan")

        self.level = 0
        self.father = -1  # father id, -1 = no father
        self.f_id = 0  # fractal id at a given level
        self.c_id = 0  # Children id

        self.score = float("nan")

        self.solutions = []
        self.losses = []

    @Searchspace.loss.setter
    def loss(self, value):
        self._loss = value
        if self._loss and not self.loss.labels:
            self.loss.labels = [v.label for v in self.variables]

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        def init_measurement(self, *args, init=cls.__init__, **kwargs):
            init(self, *args, **kwargs)
            if cls is type(self):
                self._update_measure()

        cls.__init__ = init_measurement

    def _update_measure(self):
        if self._compute_mesure:
            self.measure = self._compute_mesure(self)
        else:
            self.measure = float("nan")

    def add_solutions(self, X, Y):
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
        self.losses.extend(Y)

    def get_id(self):
        return (self.level, self.father, self.c_id)

    def _compute_f_id(self, k):
        if self.father == -1:
            return 0, k
        else:
            base = self.father * k
            return base, base + k

    def create_children(self, k, *args, **kwargs):
        """create_children(self)

        Defines the partition function.
        Determines how children of the current space should be created.

        The child will inherit the parent's score.

        """

        children = [
            type(self)(
                self.variables,
                self.loss,
                self._compute_mesure,
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

    def _modify(self, level, father, f_id, c_id, score, measure):
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

    def _essential_info(self):
        return {
            "level": self.level,
            "father": self.father,
            "f_id": self.f_id,
            "c_id": self.c_id,
            "score": self.score,
            "measure": self.measure,
        }

    def __repr__(self):
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


class Fractal(BaseFractal):
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
        variables,
        loss,
        measure=None,
        **kwargs,
    ):
        """__init__(variables, loss, measure=None, **kwargs)

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

        # remove distance from kwargs as it is a default addon.
        distance = kwargs.pop("distance", Euclidean())

        super(Fractal, self).__init__(variables, loss, **kwargs)

        assert self._is_valid_fractal(), logger.error(
            "`variables` must be be an `ArrayVar` of `FloatVar`,"
            f"or all `Var` must have a `converter` addon, got {variables}"
        )

        assert isinstance(distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {distance}"
        )
        self.distance = distance
        self.distance.target = self

        self.lower = np.zeros(self.size)
        self.upper = np.ones(self.size)

    def _is_valid_fractal(self):
        conv_condition = all(
            hasattr(v, "converter") for v in self.variables
        ) and hasattr(self.variables, "converter")
        unit_cond = True

        cv = 0  # current value
        while cv < self.size and unit_cond:
            v = self.variables[cv]
            if isinstance(v, FloatVar):
                if v.lower != 0 or v.upper != 1:
                    unit_cond = False
            else:
                unit_cond = False

        return isinstance(self.variables, ArrayVar) and (unit_cond or conv_condition)
