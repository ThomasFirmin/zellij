# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-07T16:58:11+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.variables import Variable, ArrayVar, FloatVar, IntVar
from zellij.core.loss_func import LossFunc
from zellij.utils.distances import Distance, Mixed, Euclidean
from zellij.core.addons import SearchspaceAddon
import numpy as np
import copy
import os
from abc import ABC, abstractmethod


import logging

logger = logging.getLogger("zellij.space")


class Searchspace(ABC):
    """Searchspace

    Searchspace is an essential class for Zellij. Define your search space with this object.

    Attributes
    ----------

    values : Variable
        Determines the decision space. See `MixedSearchspace`, `ContinuousSearchspace`,
        `DiscreteSearchspace` for more info.

    loss : LossFunc
        Callable of type `LossFunc`. See :ref:`lf` for more information.
        `loss` will be used by the :ref:`sp` object and by optimization

    See Also
    --------
    LossFunc : Parent class for a loss function.
    """

    def __init__(self, values, loss, **kwargs):
        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the decision space. See :code:`MixedSearchspace`, :code:`ContinuousSearchspace`,
            :code:`DiscreteSearchspace` for more info.

        loss : LossFunc
            Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
            :code:`loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
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

        self.values = values

        self.loss = loss
        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############

        self.size = len(self.values)

        self._add_addons(**kwargs)

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
            Array of index, corresponding to the selected Variable in `values`.

        Examples
        --------
        >>> rand_att = sp.random_attribute(3)
        >>> print(rand_att)
        array([FloatVar(float_1, [2;12]), CatVar(cat_1, ['Hello', 87, 2.56]),
         FloatVar(float_1, [2;12])], dtype=object)
        """

        if exclude:
            if isinstance(exclude, int):
                index = [exclude]
            elif isinstance(exclude, Variable):
                index = [exclude._idx]
            elif isinstance(exclude, type):
                index = []
                for elem in self.values.values:
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
                    for elem in self.values.values:
                        if isinstance(elem, tuple(exclude)):
                            index.append(elem._idx)

            p = np.full(self.size, 1 / (self.size - len(index)))
            p[index] = 0
        else:
            p = np.full(self.size, 1 / self.size)

        return np.random.choice(
            self.values.values, size=size, replace=replace, p=p
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

        return self.values.random(size)

    def subspace(self, low_bounds, up_bounds, **kwargs):

        """subspace(self, lo_bounds, up_bounds)

        Build a sub space according to the actual Searchspace using two vectors containing lower and upper bounds of the subspace.
        Can change type to Constant if necessary

        Parameters
        ----------

        lo_bounds : list
            Lower bounds of the subspace. See `Variable` for more info.
        up_bounds : boolean, default=False
            Upper bounds of the subspace. See `Variable` for more info.

        Returns
        -------

        out : Searchspace
            Return a subspace of the actual Searchspace.

        Examples
        --------

        """

        sub = self.values.subset(low_bounds, up_bounds)
        sp = type(self)(sub, self.loss, **kwargs, **self.kwargs)

        return sp

    def __len__(self):
        return len(self.values)


class MixedSearchspace(Searchspace):

    """MixedSearchspace

    :code:`MixedSearchspace` is a search space made for HyperParameter Optimization (HPO).
    The decision space can be made of various `Variable` types.

    Attributes
    ----------

    values : ArrayVar
        Determines the bounds of the search space.
        For `ContinuousSearchspace` the `values` must be an `ArrayVar`
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

    subspace(self,lo_bounds,up_bounds)
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
    def __init__(self, values, loss, **kwargs):

        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space.
            For :code:`ContinuousSearchspace` the :code:`values` must be an :code:`ArrayVar`
            of :code:`FloatVar`, :code:`IntVar`, :code:`CatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
            :code:`loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)


        """

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(values, ArrayVar), logger.error(
            f"`values` must be be an `ArrayVar`, got {values}"
        )

        self.distance = kwargs.pop("distance", Mixed())
        super(MixedSearchspace, self).__init__(values, loss, **kwargs)
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self

        self.loss.labels = [v.label for v in self.values]


class ContinuousSearchspace(Searchspace):

    """ContinuousSearchspace

    :code:`ContinuousSearchspace` is a search space made for continuous optimization.
    The decision space is made of `FloatVar`.

    Attributes
    ----------

    values : ArrayVar
        Determines the bounds of the search space.
        For :code:`ContinuousSearchspace` the :code:`values` must be an :code:`ArrayVar`
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

    subspace(self,lo_bounds,up_bounds)
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
    def __init__(self, values, loss, **kwargs):

        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `values` must be an `ArrayVar`
            of `FloatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)

        """
        super(ContinuousSearchspace, self).__init__(values, loss, **kwargs)

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(values, ArrayVar) and all(
            isinstance(v, FloatVar) for v in values
        ), logger.error(
            f"`values` must be be an `ArrayVar` of `FloatVar`, got {values}"
        )

        self.distance = kwargs.pop("distance", Euclidean(self))
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self


class DiscreteSearchspace(Searchspace):

    """DiscreteSearchspace

    :code:`DiscreteSearchspace` is a search space made for continuous optimization.
    The decision space is made of :code:`IntVar`.

    Attributes
    ----------

    values : ArrayVar
        Determines the bounds of the search space.
        For :code:`DiscreteSearchspace` the :code:`values` must be an :code:`ArrayVar`
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

    subspace(self,lo_bounds,up_bounds)
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
    def __init__(self, values, loss, **kwargs):

        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space.
            For :code:`DiscreteSearchspace` the :code:`values` must be an :code:`ArrayVar`
            of :code:`IntVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type :code:`LossFunc`. See :ref:`lf` for more information.
            :code:`loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        **kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Common addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
            * And other operators linked to the optimization algorithms (crossover, mutation,...)

        """
        super(DiscreteSearchspace, self).__init__(values, loss, **kwargs)

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(values, ArrayVar) and all(
            isinstance(v, IntVar) for v in values
        ), logger.error(
            f"`values` must be be an `ArrayVar` of `FloatVar`, got {values}"
        )

        self.distance = kwargs.pop("distance", Euclidean(self))
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self
