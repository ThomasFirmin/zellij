# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-05T16:18:04+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T23:04:17+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("zellij.addons")


class Addon(ABC):
    """Addon

    Abstract class describing what an addon is.
    An :code:`Addon` in Zellij, is an additionnal feature that can be added to a
    :code:`target` object. See :ref:`varadd` for addon targeting :ref:`var` or
    :ref:`spadd` targeting :ref:`sp`.

    Parameters
    ----------
    target : Object, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : Object, default=None
        Object targeted by the addons

    """

    def __init__(self, object=None):
        self.target = object

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, object):
        self._target = object


class VarAddon(Addon):
    """VarAddon

    :ref:`addons` where the target must be of type :ref:`var`.

    Parameters
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    """

    def __init__(self, variable=None):
        super(VarAddon, self).__init__(variable)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, variable):
        from zellij.core.variables import Variable

        if variable:
            assert isinstance(variable, Variable), logger.error(
                f"Object must be a `Variable` for {self.__class__.__name__}, got {variable}"
            )

        self._target = variable


class SearchspaceAddon(Addon):
    """SearchspaceAddon

    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(SearchspaceAddon, self).__init__(search_space)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, search_space):
        from zellij.core.search_space import Searchspace

        if search_space:
            assert isinstance(search_space, Searchspace), logger.error(
                f"Object must be a `Searchspace` for {self.__class__.__name__},\
                 got {search_space}"
            )

        self._target = search_space


class Neighborhood(SearchspaceAddon):
    """Neighborhood

    :ref:`addons` where the target must be of type :ref:`sp`.
    Describes what a neighborhood is for a :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, neighborhood, search_space=None):
        super(Neighborhood, self).__init__(search_space)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def __call__(self, point, size=1):
        pass


class VarNeighborhood(VarAddon):
    """VarNeighborhood

    :ref:`addons` where the target must be of type :ref:`var`.
    Describes what a neighborhood is for a :ref:`var`.

    Parameters
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    """

    def __init__(self, neighborhood, variable=None):
        super(VarAddon, self).__init__(variable)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def __call__(self, point, size=1):
        pass


class Converter(SearchspaceAddon):
    """Converter

    :ref:`addons` where the target must be of type :ref:`sp`.
    Describes what a converter is for a :ref:`sp`.
    Converter allows to convert the type of a :ref:`sp` to another one.
    All :ref:`var` must have a converter :ref:`varadd` implemented.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Converter, self).__init__(search_space)

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def reverse(self):
        pass


class VarConverter(VarAddon):
    """VarConverter

    :ref:`addons` where the target must be of type :ref:`var`.
    Describes what a converter is for a :ref:`var`.
    Converter allows to convert the type of a :ref:`var` to another one.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, variable=None):
        super(VarConverter, self).__init__(variable)

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def reverse(self):
        pass


class Operator(SearchspaceAddon):
    """Operator

    Abstract class describing what an operator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Operator, self).__init__(search_space)

    @abstractmethod
    def __call__(self):
        pass


class Mutator(SearchspaceAddon):
    """Mutator

    Abstract class describing what an Mutator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Mutator, self).__init__(search_space)


class Crossover(SearchspaceAddon):
    """Crossover

    Abstract class describing what an MCrossover is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Crossover, self).__init__(search_space)


class Selector(SearchspaceAddon):
    """Selector

    Abstract class describing what an Selector is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Selector, self).__init__(search_space)


class Distance(SearchspaceAddon):
    """Distance

    Abstract class describing what an Distance is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, search_space=None, weights=None):
        super(Distance, self).__init__(search_space)
        self.weights = None

    @abstractmethod
    def __call__(self, point_a, point_b):
        pass
