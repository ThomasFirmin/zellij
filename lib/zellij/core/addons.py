# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-05T16:18:04+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-02T11:40:04+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("zellij.addons")


class Addon(ABC):
    def __init__(self, object=None):
        self.target = object

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, object):
        self._target = object


class VarAddon(Addon):
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


class SearchSpaceAddon(Addon):
    def __init__(self, search_space=None):
        super(SearchSpaceAddon, self).__init__(search_space)

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


class Neighborhood(SearchSpaceAddon):
    def __init__(self, neighborhood, search_space=None):
        super(Neighborhood, self).__init__(search_space)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def get_neighbor(self, point, size=1):
        pass


class VarNeighborhood(VarAddon):
    def __init__(self, neighborhood, variable=None):
        super(VarAddon, self).__init__(variable)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def __call__(self, point, size=1):
        pass


class Converter(SearchSpaceAddon):
    def __init__(self, search_space=None):
        super(Converter, self).__init__(search_space)

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def reverse(self):
        pass


class VarConverter(VarAddon):
    def __init__(self, variable=None):
        super(VarConverter, self).__init__(variable)

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def reverse(self):
        pass


class Operator(SearchSpaceAddon):
    def __init__(self, search_space=None):
        super(Operator, self).__init__(search_space)

    @abstractmethod
    def __call__(self):
        pass


class Mutator(SearchSpaceAddon):
    def __init__(self, search_space=None):
        super(Mutator, self).__init__(search_space)


class Crossover(SearchSpaceAddon):
    def __init__(self, search_space=None):
        super(Crossover, self).__init__(search_space)


class Selector(SearchSpaceAddon):
    def __init__(self, search_space=None):
        super(Selector, self).__init__(search_space)


class Distance(SearchSpaceAddon):
    def __init__(self, search_space=None, weights=None):
        super(Distance, self).__init__(search_space)
        self.weights = None

    @abstractmethod
    def __call__(self, point_a, point_b):
        pass
