from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("zellij.Addons")


class Addons(ABC):
    def __init__(self, search_space):
        self.search_space = search_space
        self.search_space._add_addon(self)

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        self._search_space = search_space

    def delete(self):
        key = f"{self.__class__.__bases__[0].__name__}".lower()
        if hasattr(self, key):
            logger.info(f"{key} will be deleted")
            delattr(self.search_space, key)
        else:
            logger.warning(f"{key} is not implemented this search space")


class Neighborhood(Addons):
    def __init__(self, search_space, neighborhood):
        super(Neighborhood, self).__init__(search_space)
        self.neighborhood = neighborhood

    @property
    def neighborhood(self):
        return self._neighborhood

    @abstractmethod
    def get_neighbor(self, point, size=1):
        pass


class Converter(Addons):
    def __init__(self, search_space):
        super(Converter, self).__init__(search_space)

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def reverse(self):
        pass


class Operator(Addons):
    def __init__(self, search_space):
        super(Operator, self).__init__(search_space)


class Mutator(Operator):
    def __init__(self, search_space):
        super(Mutator, self).__init__(search_space)


class Crossover(Operator):
    def __init__(self, search_space):
        super(Crossover, self).__init__(search_space)


class Selection(Operator):
    def __init__(self, search_space):
        super(Selection, self).__init__(search_space)


class Distance(Addons):
    def __init__(self, search_space):
        super(Distance, self).__init__(search_space)

    @abstractmethod
    def __call__(self, point_a, point_b):
        pass
