# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod

from zellij.core.errors import InitializationError
from typing import Optional, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.variables import (
        Variable,
        IntVar,
        FloatVar,
        CatVar,
        ArrayVar,
        PermutationVar,
    )
    from zellij.core.search_space import Searchspace


import logging

logger = logging.getLogger("zellij.addons")

####################
# ABSTRACT GENERAL #
####################


class Addon(ABC):
    """Addon

    Abstract class describing what an addon is.
    An :code:`Addon` in Zellij, is an additionnal feature that can be added to a
    :code:`target` object. See :ref:`varadd` for addon targeting :ref:`var` or
    :ref:`spadd` targeting :ref:`sp`.

    Attributes
    ----------
    target : Object, default=None
        Object targeted by the addons.

    """

    def __init__(self):
        self._target = None

    @property
    def target(self) -> object:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: object):
        self._target = value


class SearchspaceAddon(Addon):
    """
    :ref:`addons` where the target must be of type :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self):
        super(SearchspaceAddon, self).__init__()

    @property
    def target(self) -> Searchspace:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: Searchspace):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. Searchspace expected."
            )


class VarAddon(Addon):
    """
    :ref:`addons` where the target must be of type :ref:`var`.

    Attributes
    ----------
    target : :ref:`var`, default=None
        :ref:`var` targeted by the addons

    """

    def __init__(self):
        super(VarAddon, self).__init__()

    @property
    def target(self) -> Variable:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: Variable):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. Variable expected."
            )


#########################
# ABSTRACT VAR SPECIFIC #
#########################


class IntAddon(VarAddon):
    @property
    def target(self) -> IntVar:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: IntVar):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. IntVar expected."
            )


class FloatAddon(VarAddon):
    @property
    def target(self) -> FloatVar:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: FloatVar):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. FloatVar expected."
            )


class CatAddon(VarAddon):
    @property
    def target(self) -> CatVar:
        if self._target is None:
            raise InitializationError("The target is not initialized.")
        return self._target

    @target.setter
    def target(self, value: CatVar):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. CatVar expected."
            )


class ArrayAddon(VarAddon):
    @property
    def target(self) -> ArrayVar:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: ArrayVar):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. ArrayVar expected."
            )


class PermutationAddon(VarAddon):
    @property
    def target(self) -> PermutationVar:
        if self._target:
            return self._target
        else:
            raise InitializationError("The target is not initialized.")

    @target.setter
    def target(self, value: PermutationVar):
        if value:
            self._target = value
        else:
            raise InitializationError(
                f"The target value cannot be {type(value)}. ArrayVar expected."
            )


################
# Neighborhood #
################


class VarNeighborhood(VarAddon):
    """
    :ref:`addons` where the target must be of type :ref:`var`.
    Describes what a neighborhood is for a :ref:`var`.

    Attributes
    ----------
    neighborhood : object, optional
        Defines what is the neighborhood of the targetted variable.

    """

    def __init__(self, neighborhood: Optional[object] = None):
        super(VarAddon, self).__init__()
        self.neighborhood = neighborhood

    @property
    @abstractmethod
    def neighborhood(self):
        pass

    @neighborhood.setter
    @abstractmethod
    def neighborhood(self, value):
        pass

    @abstractmethod
    def __call__(
        self, point: object, size: Optional[int] = None
    ) -> Union[List, List[list]]:
        pass


class IntNeighborhood(VarNeighborhood, IntAddon):
    """
    Describes what a converter is for a :ref:`IntVar`.
    """

    def __init__(self, neighborhood: Optional[object] = None):
        super().__init__(neighborhood)


class FloatNeighborhood(VarNeighborhood, FloatAddon):
    """
    Describes what a converter is for a :ref:`FloatVar`.
    """

    def __init__(self, neighborhood: Optional[object] = None):
        super().__init__(neighborhood)


class CatNeighborhood(VarNeighborhood, CatAddon):
    """
    Describes what a converter is for a :ref:`CatVar`.
    """

    def __init__(self, neighborhood: Optional[object] = None):
        super().__init__(neighborhood)


class ArrayNeighborhood(VarNeighborhood, ArrayAddon):
    """
    Describes what a converter is for a :ref:`ArrayVar`.
    """

    def __init__(self, neighborhood: Optional[object] = None):
        super().__init__(neighborhood)


class PermutationNeighborhood(VarNeighborhood, PermutationAddon):
    """
    Describes what a converter is for a :ref:`PermutationVar`.
    """

    def __init__(self, neighborhood: Optional[object] = None):
        super().__init__(neighborhood)


##############
# CONVERTERS #
##############


class VarConverter(VarAddon):
    """
    Describes what a converter is for a :ref:`var`.
    Converter allows to convert values from a :ref:`var` to another type.
    """

    @abstractmethod
    def convert(self, value: object) -> object:
        pass

    @abstractmethod
    def reverse(self, value: object) -> object:
        pass


class IntConverter(VarConverter, IntAddon):
    """
    Describes what a converter is for a :ref:`IntVar`.
    """

    def __init__(self):
        super().__init__()


class FloatConverter(VarConverter, FloatAddon):
    """
    Describes what a converter is for a :ref:`FloatVar`.
    """

    def __init__(self):
        super().__init__()


class CatConverter(VarConverter, CatAddon):
    """
    Describes what a converter is for a :ref:`CatVar`.
    """

    def __init__(self):
        super().__init__()


class ArrayConverter(VarConverter, ArrayAddon):
    """
    Describes what a converter is for a :ref:`ArrayVar`.
    """

    def __init__(self):
        super().__init__()


class PermutationConverter(VarConverter, PermutationAddon):
    """
    Describes what a converter is for a :ref:`PermutationVar`.
    """

    def __init__(self):
        super().__init__()


#############
# DISTANCES #
#############


class Distance(SearchspaceAddon):
    """
    Abstract class describing what an Distance is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    weights : list[float], default=None
        List of floats giving weights for each feature of the :ref:`sp`
    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
    ):
        super(Distance, self).__init__()
        self.weights = weights

    @abstractmethod
    def __call__(self, point_a, point_b) -> float:
        pass


############
# OPERATOR #
############


class Operator(SearchspaceAddon):
    """
    Abstract class describing what an operator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self):
        super(Operator, self).__init__()

    @abstractmethod
    def _build(self, toolbox):
        pass

    @abstractmethod
    def __call__(self):
        pass


class Mutation(Operator):
    """
    Abstract class describing what an Mutation is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self):
        super(Mutation, self).__init__()

    @abstractmethod
    def __call__(self, individual):
        pass


class Crossover(Operator):
    """
    Abstract class describing what an MCrossover is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self):
        super(Crossover, self).__init__()

    @abstractmethod
    def __call__(self, children1, children2):
        pass


class Selection(Operator):
    """
    Abstract class describing what an Selection is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self):
        super(Selection, self).__init__()

    @abstractmethod
    def __call__(self, population, k) -> list:
        pass
