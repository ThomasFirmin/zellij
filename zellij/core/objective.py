# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, List, Callable
import os
import numpy as np

from zellij.core.errors import TargetError

import logging

logger = logging.getLogger("zellij.objective")


class Objective(ABC):
    """Objective

    This absract object allows to define what is the objective of
    the optimization process.

    Parameters
    ----------
    target : str
        Key of the :ref:`lf` :code:`outputs` to consider as the objective.

    Attributes
    ----------
    target

    """

    @abstractmethod
    def __init__(self, target: Union[str, List[str]]):
        pass

    @abstractmethod
    def _compute_obj(self, outputs: dict):
        pass

    def __call__(self, outputs: dict, num=0) -> dict:
        """__call__

        Add the objective value to the outputs.

        Parameters
        ----------
        outputs : int, float, list, dict
            Outputs of the loss function.
        num : int, default=0
            Objective id.

        Returns
        -------
        dict
            Outputs

        """

        outputs[f"objective{num}"] = self._compute_obj(outputs)
        return outputs


class SingleObjective(Objective):
    """SingleObjective

    Objective made of a single target

    Parameters
    ----------
    target : str
        Key to consider as objective from the :code:`outputs` of :ref:`lf`.

    Attribute
    ---------
    target

    """

    def __init__(self, target: str):
        if isinstance(target, str):
            self.target = target
        else:
            raise TargetError(f"Unknown target type got, {target}")


class MultiObjective(Objective):
    """MultiObjective

    Objective made of multiple outputs from the :ref:`lf`.

    Parameters
    ----------
    target : list[str]
        List of keys to consider as objectives from the :code:`outputs` of :ref:`lf`.
    """

    def __init__(self, target: List[str]):
        for t in target:
            if not isinstance(t, str):
                raise TargetError(f"Unknown target type got, {t} in {target}")

        self.target = target


class Minimizer(SingleObjective):
    """Minimizer

    Minimizer allows to minimize the given target.
    Do, :math:`f(y)=y`. With :math:`y` a given scores.
    By default Zellij metaheuristics minimize the loss value.
    So this object simply returns the given scores.

    Parameters
    ----------
    target : str
        Key of the :ref:`lf` :code:`outputs` to consider as the objective.

    Attributes
    ----------
    target

    """

    def _compute_obj(self, outputs: dict) -> float:
        try:
            res = outputs[self.target]
        except KeyError:
            raise TargetError(
                f"Target error in objective, no {self.target} in returned outputs."
            )
        return res


class Maximizer(SingleObjective):
    """Maximizer

    Maximizer allows to maximize the given target.
    Do, :math:`f(y)=-y`. With :math:`y` a given scores.
    By default Zellij metaheuristics minimize the loss value.
    So this object will negate the given scores.

    Parameters
    ----------
    target : int or str, default=0
        Key of the :ref:`lf` :code:`outputs` to consider as the objective.

    Attributes
    ----------
    target

    """

    def _compute_obj(self, outputs: dict) -> float:
        try:
            res = -outputs[self.target]
        except KeyError:
            raise TargetError(
                f"Target error in objective, no {self.target} in returned outputs."
            )
        return res


class Lambda(MultiObjective):
    """Lambda

    Lambda allows to transform the given target.
    Do, :math:`f(y)=function(y)`. With :math:`y` a given scores.
    By default Zellij metaheuristics minimize the loss value.

    Parameters
    ----------
    function : Callable
        Function with `len(target)` parameters which return an objective value.
    target : list[str]
        List of keys linked to the outputs of the :ref:`lf`.
        Selected targets will be passed to :code:`function` with the same order as :code:`target` list.

    Attributes
    ----------
    target

    """

    def __init__(self, target: List[str], function: Callable):
        super().__init__(target)

        if function.__code__.co_argcount != len(self.target):
            raise TargetError(
                f"""
                    Number of parameters of `function` must be equal to
                    the length of `target`,
                    got {function.__code__.co_argcount} != {len(self.target)}
                    """
            )

        self.function = function

    def _compute_obj(self, outputs: dict) -> float:
        parameter = [outputs[t] for t in self.target]
        res = self.function(*parameter)
        return res


class DoNothing(Minimizer):
    pass
