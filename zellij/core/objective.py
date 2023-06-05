# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-11-08T16:57:09+01:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-09T14:17:02+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from abc import ABC, abstractmethod
import os
import numpy as np

import logging

logger = logging.getLogger("zellij.objective")


class Objective(ABC):
    """Objective

    This absract object allows to define what is the objective of
    the optimization process.

    Parameters
    ----------
    target : int or str, default=0
        Which outputs of the loss function should it target.
        Default is 0, it will consider the value at index 0 in the outputs of
        the loss function. If output is a dict, it can target one of its key.

    Attributes
    ----------
    target

    """

    def __init__(self, target=0):
        if isinstance(target, str):
            self.target = [target]
        elif isinstance(target, int):
            self.target = [target]
        elif isinstance(target, list) and (
            all(isinstance(i, int) for i in target)
            or all(isinstance(i, str) for i in target)
        ):
            self.target = target
        else:
            raise AssertionError(f"Unknown target type got, {target}")
        self.index_built = False

    @abstractmethod
    def __call__(self, outputs):
        """__call__

        Add the objective value to the outputs.

        Parameters
        ----------
        outputs : int, float, list, dict
            Outputs of the loss function.

        Returns
        -------
        dict
            Outputs

        """
        pass

    @abstractmethod
    def _select(self, X, Y, Q, *args, **kwargs):
        """_select

        Define how to select solution according to their associated objective
        value.

        Parameters
        ----------
        X : list[solutions]
            List of solutions
        Y : list[{float,int}]
            List of loss values associated to X.
        Q : int
            Number of solution to select according to the objective.

        Returns
        -------
        dict
            Outputs

        """
        pass

    def _cleaner(self, outputs):
        """__call__

        Parameters
        ----------
        outputs : int, float, list, dict
            Outputs of the loss function.

        Returns
        -------
        dict
            Cleaned outputs

        """
        rd = {}
        # Separate results
        if isinstance(outputs, int) or isinstance(outputs, float):
            rd["objective"] = outputs
        elif isinstance(outputs, dict):
            rd = outputs
        elif isinstance(outputs, list):
            rd = {f"r{i}": j for i, j in enumerate(outputs)}

        return rd

    def _build_index(self, outputs):
        if not self.index_built:
            for i, t in enumerate(self.target):
                if isinstance(t, int):
                    self.target[i] = list(outputs.keys())[t]
            self.index_built = True

    def reset(self):
        self.index_built = False


class Minimizer(Objective):
    """Minimizer

    Minimizer allows to minimize the given target.
    Do, :math:`f(y)=y`. With :math:`y` a given scores.
    /!\ By default Zellij metaheuristics minimize the loss value.
    So this object will just return the given scores.

    Parameters
    ----------
    target : int or str, default=0
        Which outputs of the loss function should it target.
        Default is 0, it will consider the value at index 0 in the outputs of
        the loss function. If output is a dict, it can target one of its key.

    Attributes
    ----------
    target

    """

    def _select(self, X, Y, Q, *args, **kwargs):
        """_select

        Define how to select solution by minimizing the objective value.

        Parameters
        ----------
        X : list[solutions]
            List of solutions
        Y : list[{float,int}]
            List of loss values associated to X.
        Q : int
            Number of solution to select according to the objective.

        Returns
        -------
        dict
            Outputs

        """
        index = np.argsort(Y)
        new_x = [X[i] for i in index[:Q]]
        return new_x, Y[:Q]

    def __call__(self, outputs):
        clean = self._cleaner(outputs)
        self._build_index(clean)
        clean["objective"] = clean[self.target[0]]
        return clean


class Maximizer(Objective):
    """Maximizer

    Maximizer allows to maximize the given target.
    Do, :math:`f(y)=-y`. With :math:`y` a given scores.
    /!\ By default Zellij metaheuristics minimize the loss value.
    So this object will compute the negative of the given scores.

    Parameters
    ----------
    target : int or str, default=0
        Which outputs of the loss function should it target.
        Default is 0, it will consider the value at index 0 in the outputs of
        the loss function. If output is a dict, it can target one of its key.

    Attributes
    ----------
    target

    """

    def _select(self, X, Y, Q, *args, **kwargs):
        """_select

        Define how to select solution by maximizing the objective value.

        Parameters
        ----------
        X : list[solutions]
            List of solutions
        Y : list[{float,int}]
            List of loss values associated to X.
        Q : int
            Number of solution to select according to the objective.

        Returns
        -------
        dict
            Outputs

        """
        index = np.argsort(Y)
        new_x = [X[-i] for i in index[-Q:]]
        return new_x, Y[-Q:]

    def __call__(self, outputs):
        clean = self._cleaner(outputs)
        self._build_index(clean)

        clean["objective"] = -clean[self.target[0]]

        return clean


class Lambda(Objective):
    """Lambda

    Lambda allows to transform the given target.
    Do, :math:`f(y)=function(y)`. With :math:`y` a given scores.
    /!\ By default Zellij metaheuristics minimize the loss value.

    Parameters
    ----------
    function : Callable
        Function with `len(target)` parameters which return an objective value.
    selector : {"min","max"}
        Minimize or maximize the results from `function`
    target : {int,str,list[{int, str}]} default=0
        Which outputs of the loss function should it target.
        Default is 0, it will consider the value at index 0 in the outputs of
        the loss function. If output is a dict, it can target one of its key.

    Attributes
    ----------
    target

    """

    def __init__(self, function, selector="min", target=0):
        super().__init__(target)

        if function.__code__.co_argcount != len(self.target):
            raise AssertionError(
                logger.error(
                    f"""
                    Number of parameters of `function` must be equal to
                    the length of `target`,
                    got {function.__code__.co_argcount} != {len(self.target)}
                    """
                )
            )

        self.function = function
        self.selector = selector

    def _select(self, X, Y, Q, *args, **kwargs):
        """_select

        Define how to select solution by maximizing or minimizing
        the objective value.

        Parameters
        ----------
        X : list[solutions]
            List of solutions
        Y : list[{float,int}]
            List of loss values associated to X.
        Q : int
            Number of solution to select according to the objective.

        Returns
        -------
        dict
            Outputs

        """
        if self.selector == "min":
            index = np.argsort(Y)
            new_x = [X[i] for i in index[:Q]]
            return new_x, Y[:Q]
        elif self.selector == "max":
            index = np.argsort(Y)
            new_x = [X[-i] for i in index[-Q:]]
            return new_x, Y[-Q:]

    def _build_index(self, outputs):
        if not self.index_built:
            if isinstance(self.target, list):
                for i, t in enumerate(self.target):
                    if isinstance(t, int):
                        self.target[i] = list(outputs.keys())[self.target[i]]
            self.index_built = True

    def __call__(self, outputs):
        clean = self._cleaner(outputs)
        self._build_index(clean)
        parameter = [clean[t] for t in self.target]
        for t in self.target:
            clean["objective"] = self.function(*parameter)

        if self.selector == "max":
            clean["objective"] = -clean["objective"]

        return clean
