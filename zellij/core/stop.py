# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import time


class Stopping(ABC):
    """Stopping

    Callable describing what a stopping criterion is for a :ref:`meta`.
    Stopping object can be combined using :code:`&`, :code:`|`, :code:`^`,
    :code:`&=`, :code:`|=` and :code:`^=` operators.

    Parameters
    ----------
    target : object
        Targeted object.
    attribute : str
        Name of the attribute of :code:`target` to watch.



    Attributes
    ----------
    target
    attribute

    """

    def __init__(self, target, attribute):
        self.target = target
        self.attribute = attribute

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    @property
    def attribute(self):
        return self._attribute

    @attribute.setter
    def attribute(self, value):
        self._attribute = value

    @abstractmethod
    def __call__(self) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __and__(self, other):
        return Combined(self, other, lambda a, b: a() & b(), "&")

    def __or__(self, other):
        return Combined(self, other, lambda a, b: a() | b(), "|")

    def __xor__(self, other):
        return Combined(self, other, lambda a, b: a() ^ b(), "^")

    def __rand__(self, other):
        return Combined(other, self, lambda a, b: a() & b(), "&")

    def __ror__(self, other):
        return Combined(other, self, lambda a, b: a() | b(), "|")

    def __rxor__(self, other):
        return Combined(other, self, lambda a, b: a() ^ b(), "^")

    def __iand__(self, other):
        return Combined(self, other, lambda a, b: a() & b(), "&")

    def __ior__(self, other):
        return Combined(self, other, lambda a, b: a() | b(), "|")

    def __ixor__(self, other):
        return Combined(self, other, lambda a, b: a() ^ b(), "^")


class Combined(Stopping):
    """Combined

    Combine two :ref:`stop` :code:`a` and :code:`b`, with a given callable
    :code:`op` which takes :code:`a` and `b` as parameters.

    Parameters
    ----------
    a : Stopping
        Stopping criterion 2.
    b : Stopping
        Stopping criterion 1.
    op : Callable
        Callable that takes :code:`a` and :code:`b` as parameters and return a
        boolean.

    Attributes
    ----------
    a
    b
    op

    """

    def __init__(self, a: Stopping, b: Stopping, op: Callable, str_op: str):
        # Stopping 1
        self.a = a
        # Stopping 2
        self.b = b
        # operation between a and b
        self.op = op
        self.str_op = str_op

    def __call__(self):
        return self.op(self.a, self.b)

    def __str__(self) -> str:
        return f"({self.a}{self.str_op}{self.b})"


class Threshold(Stopping):
    """Threshold

    Stoppping criterion based on a budget. If the observed value
    exceed a threshold, then it returns False.

    Parameters
    ----------
    threshold : int
        Int describing the maximum value
        that the observed value should not cross.

    Attributes
    ----------
    threshold
    """

    def __init__(self, target, attribute, threshold):
        super(Threshold, self).__init__(target, attribute)
        self.threshold = threshold

    def __call__(self):
        return getattr(self.target, self.attribute) >= self.threshold

    def __str__(self) -> str:
        return f"|T|{self.attribute}:{getattr(self.target, self.attribute)}>={self.threshold}"


class IThreshold(Stopping):
    """IThreshold

    Stoppping criterion based on a budget. If the observed value
    is below a threshold, then it returns False.

    Parameters
    ----------
    threshold : int
        Int describing the maximum value
        that the observed value should not cross.

    Attributes
    ----------
    threshold
    """

    def __init__(self, target, attribute, threshold):
        super(IThreshold, self).__init__(target, attribute)
        self.threshold = threshold

    def __call__(self):
        return getattr(self.target, self.attribute) <= self.threshold  # type: ignore

    def __str__(self) -> str:
        return f"|T|{self.attribute}:{getattr(self.target, self.attribute)}<={self.threshold}"


class Calls(Threshold):
    """Calls

    Stoppping criterion based on a budget. If the numbers of calls to the
    :ref:`lf` exceed a threshold, then it returns False.

    Parameters
    ----------
    loss : LossFunc
        A :ref:`lf`.
    threshold : int
        Int describing the maximum calls to the :ref:`lf`.

    Attributes
    ----------
    threshold

    """

    def __init__(self, loss, threshold):
        super(Calls, self).__init__(loss, "calls", threshold)


class BooleanStop(Stopping):
    """BooleanStop

    Stoppping criterion based on a boolean attribute.
    Return the value of the targetted attribute.

    Attributes
    ----------
    threshold
    """

    def __call__(self):
        return getattr(self.target, self.attribute)  # type: ignore

    def __str__(self) -> str:
        return f"|B|{self.attribute}:{getattr(self.target, self.attribute)}"


class Convergence(Stopping):
    """Convergence

    Stoppping criterion based on convergence. If the targetted attribute
    does not change after :code:`patience` calls to the :ref:`lf`, then
    returns false.

    Parameters
    ----------
    patience : int
        Int describing the maximum number of calls to the :ref:`lf`
        without any change in the best loss value found so far.

    Attributes
    ----------
    patience

    """

    def __init__(self, target, attribute, patience):
        super(Convergence, self).__init__(target, attribute)
        self.patience = patience

        # accumulation
        self.acc = 0
        self.previous = None

    def __call__(self):
        if self.previous != getattr(self.target, self.attribute):  # type: ignore
            self.previous = getattr(self.target, self.attribute)  # type: ignore
        else:
            self.acc += 1

        return self.acc >= self.patience

    def __str__(self) -> str:
        return f"|P|{self.attribute}:{self.acc}>={self.patience}"


class Time(Stopping):
    """Time

    Stoppping criterion based on time in seconds.

    Parameters
    ----------
    ttime : float
        Total time of the experiment.

    Attributes
    ----------
    threshold
    """

    def __init__(self, ttime, target=None, attribute=""):
        super(Time, self).__init__(target, attribute)
        self.ttime = ttime
        self.start_time = time.time()

    def __call__(self):
        elapsed = time.time() - self.start_time
        return elapsed >= self.ttime

    def __str__(self) -> str:
        elapsed = time.time() - self.start_time
        return f"|T|Elapsed Time:{elapsed}>={self.ttime}"
