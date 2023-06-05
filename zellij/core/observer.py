# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-11-08T14:43:01+01:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T12:45:53+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from abc import ABC


class AbstractObserver(ABC):
    """
    Abstract base class for state variable monitors.
    """


class Observer(AbstractObserver):
    """Observer

    Records targeted state variables.
    There is no historic. When the value of the targeted attribute changes,
    it overwrites its current record.

    Parameters
    ----------
    target : Object
        Targetet object.
    attributes : str or list[str]
        Name of the attribute to record.

    Attributes
    ----------
    records : dict
        Dictionnary of shape {attribute:value}.
        Record the values of the targeted attributes. There is no historic.
        Values are overwrote when the value of the attribute change
    target
    attributes

    """

    def __init__(
        self,
        target,
        attributes,
    ):
        super().__init__()

        self.target = target
        self.attributes = attributes

        self.records = {a: None for a in attributes}

    def get(self, attribute):
        return self.records[attribute]

    def update(self):
        for a in self.attributes:
            self.records[a] = getattr(self.target, a)


class Monitor(AbstractObserver):
    """Monitor

    Records state variables of interest.
    There is a historic. When the value of the targeted attribute changes,
    its new value is appended to the record.

    Parameters
    ----------
    target : Object
        Targetet object.
    attributes : str or list[str]
        Name of the attribute to record.

    Attributes
    ----------
    records : dict
        Dictionnary of shape {attribute:[values]}.
        Record the values of the targeted attributes. There is historic.
        Values are appended when the value of the attribute change.
    target
    attributes

    """

    def __init__(
        self,
        target,
        attributes,
    ):
        super().__init__()

        self.target = target
        self.attributes = attributes

        self.records = {a: [] for a in attributes}

    def get(self, attribute):
        return self.records[attribute]

    def update(self):
        for a in self.attributes:
            self.records[a].append(getattr(self.target, a))
