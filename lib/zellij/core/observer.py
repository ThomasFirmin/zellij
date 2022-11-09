# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-11-08T14:43:01+01:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-08T16:46:29+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


class AbstractObserver(ABC):
    """
    Abstract base class for state variable monitors.
    """


class Observer(AbstractMonitor):
    """
    Records state variables of interest.
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


class Monitor(AbstractMonitor):
    """
    Records state variables of interest.
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
