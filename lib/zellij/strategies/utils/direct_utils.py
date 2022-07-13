# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-06-17T18:23:26+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-22T12:37:10+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin

import numpy as np
from abc import ABC, abstractmethod


class Direct_size(ABC):
    @abstractmethod
    def __call__(self, hyperrectangle):
        pass


class Sigma2(Direct_size):
    def __init__(self, size):
        assert (
            size > 0
        ), f"""
        Size must be > 0, for Sigma2, got {size}
        """
        self.size = size

    def __call__(self, hyperrectangle):
        return round(
            hyperrectangle.width
            * (self.size - 8 / 9 * hyperrectangle.stage) ** (1 / 2),
            13,
        )


class SigmaInf(Direct_size):
    def __call__(self, hyperrectangle):
        return round(hyperrectangle.width / 2, 13)
