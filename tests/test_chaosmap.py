# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from zellij.strategies.tools import Henon, Kent, Logistic, Tent, Random
import numpy as np


class TestHenon(unittest.TestCase):
    def test_creation(self):
        Henon(10, 2)


class TestKent(unittest.TestCase):
    def test_creation(self):
        Kent(10, 2)


class TestLogistic(unittest.TestCase):
    def test_creation(self):
        Logistic(10, 2)


class TestTent(unittest.TestCase):
    def test_creation(self):
        Tent(10, 2)


class TestRandom(unittest.TestCase):
    def test_creation(self):
        Random(10, 2)


if __name__ == "__main__":
    unittest.main()
