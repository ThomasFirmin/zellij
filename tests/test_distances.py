# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from zellij.core import (
    IntVar,
    FloatVar,
    CatVar,
    ArrayVar,
    Constant,
    MixedSearchspace,
    ContinuousSearchspace,
    Loss,
    MockModel,
)
from zellij.utils import Euclidean, Manhattan, Mixed
import numpy as np


class TestEuclidean(unittest.TestCase):
    def setUp(self):
        self.values = ArrayVar(
            FloatVar("float_1", 0, 5),
            FloatVar("float_2", 0, 5),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = ContinuousSearchspace(
            self.values, self.loss, distance=Euclidean()
        )

        self.p1 = ([0, 1], [3, 5])

    def test_creation(self):

        self.assertTrue(hasattr(self.sp, "distance"))
        self.assertIsInstance(self.sp.distance, Euclidean)

    def test_call(self):
        dist1 = self.sp.distance(self.p1[0], self.p1[1])
        dist2 = self.sp.distance(self.p1[1], self.p1[0])
        self.assertEqual(dist1, dist2)
        self.assertEqual(dist1, 5.0)


class TestManhattan(unittest.TestCase):
    def setUp(self):
        self.values = ArrayVar(
            FloatVar("float_1", 0, 5),
            FloatVar("float_2", 10, 15),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = ContinuousSearchspace(
            self.values, self.loss, distance=Manhattan()
        )

        self.p1 = ([0, 10], [5, 15])

    def test_creation(self):

        self.assertTrue(hasattr(self.sp, "distance"))
        self.assertIsInstance(self.sp.distance, Manhattan)

    def test_call(self):
        dist1 = self.sp.distance(self.p1[0], self.p1[1])
        dist2 = self.sp.distance(self.p1[1], self.p1[0])
        self.assertEqual(dist1, dist2)
        self.assertEqual(dist1, 10.0)


class TestMixed(unittest.TestCase):
    def setUp(self):
        self.values = ArrayVar(
            IntVar("int_1", 0, 10),
            IntVar("int_2", 20, 30),
            FloatVar("float_1", 2, 12),
            CatVar("cat_1", ["Hello", 87, 2.56]),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = MixedSearchspace(self.values, self.loss, distance=Mixed())

        self.p1 = ([0, 20, 1.3, 87], [2, 25, 4.3, "Hello"])

    def test_creation(self):

        self.assertTrue(hasattr(self.sp, "distance"))
        self.assertIsInstance(self.sp.distance, Mixed)

    def test_call(self):
        dist1 = self.sp.distance(self.p1[0], self.p1[1])
        dist2 = self.sp.distance(self.p1[1], self.p1[0])
        print(dist1, dist2)
        self.assertEqual(dist1, dist2)
        self.assertEqual(dist1, 0.38)


if __name__ == "__main__":
    unittest.main()
