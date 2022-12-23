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
    Loss,
    MockModel,
)
from zellij.utils import (
    ArrayInterval,
    FloatInterval,
    IntInterval,
    CatInterval,
    ConstantInterval,
    Intervals,
)
import numpy as np


class TestSP(unittest.TestCase):
    def setUp(self):
        self.intvar = IntVar("int", -5, 5, neighbor=IntInterval(1))
        self.floatvar = FloatVar("float", -5, 5, neighbor=FloatInterval(1))
        self.catvar = CatVar("cat", ["I", 1, 5], neighbor=CatInterval())
        self.consvar = Constant("constant", 5, neighbor=ConstantInterval())
        self.arrayvar = ArrayVar(
            self.intvar,
            self.floatvar,
            self.catvar,
            self.consvar,
            label="array",
            neighbor=ArrayInterval(),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = MixedSearchspace(
            self.arrayvar, self.loss, neighbor=Intervals()
        )

        self.p1 = [-5, -5, "I", 5]
        self.p2 = [0, 0, 1, 5]
        self.p3 = [5, 5, 5, 5]

    def test_creation(self):

        self.assertTrue(hasattr(self.intvar, "neighbor"))
        self.assertTrue(hasattr(self.floatvar, "neighbor"))
        self.assertTrue(hasattr(self.catvar, "neighbor"))
        self.assertTrue(hasattr(self.consvar, "neighbor"))
        self.assertTrue(hasattr(self.arrayvar, "neighbor"))
        self.assertTrue(hasattr(self.sp, "neighbor"))

        self.assertIsInstance(self.intvar.neighbor, IntInterval)
        self.assertIsInstance(self.floatvar.neighbor, FloatInterval)
        self.assertIsInstance(self.catvar.neighbor, CatInterval)
        self.assertIsInstance(self.consvar.neighbor, ConstantInterval)
        self.assertIsInstance(self.arrayvar.neighbor, ArrayInterval)
        self.assertIsInstance(self.sp.neighbor, Intervals)

    def test_float(self):

        # Single
        n = self.floatvar.neighbor(0)
        self.assertTrue(-1 <= n <= 1)

        # Middle
        n = self.floatvar.neighbor(0, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(-1 <= e <= 1)

        # Upper
        n = self.floatvar.neighbor(5, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(4 <= e <= 5)

        # Lower
        n = self.floatvar.neighbor(-5, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(-5 <= e <= -4)

    def test_int(self):
        # Single
        n = self.intvar.neighbor(0)
        self.assertTrue(-1 <= n <= 1)

        # Middle
        n = self.intvar.neighbor(0, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(-1 <= e <= 1)

        # Upper
        n = self.intvar.neighbor(5, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(4 <= e <= 5)

        # Lower
        n = self.intvar.neighbor(-5, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(-5 <= e <= -4)

    def test_cat(self):

        # Single
        n = self.catvar.neighbor("I")
        self.assertTrue(n in self.catvar.features)

        n = self.catvar.neighbor(0, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertTrue(e in self.catvar.features)

    def test_cons(self):
        # Single
        n = self.consvar.neighbor(5)
        self.assertEqual(n, 5)

        n = self.consvar.neighbor(0, 10)
        self.assertEqual(len(n), 10)
        for e in n:
            self.assertEqual(e, 5)

    def test_array(self):

        # Lower
        n = self.arrayvar.neighbor(self.p1, 10)
        self.assertEqual(len(n), 10)
        for array in n:
            self.assertTrue(array[0] >= -5 and array[0] <= -4)
            self.assertTrue(array[1] >= -5 and array[1] <= -4)
            self.assertTrue(array[2] in self.catvar.features)
            self.assertEqual(array[3], 5)

        # Middle
        n = self.arrayvar.neighbor(self.p2, 10)
        self.assertEqual(len(n), 10)
        for array in n:
            self.assertTrue(array[0] >= -1 and array[0] <= 1)
            self.assertTrue(array[1] >= -1 and array[1] <= 1)
            self.assertTrue(array[2] in self.catvar.features)
            self.assertEqual(array[3], 5)

        # Upper
        n = self.arrayvar.neighbor(self.p3, 10)
        self.assertEqual(len(n), 10)
        for array in n:
            self.assertTrue(array[0] >= 4 and array[0] <= 5)
            self.assertTrue(array[1] >= 4 and array[1] <= 5)
            self.assertTrue(array[2] in self.catvar.features)
            self.assertEqual(array[3], 5)

    def test_sp(self):

        # Lower
        n = self.sp.neighbor(self.p1, 10)
        self.assertEqual(len(n), 10)
        for array in n:
            self.assertTrue(array[0] >= -5 and array[0] <= -4)
            self.assertTrue(array[1] >= -5 and array[1] <= -4)
            self.assertTrue(array[2] in self.catvar.features)
            self.assertEqual(array[3], 5)

        # Middle
        n = self.sp.neighbor(self.p2, 10)
        self.assertEqual(len(n), 10)
        for array in n:
            self.assertTrue(array[0] >= -1 and array[0] <= 1)
            self.assertTrue(array[1] >= -1 and array[1] <= 1)
            self.assertTrue(array[2] in self.catvar.features)
            self.assertEqual(array[3], 5)

        # Upper
        n = self.sp.neighbor(self.p3, 10)
        self.assertEqual(len(n), 10)
        for array in n:
            self.assertTrue(array[0] >= 4 and array[0] <= 5)
            self.assertTrue(array[1] >= 4 and array[1] <= 5)
            self.assertTrue(array[2] in self.catvar.features)
            self.assertEqual(array[3], 5)


if __name__ == "__main__":
    unittest.main()
