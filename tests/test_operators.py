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
    NeighborMutation,
)
import numpy as np


class TestNeighborMutation(unittest.TestCase):
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
            self.arrayvar, self.loss, mutation=NeighborMutation(1)
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
        self.assertTrue(hasattr(self.sp, "mutation"))

        self.assertIsInstance(self.intvar.neighbor, IntInterval)
        self.assertIsInstance(self.floatvar.neighbor, FloatInterval)
        self.assertIsInstance(self.catvar.neighbor, CatInterval)
        self.assertIsInstance(self.consvar.neighbor, ConstantInterval)
        self.assertIsInstance(self.arrayvar.neighbor, ArrayInterval)
        self.assertIsInstance(self.sp.mutation, NeighborMutation)

    def test_sp(self):

        for _ in range(10):
            # Lower
            n = self.sp.mutation(self.p1[:])[0]
            self.assertEqual(len(n), 4)
            self.assertTrue(n[0] >= -5 and n[0] <= -4)
            self.assertTrue(n[1] >= -5 and n[1] <= -4)
            self.assertTrue(n[2] in self.catvar.features)
            self.assertEqual(n[3], 5)

            # Middle
            n = self.sp.mutation(self.p2[:])[0]
            self.assertEqual(len(n), 4)
            for e in n:
                self.assertTrue(n[0] >= -1 and n[0] <= 1)
                self.assertTrue(n[1] >= -1 and n[1] <= 1)
                self.assertTrue(n[2] in self.catvar.features)
                self.assertEqual(n[3], 5)

            # Upper
            n = self.sp.mutation(self.p3[:])[0]
            self.assertEqual(len(n), 4)
            for e in n:
                self.assertTrue(n[0] >= 4 and n[0] <= 5)
                self.assertTrue(n[1] >= 4 and n[1] <= 5)
                self.assertTrue(n[2] in self.catvar.features)
                self.assertEqual(n[3], 5)


if __name__ == "__main__":
    unittest.main()
