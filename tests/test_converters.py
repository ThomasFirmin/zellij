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
    ArrayMinmax,
    FloatMinmax,
    IntMinmax,
    CatMinmax,
    ConstantMinmax,
    ArrayBinning,
    FloatBinning,
    IntBinning,
    CatBinning,
    ConstantBinning,
    Continuous,
    Discrete,
    DoNothing,
)
import numpy as np


class TestDoNothing(unittest.TestCase):
    def setUp(self):
        self.intvar_cont = IntVar("int", -5, 5, to_continuous=DoNothing())
        self.floatvar_cont = FloatVar("float", -5, 5, to_continuous=DoNothing())
        self.catvar_cont = CatVar("cat", ["I", 1, 5], to_continuous=DoNothing())
        self.consvar_cont = Constant("constant", 5, to_continuous=DoNothing())
        self.arrayvar_cont = ArrayVar(
            self.intvar_cont,
            self.floatvar_cont,
            self.catvar_cont,
            self.consvar_cont,
            label="array",
            to_continuous=DoNothing(),
        )

        self.intvar_disc = IntVar("int", -5, 5, to_discrete=DoNothing())
        self.floatvar_disc = FloatVar("float", -5, 5, to_discrete=DoNothing())
        self.catvar_disc = CatVar("cat", ["I", 1, 5], to_discrete=DoNothing())
        self.consvar_disc = Constant("constant", 5, to_discrete=DoNothing())
        self.arrayvar_disc = ArrayVar(
            self.intvar_disc,
            self.floatvar_disc,
            self.catvar_disc,
            self.consvar_disc,
            label="array",
            to_discrete=DoNothing(),
        )

        self.p1 = [-5, -5, "I", 5]
        self.p2 = [0, 0, 1, 5]
        self.p3 = [5, 5, 5, 5]

    def test_creation(self):

        self.assertTrue(hasattr(self.intvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.floatvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.catvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.consvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.arrayvar_cont, "to_continuous"))

        self.assertIsInstance(self.intvar_cont.to_continuous, DoNothing)
        self.assertIsInstance(self.floatvar_cont.to_continuous, DoNothing)
        self.assertIsInstance(self.catvar_cont.to_continuous, DoNothing)
        self.assertIsInstance(self.consvar_cont.to_continuous, DoNothing)
        self.assertIsInstance(self.arrayvar_cont.to_continuous, DoNothing)

        self.assertTrue(hasattr(self.intvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.floatvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.catvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.consvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.arrayvar_disc, "to_discrete"))

        self.assertIsInstance(self.intvar_disc.to_discrete, DoNothing)
        self.assertIsInstance(self.floatvar_disc.to_discrete, DoNothing)
        self.assertIsInstance(self.catvar_disc.to_discrete, DoNothing)
        self.assertIsInstance(self.consvar_disc.to_discrete, DoNothing)
        self.assertIsInstance(self.arrayvar_disc.to_discrete, DoNothing)

    def test_convert_reverse(self):
        conv = self.arrayvar_cont.to_continuous.convert(self.p1)
        rev = self.arrayvar_cont.to_continuous.reverse(conv)
        for e1, e2, e3 in zip(self.p1, conv, rev):
            self.assertEqual(e1, e2, e3)

        conv = self.arrayvar_cont.to_continuous.convert(self.p2)
        rev = self.arrayvar_cont.to_continuous.reverse(conv)
        for e1, e2, e3 in zip(self.p2, conv, rev):
            self.assertEqual(e1, e2, e3)

        conv = self.arrayvar_cont.to_continuous.convert(self.p3)
        rev = self.arrayvar_cont.to_continuous.reverse(conv)
        for e1, e2, e3 in zip(self.p3, conv, rev):
            self.assertEqual(e1, e2, e3)

        conv = self.arrayvar_disc.to_discrete.convert(self.p1)
        rev = self.arrayvar_disc.to_discrete.reverse(conv)
        for e1, e2, e3 in zip(self.p1, conv, rev):
            self.assertEqual(e1, e2, e3)

        conv = self.arrayvar_disc.to_discrete.convert(self.p2)
        rev = self.arrayvar_disc.to_discrete.reverse(conv)
        for e1, e2, e3 in zip(self.p2, conv, rev):
            self.assertEqual(e1, e2, e3)

        conv = self.arrayvar_disc.to_discrete.convert(self.p3)
        rev = self.arrayvar_disc.to_discrete.reverse(conv)
        for e1, e2, e3 in zip(self.p3, conv, rev):
            self.assertEqual(e1, e2, e3)


class TestMinmax(unittest.TestCase):
    def setUp(self):
        self.intvar_cont = IntVar("int", -5, 5, to_continuous=IntMinmax())
        self.floatvar_cont = FloatVar(
            "float", -5, 5, to_continuous=FloatMinmax()
        )
        self.catvar_cont = CatVar("cat", ["I", 1, 5], to_continuous=CatMinmax())
        self.consvar_cont = Constant(
            "constant", 5, to_continuous=ConstantMinmax()
        )
        self.arrayvar_cont = ArrayVar(
            self.intvar_cont,
            self.floatvar_cont,
            self.catvar_cont,
            self.consvar_cont,
            label="array",
            to_continuous=ArrayMinmax(),
        )

        self.p1 = [-5, -5, "I", 5]
        self.p2 = [0, 0, 1, 5]
        self.p3 = [5, 5, 5, 5]

    def test_creation(self):

        self.assertTrue(hasattr(self.intvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.floatvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.catvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.consvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.arrayvar_cont, "to_continuous"))

        self.assertIsInstance(self.intvar_cont.to_continuous, IntMinmax)
        self.assertIsInstance(self.floatvar_cont.to_continuous, FloatMinmax)
        self.assertIsInstance(self.catvar_cont.to_continuous, CatMinmax)
        self.assertIsInstance(self.consvar_cont.to_continuous, ConstantMinmax)
        self.assertIsInstance(self.arrayvar_cont.to_continuous, ArrayMinmax)

    def test_convert_reverse(self):
        conv = self.arrayvar_cont.to_continuous.convert(self.p1)
        rev = self.arrayvar_cont.to_continuous.reverse(conv)
        for e1, e2 in zip(self.p1, rev):
            self.assertEqual(e1, e2)
        for e3 in conv:
            self.assertIsInstance(e3, float)
            self.assertTrue(e3 >= 0 and e3 <= 1)

        conv = self.arrayvar_cont.to_continuous.convert(self.p2)
        rev = self.arrayvar_cont.to_continuous.reverse(conv)
        for e1, e2 in zip(self.p2, rev):
            self.assertEqual(e1, e2)
        for e3 in conv:
            self.assertIsInstance(e3, float)
            self.assertTrue(e3 >= 0 and e3 <= 1)

        conv = self.arrayvar_cont.to_continuous.convert(self.p3)
        rev = self.arrayvar_cont.to_continuous.reverse(conv)
        for e1, e2 in zip(self.p3, rev):
            self.assertEqual(e1, e2)
        for e3 in conv:
            self.assertIsInstance(e3, float)
            self.assertTrue(e3 >= 0 and e3 <= 1)


class TestBinning(unittest.TestCase):
    def setUp(self):
        self.intvar_disc = IntVar("int", -5, 5, to_discrete=IntBinning(5))
        self.floatvar_disc = FloatVar(
            "float", -5, 5, to_discrete=FloatBinning(5)
        )
        self.catvar_disc = CatVar("cat", ["I", 1, 5], to_discrete=CatBinning())
        self.consvar_disc = Constant(
            "constant", 5, to_discrete=ConstantBinning()
        )
        self.arrayvar_disc = ArrayVar(
            self.intvar_disc,
            self.floatvar_disc,
            self.catvar_disc,
            self.consvar_disc,
            label="array",
            to_discrete=ArrayBinning(),
        )

        self.p1 = [-5, -5, "I", 5]
        self.p2 = [0, 0, 1, 5]
        self.p3 = [5, 5, 5, 5]

    def test_creation(self):

        self.assertTrue(hasattr(self.intvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.floatvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.catvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.consvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.arrayvar_disc, "to_discrete"))

        self.assertIsInstance(self.intvar_disc.to_discrete, IntBinning)
        self.assertIsInstance(self.floatvar_disc.to_discrete, FloatBinning)
        self.assertIsInstance(self.catvar_disc.to_discrete, CatBinning)
        self.assertIsInstance(self.consvar_disc.to_discrete, ConstantBinning)
        self.assertIsInstance(self.arrayvar_disc.to_discrete, ArrayBinning)

    def test_convert_reverse(self):
        conv = self.arrayvar_disc.to_discrete.convert(self.p1)
        rev = self.arrayvar_disc.to_discrete.reverse(conv)
        true_conv = [0, 0, 0, 1]
        true_rev = [-5.0, -5.0, "I", 5]
        for c1, t1, r1, t2 in zip(conv, true_conv, rev, true_rev):
            self.assertEqual(c1, t1)
            self.assertEqual(r1, t2)

        conv = self.arrayvar_disc.to_discrete.convert(self.p2)
        rev = self.arrayvar_disc.to_discrete.reverse(conv)
        true_conv = [1, 2, 1, 1]
        true_rev = [-2.25, 0.0, 1, 5]
        for c1, t1, r1, t2 in zip(conv, true_conv, rev, true_rev):
            self.assertEqual(c1, t1)
            self.assertEqual(r1, t2)

        conv = self.arrayvar_disc.to_discrete.convert(self.p3)
        rev = self.arrayvar_disc.to_discrete.reverse(conv)
        true_conv = [3, 4, 2, 1]
        true_rev = [3.25, 5.0, 5, 5]
        for c1, t1, r1, t2 in zip(conv, true_conv, rev, true_rev):
            self.assertEqual(c1, t1)
            self.assertEqual(r1, t2)


class TestSPDiscrete(unittest.TestCase):
    def setUp(self):
        self.intvar_disc = IntVar("int", -5, 5, to_discrete=IntBinning(5))
        self.floatvar_disc = FloatVar(
            "float", -5, 5, to_discrete=FloatBinning(5)
        )
        self.catvar_disc = CatVar("cat", ["I", 1, 5], to_discrete=CatBinning())
        self.consvar_disc = Constant(
            "constant", 5, to_discrete=ConstantBinning()
        )
        self.arrayvar_disc = ArrayVar(
            self.intvar_disc,
            self.floatvar_disc,
            self.catvar_disc,
            self.consvar_disc,
            label="array",
            to_discrete=ArrayBinning(),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = MixedSearchspace(
            self.arrayvar_disc, self.loss, to_discrete=Discrete()
        )

        self.p1 = [-5, -5, "I", 5]
        self.p2 = [0, 0, 1, 5]
        self.p3 = [5, 5, 5, 5]

    def test_creation(self):

        self.assertTrue(hasattr(self.intvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.floatvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.catvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.consvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.arrayvar_disc, "to_discrete"))
        self.assertTrue(hasattr(self.sp, "to_discrete"))

        self.assertIsInstance(self.intvar_disc.to_discrete, IntBinning)
        self.assertIsInstance(self.floatvar_disc.to_discrete, FloatBinning)
        self.assertIsInstance(self.catvar_disc.to_discrete, CatBinning)
        self.assertIsInstance(self.consvar_disc.to_discrete, ConstantBinning)
        self.assertIsInstance(self.arrayvar_disc.to_discrete, ArrayBinning)

        self.assertIsInstance(self.sp.to_discrete, Discrete)

    def test_convert_reverse(self):
        conv = self.sp.to_discrete.convert([self.p1])[0]
        rev = self.sp.to_discrete.reverse([conv])[0]
        true_conv = [0, 0, 0, 1]
        true_rev = [-5.0, -5.0, "I", 5]
        for c1, t1, r1, t2 in zip(conv, true_conv, rev, true_rev):
            self.assertEqual(c1, t1)
            self.assertEqual(r1, t2)

        conv = self.sp.to_discrete.convert([self.p2])[0]
        rev = self.sp.to_discrete.reverse([conv])[0]
        true_conv = [1, 2, 1, 1]
        true_rev = [-2.25, 0.0, 1, 5]
        for c1, t1, r1, t2 in zip(conv, true_conv, rev, true_rev):
            self.assertEqual(c1, t1)
            self.assertEqual(r1, t2)

        conv = self.sp.to_discrete.convert([self.p3])[0]
        rev = self.sp.to_discrete.reverse([conv])[0]
        true_conv = [3, 4, 2, 1]
        true_rev = [3.25, 5.0, 5, 5]
        for c1, t1, r1, t2 in zip(conv, true_conv, rev, true_rev):
            self.assertEqual(c1, t1)
            self.assertEqual(r1, t2)


class TestSPContinuous(unittest.TestCase):
    def setUp(self):
        self.intvar_cont = IntVar("int", -5, 5, to_continuous=IntMinmax())
        self.floatvar_cont = FloatVar(
            "float", -5, 5, to_continuous=FloatMinmax()
        )
        self.catvar_cont = CatVar("cat", ["I", 1, 5], to_continuous=CatMinmax())
        self.consvar_cont = Constant(
            "constant", 5, to_continuous=ConstantMinmax()
        )
        self.arrayvar_cont = ArrayVar(
            self.intvar_cont,
            self.floatvar_cont,
            self.catvar_cont,
            self.consvar_cont,
            label="array",
            to_continuous=ArrayMinmax(),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = MixedSearchspace(
            self.arrayvar_cont, self.loss, to_continuous=Continuous()
        )

        self.p1 = [-5, -5, "I", 5]
        self.p2 = [0, 0, 1, 5]
        self.p3 = [5, 5, 5, 5]

    def test_creation(self):

        self.assertTrue(hasattr(self.intvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.floatvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.catvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.consvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.arrayvar_cont, "to_continuous"))
        self.assertTrue(hasattr(self.sp, "to_continuous"))

        self.assertIsInstance(self.intvar_cont.to_continuous, IntMinmax)
        self.assertIsInstance(self.floatvar_cont.to_continuous, FloatMinmax)
        self.assertIsInstance(self.catvar_cont.to_continuous, CatMinmax)
        self.assertIsInstance(self.consvar_cont.to_continuous, ConstantMinmax)
        self.assertIsInstance(self.arrayvar_cont.to_continuous, ArrayMinmax)
        self.assertIsInstance(self.sp.to_continuous, Continuous)

    def test_convert_reverse(self):
        conv = self.sp.to_continuous.convert([self.p1])[0]
        rev = self.sp.to_continuous.reverse([conv])[0]
        for e1, e2 in zip(self.p1, rev):
            self.assertEqual(e1, e2)
        for e3 in conv:
            self.assertIsInstance(e3, float)
            self.assertTrue(e3 >= 0 and e3 <= 1)

        conv = self.sp.to_continuous.convert([self.p2])[0]
        rev = self.sp.to_continuous.reverse([conv])[0]
        for e1, e2 in zip(self.p2, rev):
            self.assertEqual(e1, e2)
        for e3 in conv:
            self.assertIsInstance(e3, float)
            self.assertTrue(e3 >= 0 and e3 <= 1)

        conv = self.sp.to_continuous.convert([self.p3])[0]
        rev = self.sp.to_continuous.reverse([conv])[0]
        for e1, e2 in zip(self.p3, rev):
            self.assertEqual(e1, e2)
        for e3 in conv:
            self.assertIsInstance(e3, float)
            self.assertTrue(e3 >= 0 and e3 <= 1)


if __name__ == "__main__":
    unittest.main()
