# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from zellij.core import IntVar, FloatVar, CatVar, ArrayVar, Constant
import numpy as np


class TestInt(unittest.TestCase):
    def setUp(self):
        self.var = IntVar("test", -5, 5)

    def test_creation(self):

        self.assertEqual(self.var.label, "test")
        self.assertEqual(self.var.low_bound, -5)
        self.assertEqual(self.var.up_bound, 6)

    def test_random(self):

        rdm = self.var.random(50)

        self.assertTrue(len(rdm), 50)
        for e in rdm:
            self.assertTrue(e <= 5 and e >= -5)

        rdm = self.var.random()
        self.assertIsInstance(rdm, int)

    def test_subset(self):
        new1 = self.var.subset(-5, 0)
        new2 = self.var.subset(0, 5)
        new3 = self.var.subset(-4, 4)
        new4 = self.var.subset(1, 1)

        self.assertIsInstance(new1, IntVar)
        self.assertIsInstance(new2, IntVar)
        self.assertIsInstance(new3, IntVar)
        self.assertIsInstance(new4, Constant)

        self.assertEqual(new1.low_bound, -5)
        self.assertEqual(new2.low_bound, 0)
        self.assertEqual(new3.low_bound, -4)

        self.assertEqual(new1.up_bound, 1)
        self.assertEqual(new2.up_bound, 6)
        self.assertEqual(new3.up_bound, 5)

        self.assertEqual(new4.value, 1)


class TestFloat(unittest.TestCase):
    def setUp(self):
        self.var = FloatVar("test", -5, 5)

    def test_creation(self):

        self.assertEqual(self.var.label, "test")
        self.assertEqual(self.var.low_bound, -5)
        self.assertEqual(self.var.up_bound, 5)

    def test_random(self):

        rdm = self.var.random(50)

        self.assertTrue(len(rdm), 50)
        for e in rdm:
            self.assertTrue(e <= 5 and e >= -5)

        rdm = self.var.random()
        self.assertIsInstance(rdm, float)

    def test_subset(self):
        new1 = self.var.subset(-5, 0)
        new2 = self.var.subset(0, 5)
        new3 = self.var.subset(-4, 4)
        new4 = self.var.subset(1, 1)

        self.assertIsInstance(new1, FloatVar)
        self.assertIsInstance(new2, FloatVar)
        self.assertIsInstance(new3, FloatVar)
        self.assertIsInstance(new4, Constant)

        self.assertEqual(new1.low_bound, -5)
        self.assertEqual(new2.low_bound, 0)
        self.assertEqual(new3.low_bound, -4)

        self.assertEqual(new1.up_bound, 0)
        self.assertEqual(new2.up_bound, 5)
        self.assertEqual(new3.up_bound, 4)

        self.assertEqual(new4.value, 1)


class TestCat(unittest.TestCase):
    def setUp(self):
        self.features = ["a", 1, 1.5, "b"]
        self.var = CatVar("test", self.features)

    def test_creation(self):

        self.assertEqual(self.var.label, "test")
        self.assertEqual(self.var.features, self.features)
        self.assertTrue(
            all([w == 1 / len(self.features) for w in self.var.weights]),
        )

    def test_random(self):

        rdm = self.var.random(50)

        self.assertTrue(len(rdm), 50)
        for e in rdm:
            self.assertTrue(e in self.features)

        rdm = self.var.random()
        self.assertTrue(rdm in self.features)

    def test_subset(self):
        new1 = self.var.subset("a", 1.5)
        new2 = self.var.subset(1, "b")
        new3 = self.var.subset("b", "a")
        new4 = self.var.subset(1, 1)

        self.assertIsInstance(new1, CatVar)
        self.assertIsInstance(new2, CatVar)
        self.assertIsInstance(new3, CatVar)
        self.assertIsInstance(new4, Constant)

        self.assertTrue(all([e != "b" for e in new1.features]))
        self.assertTrue(all([e != "a" for e in new2.features]))
        self.assertTrue(all([(e != 1 and e != 1.5) for e in new3.features]))

        self.assertEqual(new4.value, 1)


class TestArray(unittest.TestCase):
    def setUp(self):
        self.values = [
            IntVar("int_1", 0, 8),
            IntVar("int_2", 4, 45),
            FloatVar("float_1", 2, 12),
            CatVar("cat_1", ["Hello", 87, 2.56]),
        ]

        self.var = ArrayVar(label="test", *self.values)

    def test_creation(self):

        self.assertEqual(self.var.label, "test")
        self.assertEqual(self.var.values, self.values)

        self.assertRaises(AssertionError, ArrayVar, *self.values, 1, 2, 3)

    def test_random(self):

        rdm = self.var.random(50)

        self.assertTrue(len(rdm), 50)
        for lst in rdm:
            self.assertTrue(lst[0] <= 8 and lst[0] >= 0)
            self.assertTrue(lst[1] <= 45 and lst[1] >= 4)
            self.assertTrue(lst[2] <= 12 and lst[2] >= 2)
            self.assertTrue(lst[3] in ["Hello", 87, 2.56])

        rdm = self.var.random()
        self.assertTrue(rdm[0] <= 8 and rdm[0] >= 0)
        self.assertTrue(rdm[1] <= 45 and rdm[1] >= 4)
        self.assertTrue(rdm[2] <= 12 and rdm[2] >= 2)
        self.assertTrue(rdm[3] in ["Hello", 87, 2.56])

    def test_subset(self):
        new1 = self.var.subset([4, 20, 10, 87], [5, 20, 11, 2.56])

        self.assertIsInstance(new1, ArrayVar)
        self.assertIsInstance(new1.values[0], IntVar)
        self.assertIsInstance(new1.values[1], Constant)
        self.assertIsInstance(new1.values[2], FloatVar)
        self.assertIsInstance(new1.values[3], CatVar)

        self.assertEqual(new1.values[0].low_bound, 4)
        self.assertEqual(new1.values[0].up_bound, 6)

        self.assertEqual(new1.values[1].value, 20)

        self.assertEqual(new1.values[2].low_bound, 10)
        self.assertEqual(new1.values[2].up_bound, 11)

        self.assertTrue(all([e != "Hello"] for e in new1.values[3].features))

        self.assertRaises(
            AssertionError, self.var.subset, [4, 20, 10], [5, 20, 11, 2.56]
        )
        self.assertRaises(
            AssertionError, self.var.subset, [4, 20, 10, 87], [5, 20, 11]
        )

    def test_index(self):

        for i, e in enumerate(self.values):
            self.assertEqual(self.var.index(e), i)

    def test_append(self):
        new = CatVar("new", [1, 2, 3])
        self.var.append(new)
        self.assertTrue(self.var.values[-1], new)
        self.assertRaises(ValueError, self.var.append, 1)

    def test_iteration(self):
        for e1, e2 in zip(self.var, self.values):
            self.assertEqual(e1, e2)

    def test_len(self):
        self.assertEqual(len(self.var), 4)


class TestConstant(unittest.TestCase):
    def setUp(self):
        self.var = Constant("test", 5)

    def test_creation(self):

        self.assertEqual(self.var.label, "test")
        self.assertEqual(self.var.value, 5)

        self.assertRaises(AssertionError, Constant, "test", IntVar("int", 1, 2))

    def test_random(self):

        rdm = self.var.random(50)

        self.assertTrue(len(rdm), 50)
        for e in rdm:
            self.assertEqual(e, 5)

        rdm = self.var.random()
        self.assertEqual(rdm, 5)

    def test_subset(self):
        new1 = self.var.subset(-5, 0)
        self.assertEqual(new1, self.var)


if __name__ == "__main__":
    unittest.main()
