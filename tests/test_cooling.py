# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from zellij.strategies.tools import (
    MulExponential,
    MulLogarithmic,
    MulLinear,
    MulQuadratic,
    AddLinear,
    AddQuadratic,
    AddExponential,
    AddTrigonometric,
)
import numpy as np


class TestMulExponential(unittest.TestCase):
    def setUp(self):
        self.cooling = MulExponential(0.85, 1, 0.5, 1)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = MulExponential(0.8, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 1.0)
        self.assertEqual(self.cooling.k, 1)
        self.assertEqual(res, 1.0)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 0.85)
        self.assertEqual(self.cooling.k, 2)
        self.assertEqual(res, 0.85)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 3)
        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 4)
        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 5)
        res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.Tcurrent <= 0.5)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 5)


class TestMulLogarithmic(unittest.TestCase):
    def setUp(self):
        self.cooling = MulLogarithmic(0.85, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = MulLogarithmic(0.8, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 1.0)
        self.assertEqual(self.cooling.k, 1)
        self.assertEqual(res, 1.0)

        res = self.cooling.cool()
        self.assertTrue(
            self.cooling.Tcurrent < 0.63 and self.cooling.Tcurrent > 0.62
        )
        self.assertEqual(self.cooling.k, 2)
        self.assertTrue(res < 0.63 and res > 0.62)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 3)

        res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.Tcurrent <= 0.5)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 3)


class TestMulLinear(unittest.TestCase):
    def setUp(self):
        self.cooling = MulLinear(0.85, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = MulLinear(0.8, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 1.0)
        self.assertEqual(self.cooling.k, 1)
        self.assertEqual(res, 1.0)

        res = self.cooling.cool()
        self.assertTrue(
            self.cooling.Tcurrent < 0.55 and self.cooling.Tcurrent > 0.54
        )
        self.assertEqual(self.cooling.k, 2)
        self.assertTrue(res < 0.55 and res > 0.54)

        res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.Tcurrent <= 0.5)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 2)


class TestMulQuadratic(unittest.TestCase):
    def setUp(self):
        self.cooling = MulQuadratic(0.85, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = MulQuadratic(0.8, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 1.0)
        self.assertEqual(self.cooling.k, 1)
        self.assertEqual(res, 1.0)

        res = self.cooling.cool()
        self.assertTrue(
            self.cooling.Tcurrent < 0.55 and self.cooling.Tcurrent > 0.54
        )
        self.assertEqual(self.cooling.k, 2)
        self.assertTrue(res < 0.55 and res > 0.54)

        res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.Tcurrent <= 0.5)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 2)


class TestAddLinear(unittest.TestCase):
    def setUp(self):
        self.cooling = AddLinear(10, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = AddLinear(10, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 1.0)
        self.assertEqual(self.cooling.k, 1)
        self.assertEqual(res, 1.0)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 0.95)
        self.assertEqual(self.cooling.k, 2)
        self.assertEqual(res, 0.95)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 0.9)
        self.assertEqual(self.cooling.k, 3)
        self.assertEqual(res, 0.9)

        res = True
        while res:
            res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 10)


class TestAddQuadratic(unittest.TestCase):
    def setUp(self):
        self.cooling = AddQuadratic(10, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = AddQuadratic(10, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 1.0)
        self.assertEqual(self.cooling.k, 1)
        self.assertEqual(res, 1.0)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.Tcurrent, 0.905)
        self.assertEqual(self.cooling.k, 2)
        self.assertEqual(res, 0.905)

        res = True
        while res:
            res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 10)


class TestAddExponential(unittest.TestCase):
    def setUp(self):
        self.cooling = AddExponential(10, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = AddExponential(10, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 1)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 2)

        res = True
        while res:
            res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 10)


class TestAddTrigonometric(unittest.TestCase):
    def setUp(self):
        self.cooling = AddTrigonometric(10, 1, 0.5, 1)

    def test_reset(self):
        while self.cooling.cool():
            pass

        self.cooling.reset()
        self.assertEqual(self.cooling.Tcurrent, 1)
        self.assertEqual(self.cooling.k, 0)
        self.assertEqual(self.cooling.cross, 0)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            cooling = AddTrigonometric(10, 4, 5, 1)
            self.fail("Assertion not raised for wrong loss creation")

    def test_cool(self):
        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 1)

        res = self.cooling.cool()
        self.assertEqual(self.cooling.k, 2)

        res = True
        while res:
            res = self.cooling.cool()
        self.assertFalse(res)
        self.assertTrue(self.cooling.Tcurrent <= 0.5)
        self.assertTrue(self.cooling.k == 0)
        self.assertTrue(self.cooling.cross == 1)

    def test_iterations(self):
        self.assertTrue(self.cooling.iterations(), 10)


if __name__ == "__main__":
    unittest.main()
