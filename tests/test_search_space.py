# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:19+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from zellij.core import (
    MixedSearchspace,
    ContinuousSearchspace,
    DiscreteSearchspace,
    ArrayVar,
    FloatVar,
    IntVar,
    CatVar,
    Loss,
    MockModel,
    Variable,
)


class TestMixedSearchspace(unittest.TestCase):
    def setUp(self):

        self.values = ArrayVar(
            IntVar("int_1", 0, 8),
            IntVar("int_2", 4, 45),
            FloatVar("float_1", 2, 12),
            CatVar("cat_1", ["Hello", 87, 2.56]),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = MixedSearchspace(self.values, self.loss)

    def test_creation(self):
        """test_creation

        Test create of a MixedSearchspace

        """

        with self.assertRaises(AssertionError):
            sp = MixedSearchspace(self.values, self.loss)
            self.fail("Assertion not raised for wrong loss creation")

        with self.assertRaises(AssertionError):
            sp = MixedSearchspace(IntVar("int_1", 0, 8), self.loss)
            self.fail("Assertion not raised for wrong values creation")

        sp = MixedSearchspace(self.values, self.loss)

        self.assertIsInstance(sp, MixedSearchspace)
        self.assertTrue(len(self.sp) == 4)

    def test_random_attribute(self):
        """Test random_attribute method"""

        # One
        self.assertIsInstance(self.sp.random_attribute()[0], Variable)

        # Multiple
        r_a = self.sp.random_attribute(size=20)
        for elem in r_a:
            self.assertIsInstance(elem, Variable)

        # Excluding type
        r_a = self.sp.random_attribute(size=20, exclude=IntVar)
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, IntVar),
                "Error in excluding a type in random_attribute",
            )

        r_a = self.sp.random_attribute(size=20, exclude=[IntVar, CatVar])
        for elem in r_a:
            self.assertIsInstance(
                elem,
                FloatVar,
                "Error in excluding list of types in random_attribute",
            )

        # Excluding Variable
        r_a = self.sp.random_attribute(size=20, exclude=self.values[2])
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, FloatVar),
                "Error in excluding a Variable in random_attribute",
            )

        r_a = self.sp.random_attribute(size=20, exclude=self.values[0:2])
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, IntVar),
                "Error in excluding a list of Variable in random_attribute",
            )

        # Excluding index
        r_a = self.sp.random_attribute(size=20, exclude=2)
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, FloatVar),
                "Error in excluding an index in random_attribute",
            )

        r_a = self.sp.random_attribute(size=20, exclude=[0, 1])
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, IntVar),
                "Error in excluding a list of indexes in random_attribute",
            )

    def test_random_point(self):
        self.assertTrue(
            len(self.sp.random_point(10)) == 10, "Wrong output size"
        )

    def test_subspace(self):
        lo = [0, 30, 10, "Hello"]
        up = [5, 40, 12, 87]
        new = self.sp.subspace(lo, up)
        self.assertIsInstance(new.values, ArrayVar)
        self.assertTrue(len(new) == 4)

        self.assertIsInstance(new.values[0], IntVar)
        self.assertTrue(new.values[0].low_bound == 0)
        self.assertTrue(new.values[0].up_bound == 6)
        self.assertIsInstance(new.values[1], IntVar)
        self.assertTrue(new.values[1].low_bound == 30)
        self.assertTrue(new.values[1].up_bound == 41)
        self.assertIsInstance(new.values[2], FloatVar)
        self.assertTrue(new.values[2].low_bound == 10)
        self.assertTrue(new.values[2].up_bound == 12)
        self.assertIsInstance(new.values[3], CatVar)
        self.assertTrue(new.values[3].features[0] == "Hello")
        self.assertTrue(new.values[3].features[1] == 87)
        self.assertTrue(len(new.values[3].features) == 2)


class TestContinuousSearchspace(unittest.TestCase):
    def setUp(self):

        self.values = ArrayVar(
            FloatVar("float_1", 0, 5),
            FloatVar("float_2", 10, 15),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = ContinuousSearchspace(self.values, self.loss)

    def test_creation(self):
        """Test creation of a ContinuousSearchspace"""

        with self.assertRaises(AssertionError):
            sp = ContinuousSearchspace(self.values, self.loss)
            self.fail("Assertion not raised for wrong loss creation")

        with self.assertRaises(AssertionError):
            sp = ContinuousSearchspace(FloatVar("float_1", 0, 8), self.loss)
            self.fail("Assertion not raised for wrong values creation")

        with self.assertRaises(AssertionError):
            sp = ContinuousSearchspace(
                ArrayVar(FloatVar("float_1", 0, 8), IntVar("int_1", 0, 8)),
                self.loss,
            )
            self.fail("Assertion not raised for not FloatVar creation")

        sp = ContinuousSearchspace(self.values, self.loss)

        self.assertIsInstance(sp, ContinuousSearchspace)
        self.assertTrue(len(self.sp) == 2)

    def test_random_attribute(self):
        """Test random_attribute method"""

        # One
        self.assertIsInstance(self.sp.random_attribute()[0], FloatVar)

        # Multiple
        r_a = self.sp.random_attribute(size=20)
        for elem in r_a:
            self.assertIsInstance(elem, FloatVar)

    def test_random_point(self):
        self.assertTrue(
            len(self.sp.random_point(10)) == 10, "Wrong output size"
        )

    def test_subspace(self):
        lo = [3, 12]
        up = [4, 13]
        new = self.sp.subspace(lo, up)
        self.assertIsInstance(new.values, ArrayVar)
        self.assertTrue(len(new) == 2)

        self.assertIsInstance(new.values[0], FloatVar)
        self.assertTrue(new.values[0].low_bound == 3)
        self.assertTrue(new.values[0].up_bound == 4)
        self.assertIsInstance(new.values[1], FloatVar)
        self.assertTrue(new.values[1].low_bound == 12)
        self.assertTrue(new.values[1].up_bound == 13)


class TestDiscreteSearchspace(unittest.TestCase):
    def setUp(self):

        self.values = ArrayVar(
            IntVar("int_1", 0, 5),
            IntVar("int2", 10, 15),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = DiscreteSearchspace(self.values, self.loss)

    def test_creation(self):
        """Test creation of a DiscreteSearchspace"""

        with self.assertRaises(AssertionError):
            sp = DiscreteSearchspace(self.values, self.loss)
            self.fail("Assertion not raised for wrong loss creation")

        with self.assertRaises(AssertionError):
            sp = DiscreteSearchspace(FloatVar("float_1", 0, 8), self.loss)
            self.fail("Assertion not raised for wrong values creation")

        with self.assertRaises(AssertionError):
            sp = DiscreteSearchspace(
                ArrayVar(FloatVar("float_1", 0, 8), IntVar("int_1", 0, 8)),
                self.loss,
            )
            self.fail("Assertion not raised for not IntVar creation")

        sp = DiscreteSearchspace(self.values, self.loss)

        self.assertIsInstance(sp, DiscreteSearchspace)
        self.assertTrue(len(self.sp) == 2)

    def test_random_attribute(self):
        """Test random_attribute method"""

        # One
        self.assertIsInstance(self.sp.random_attribute()[0], IntVar)

        # Multiple
        r_a = self.sp.random_attribute(size=20)
        for elem in r_a:
            self.assertIsInstance(elem, IntVar)

    def test_random_point(self):
        self.assertTrue(
            len(self.sp.random_point(10)) == 10, "Wrong output size"
        )

    def test_subspace(self):
        lo = [3, 12]
        up = [4, 13]
        new = self.sp.subspace(lo, up)
        self.assertIsInstance(new.values, ArrayVar)
        self.assertTrue(len(new) == 2)

        self.assertIsInstance(new.values[0], IntVar)
        self.assertTrue(new.values[0].low_bound == 3)
        self.assertTrue(new.values[0].up_bound == 5)
        self.assertIsInstance(new.values[1], IntVar)
        self.assertTrue(new.values[1].low_bound == 12)
        self.assertTrue(new.values[1].up_bound == 14)


if __name__ == "__main__":
    unittest.main()
