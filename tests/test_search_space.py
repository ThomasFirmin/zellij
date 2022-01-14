import unittest
from zellij.utils.search_space import *


class TestSearchSpace(unittest.TestCase):
    def test_creation(self):

        label = ["a", "b", "c", "d"]
        type = ["R", "D", "C", "K"]
        values = [[-5, 5], [-5, 5], ["v1", "v2", "v3"], 2]
        neighborhood = [0.5, 1, -1, -1]

        with self.assertRaises(AssertionError):
            sp = Searchspace([], type, values, neighborhood)
            self.fail("Assertion not raised for labels size creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, [], values, neighborhood)
            self.fail("Assertion not raised for types size creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, [], neighborhood)
            self.fail("Assertion not raised for values size creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, values, [])
            self.fail("Assertion not raised for neighborhood size creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace([1, 2.0, "a", 2], type, values, neighborhood)
            self.fail("Assertion not raised for wrong labels creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, ["R", "R", "R", "R"], values, neighborhood)
            self.fail("Assertion not raised for wrong types creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, [["l"], [-5, 5], [-5, 5], [-5, -5]], neighborhood)
            self.fail("Assertion not raised for wrong values creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, values, [0.5, 1, 0.5, 5])
            self.fail("Assertion not raised for wrong neighborhood creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, values, 5)
            self.fail("Assertion not raised for wrong neighborhood value creation")

        sp = Searchspace(label, type, values)

        self.assertEqual(sp.neighborhood, [1.0, 1, -1, -1], "Wrong neighborhood creation")

    def setUp(self):
        labels = ["a", "b", "c", "d"]
        types = ["R", "D", "C", "K"]
        values = [[-5, 5], [-5, 5], ["v1", "v2", "v3"], 2]
        neighborhood = [0.5, 1, -1, -1]

        self.sp = Searchspace(labels, types, values, neighborhood)

    def test_random_attribute(self):
        self.assertIn(self.sp.random_attribute(), self.sp.label, "Wrong Random attribute with size=1, replace=True, exclude=None")

        self.assertIn(self.sp.random_attribute(exclude="a"), self.sp.label[1:], "Wrong Random attribute with exclusion")

        with self.assertRaises(ValueError):
            self.sp.random_attribute(size=10, replace=False, exclude="a")

    def test_random_value(self):
        self.assertTrue(
            isinstance(self.sp.random_value("a")[0], float) and (-5 <= self.sp.random_value("a")[0] <= 5), "Wrong Random real value with size=1, replace=True, exclude=None"
        )

        self.assertTrue(
            isinstance(self.sp.random_value("b")[0], int) and (-5 <= self.sp.random_value("b")[0] <= 5), "Wrong Random int value with size=1, replace=True, exclude=None"
        )

        self.assertIn(self.sp.random_value("c")[0], self.sp.values[2], "Wrong Random categorical value with size=1, replace=True, exclude=None")

        self.assertIn(self.sp.random_value("c", exclude="v1")[0], self.sp.values[2][1:], "Wrong Random attribute with exclusion")

        with self.assertRaises(ValueError):
            self.sp.random_value("c", size=10, replace=False, exclude="v1")

    def test_get_real_neighbor(self):
        neighbor = self.sp._get_real_neighbor(4, 0)
        self.assertTrue(isinstance(neighbor, float) and -5 <= neighbor <= 5 and neighbor != 4, "Wrong real neighbor generation")

    def test_get_discrete_neighbor(self):
        neighbor = self.sp._get_discrete_neighbor(4, 1)
        self.assertTrue(isinstance(neighbor, int) and -5 <= neighbor <= 5 and neighbor != 4, "Wrong discrete neighbor generation")

    def test_get_categorical_neighbor(self):
        neighbor = self.sp._get_categorical_neighbor("v2", 2)
        self.assertTrue(neighbor in ["v1", "v3"], "Wrong categorical neighbor generation")

    def test_get_neighbor(self):

        neighbor = self.sp.get_neighbor([4, 4, "v2", 2])[0]

        self.assertTrue(
            (isinstance(neighbor[0], float) or isinstance(neighbor[0], int))
            and isinstance(neighbor[1], int)
            and neighbor[2] in ["v1", "v2", "v3"]
            and neighbor[3] == 2
            and -5 <= neighbor[0] <= 5
            and -5 <= neighbor[1] <= 5
            and (neighbor[0] != 4 or neighbor[1] != 4 or neighbor[2] != "v2"),
            "Wrong neighbor generation",
        )

        neighbor = self.sp.get_neighbor([-5, -5, "v1", 2])[0]

        self.assertTrue(
            (isinstance(neighbor[0], float) or isinstance(neighbor[0], int))
            and isinstance(neighbor[1], int)
            and neighbor[2] in ["v1", "v2", "v3"]
            and neighbor[3] == 2
            and -5 <= neighbor[0] <= 5
            and -5 <= neighbor[1] <= 5
            and (neighbor[0] != -5 or neighbor[1] != -5 or neighbor[2] != "v1"),
            "Wrong neighbor generation for lower bounds",
        )

        neighbor = self.sp.get_neighbor([5, 5, "v3", 2])[0]
        self.assertTrue(
            (isinstance(neighbor[0], float) or isinstance(neighbor[0], int))
            and isinstance(neighbor[1], int)
            and neighbor[2] in ["v1", "v2", "v3"]
            and neighbor[3] == 2
            and -5 <= neighbor[0] <= 5
            and -5 <= neighbor[1] <= 5
            and (neighbor[0] != 5 or neighbor[1] != 5 or neighbor[2] != "v3"),
            "Wrong neighbor generation for upper bounds",
        )

    def test_random_point(self):
        point = self.sp.random_point()[0]

        self.assertTrue(
            (isinstance(point[0], float) or isinstance(point[0], int))
            and isinstance(point[1], int)
            and point[2] in ["v1", "v2", "v3"]
            and point[3] == 2
            and -5 <= point[0] <= 5
            and -5 <= point[1] <= 5,
            "Wrong random point generation",
        )

    def test_convert(self):

        converted = self.sp.convert_to_continuous([[4, 4, "v2", 2]])[0]
        reconverted = self.sp.convert_to_continuous([converted], reverse=True)[0]

        self.assertEqual(converted, [0.9, 0.9, 0.3333333333333333, 1], "Wrong convertion to continuous")
        self.assertEqual(reconverted, [4, 4, "v2", 2], "Wrong convertion to mixed")

        converted = self.sp.convert_to_continuous([[5, 5, "v3", 2]])[0]
        reconverted = self.sp.convert_to_continuous([converted], reverse=True)[0]

        self.assertEqual(converted, [1, 1, 0.6666666666666666, 1], "Wrong convertion to continuous of the upper bounds")
        self.assertEqual(reconverted, [5, 5, "v3", 2], "Wrong convertion to mixed of the upper bounds")

        converted = self.sp.convert_to_continuous([[-5, -5, "v1", 2]])[0]
        reconverted = self.sp.convert_to_continuous([converted], reverse=True)[0]

        self.assertEqual(converted, [0, 0, 0, 1], "Wrong convertion to continuous of the lower bounds")
        self.assertEqual(reconverted, [-5, -5, "v1", 2], "Wrong convertion to mixed of the lower bounds")

    def test_general_convert(self):
        spc = self.sp.general_convert()

        self.assertEqual(spc.label, self.sp.label, "Wrong general convertion for labels")
        self.assertTrue(all(x == "R" for x in spc.types), "Wrong general convertion for types")
        self.assertTrue(all(x == [0, 1] for x in spc.values), "Wrong general convertion for values")

        self.assertEqual(spc.neighborhood, [0.05, 0.1, 1, 1], "Wrong general convertion of the neighborhood")

    def test_subspace(self):
        lo = [-2, 2, "v2", 1]
        up = [2, 2, "v3", 3]
        spc = self.sp.subspace(lo, up)

        self.assertEqual(spc.values, [[-2.0, 2.0], 2, ["v2", "v3"], 2], "Wrong subspacing for values")
        self.assertEqual(spc.types, ["R", "K", "C", "K"], "Wrong subspacing for types")
        self.assertEqual(spc.neighborhood, [0.2, -1, -1, -1], "Wrong subspacing for neighborhood")

        lo = [-5, -5, "v3", 1]
        up = [-4.99, -4, "v1", 1]
        spc = self.sp.subspace(lo, up)

        self.assertEqual(spc.values, [[-5.0, -4.99], [-5, -4], ["v1", "v2", "v3"], 2], "Wrong subspacing for lower bounds - values")
        self.assertEqual(spc.types, ["R", "D", "C", "K"], "Wrong subspacing for lower bounds - types")
        self.assertEqual(spc.neighborhood, [0.0004999999999999894, 1, -1, -1], "Wrong subspacing for lower bounds - neighborhood")

        lo = [4.99, 4, "v1", 1]
        up = [5, 5, "v1", 1]
        spc = self.sp.subspace(lo, up)

        self.assertEqual(spc.values, [[4.99, 5.0], [4, 5], "v1", 2], "Wrong subspacing for upper bounds")
        self.assertEqual(spc.types, ["R", "D", "K", "K"], "Wrong subspacing for upper bounds - types")
        self.assertEqual(spc.neighborhood, [0.0004999999999999894, 1, -1, -1], "Wrong subspacing for upper bounds - neighborhood")


if __name__ == "__main__":
    unittest.main()
