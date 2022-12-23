# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:18+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from unittest.mock import Mock
from zellij.core import Minimizer, Maximizer, Lambda


class TestMinimizer(unittest.TestCase):
    def setUp(self):

        self.dict_out = {"o1": 1, "o2": 2, "o3": 3}
        self.list_out = [1, 2, 3]
        self.int_out = 1
        self.objective_int = Minimizer(target=2)
        self.objective_str = Minimizer(target="o2")
        self.objective_simple = Minimizer()

    def test_call_dict(self):

        self.assertEqual(self.objective_int(self.dict_out)["objective"], 3)
        self.assertEqual(self.objective_str(self.dict_out)["objective"], 2)
        self.assertEqual(self.objective_simple(self.dict_out)["objective"], 1)

    def test_call_list(self):
        self.assertEqual(self.objective_int(self.list_out)["objective"], 3)
        self.assertEqual(self.objective_simple(self.list_out)["objective"], 1)

    def test_call_int(self):
        self.assertEqual(self.objective_simple(self.int_out)["objective"], 1)


class TestMaximizer(unittest.TestCase):
    def setUp(self):

        self.dict_out = {"o1": 1, "o2": 2, "o3": 3}
        self.list_out = [1, 2, 3]
        self.int_out = 1
        self.objective_int = Maximizer(target=2)
        self.objective_str = Maximizer(target="o2")
        self.objective_simple = Maximizer()

    def test_call_dict(self):

        self.assertEqual(self.objective_int(self.dict_out)["objective"], -3)
        self.assertEqual(self.objective_str(self.dict_out)["objective"], -2)
        self.assertEqual(self.objective_simple(self.dict_out)["objective"], -1)

    def test_call_list(self):
        self.assertEqual(self.objective_int(self.list_out)["objective"], -3)
        self.assertEqual(self.objective_simple(self.list_out)["objective"], -1)

    def test_call_int(self):
        self.assertEqual(self.objective_simple(self.int_out)["objective"], -1)


class TestRatio(unittest.TestCase):
    def setUp(self):

        self.dict_out = {"o1": 1, "o2": 2, "o3": 3}
        self.list_out = [1, 2, 3]
        self.int_out = 1
        self.objective_int_min = Lambda(
            function=lambda x, y: x / y, target=[2, 0]
        )
        self.objective_str_min = Lambda(
            function=lambda x, y: x / y, target=["o2", "o1"]
        )
        self.objective_simple_min = Lambda(function=lambda x: x + 1000)

        self.objective_int_max = Lambda(
            function=lambda x, y: x / y, target=[2, 0], selector="max"
        )
        self.objective_str_max = Lambda(
            function=lambda x, y: x / y, target=["o2", "o1"], selector="max"
        )
        self.objective_simple_max = Lambda(
            function=lambda x: x + 1000, selector="max"
        )

    # MIN
    def test_call_dict_min(self):

        self.assertEqual(self.objective_int_min(self.dict_out)["objective"], 3)
        self.assertEqual(self.objective_str_min(self.dict_out)["objective"], 2)
        self.assertEqual(
            self.objective_simple_min(self.dict_out)["objective"], 1001
        )

    def test_call_list_min(self):

        self.assertEqual(self.objective_int_min(self.list_out)["objective"], 3)
        self.assertEqual(
            self.objective_simple_min(self.list_out)["objective"], 1001
        )

    def test_call_int_min(self):

        self.assertEqual(
            self.objective_simple_min(self.int_out)["objective"], 1001
        )

    # MAX
    def test_call_dict_max(self):

        self.assertEqual(self.objective_int_max(self.dict_out)["objective"], -3)
        self.assertEqual(self.objective_str_max(self.dict_out)["objective"], -2)
        self.assertEqual(
            self.objective_simple_max(self.dict_out)["objective"], -1001
        )

    def test_call_list_max(self):

        self.assertEqual(self.objective_int_max(self.list_out)["objective"], -3)
        self.assertEqual(
            self.objective_simple_max(self.list_out)["objective"], -1001
        )

    def test_call_int_max(self):

        self.assertEqual(
            self.objective_simple_max(self.int_out)["objective"], -1001
        ),


if __name__ == "__main__":
    unittest.main()
