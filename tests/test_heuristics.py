# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:18+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from unittest.mock import Mock
from zellij.utils.heuristics import *


class TestHeuristics(unittest.TestCase):
    def setUp(self):

        self.element = Mock()
        self.element.score = None
        self.element.father = "GOD"
        self.element.min_score = float("inf")
        self.element.level = 0

        self.child1 = Mock()
        self.child1.score = 5
        self.child1.father = self.element
        self.child1.min_score = 5
        self.child1.level = 1

        self.child1.all_scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.child1.best_sol_c = [0.5, 0.5, 0.5, 0.5]

        self.best_ind = [0, 0, 0, 0]
        self.best_sc = -10

    def test_heuristics(self):
        for h, v in zip(
            heuristic_list.values(),
            [5, 4.5, 4.5, 2.8722813232690143, 5.0, 1.179437955378122],
        ):

            self.assertEqual(
                h(self.child1, self.best_ind, self.best_sc),
                v,
                f"Error on the value returnes by {h.__name__}",
            )
