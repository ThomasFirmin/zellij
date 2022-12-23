# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:17+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from unittest.mock import Mock
from zellij.utils.fractal import *


class TestFractals(unittest.TestCase):
    def setUp(self):

        self.father = "GOD"
        self.lo = [-5 for i in range(3)]
        self.up = [5 for i in range(3)]
        self.level = 0
        self.id = 0

    def test_Hypercube(self):

        lo = [
            [-5, -5, -5],
            [-5, -5, 0],
            [-5, 0, -5],
            [-5, 0, 0],
            [0, -5, -5],
            [0, -5, 0],
            [0, 0, -5],
            [0, 0, 0],
        ]
        up = [
            [0, 0, 0],
            [0, 0, 5],
            [0, 5, 0],
            [0, 5, 5],
            [5, 0, 0],
            [5, 0, 5],
            [5, 5, 0],
            [5, 5, 5],
        ]

        H = Hypercube(self.father, self.lo, self.up, self.level, self.id)

        H.create_children()
        self.assertEqual(len(H.children), 8, "Wrong number of children")
        for c, l, u in zip(H.children, lo, up):
            self.assertTrue(
                (c.lo_bounds == np.array(l)).all(), "Wrong lower bounds"
            )
            self.assertTrue(
                (c.up_bounds == np.array(u)).all(), "Wrong upper bounds"
            )

        H.add_point([0], [[0, 0, 0]])

        self.assertEqual(H.min_score, 0, "Wrong min score")
        self.assertEqual(H.best_sol, [0, 0, 0], "Wrong best solution")
        self.assertEqual(H.all_scores[0], 0, "Wrong appending scores")
        self.assertEqual(H.solutions[0], [0, 0, 0], "Wrong appending solutions")

    def test_Hypersphere(self):

        lo = [
            [1.5012626584708362, -3.6243686707645817, -3.6243686707645817],
            [-5.0, -3.6243686707645817, -3.6243686707645817],
            [-3.6243686707645817, -5.0, -3.6243686707645817],
            [-3.6243686707645817, 1.5012626584708362, -3.6243686707645817],
            [-3.6243686707645817, -3.6243686707645817, 1.5012626584708362],
            [-3.6243686707645817, -3.6243686707645817, -5.0],
        ]
        up = [
            [5.0, 3.6243686707645817, 3.6243686707645817],
            [-1.5012626584708362, 3.6243686707645817, 3.6243686707645817],
            [3.6243686707645817, -1.5012626584708362, 3.6243686707645817],
            [3.6243686707645817, 5.0, 3.6243686707645817],
            [3.6243686707645817, 3.6243686707645817, 5.0],
            [3.6243686707645817, 3.6243686707645817, -1.5012626584708362],
        ]

        H = Hypersphere(self.father, self.lo, self.up, self.level, self.id)

        self.assertEqual(H.center.tolist(), [0.0, 0.0, 0.0], "Wrong center")
        self.assertEqual(H.radius.tolist(), [8.75, 8.75, 8.75], "Wrong radius")

        H.create_children()
        self.assertEqual(len(H.children), 6, "Wrong number of children")
        for c, l, u in zip(H.children, lo, up):
            self.assertTrue(
                (c.lo_bounds == np.array(l)).all(), "Wrong lower bounds"
            )
            self.assertTrue(
                (c.up_bounds == np.array(u)).all(), "Wrong upper bounds"
            )

        H.add_point([0], [[0, 0, 0]])

        self.assertEqual(H.min_score, 0, "Wrong min score")
        self.assertEqual(H.best_sol, [0, 0, 0], "Wrong best solution")
        self.assertEqual(H.all_scores[0], 0, "Wrong appending scores")
        self.assertEqual(H.solutions[0], [0, 0, 0], "Wrong appending solutions")

    def test_Direct(self):

        lo = [[-5, -5, -5], [0, -5, -5]]
        up = [[0, 5, 5], [5, 5, 5]]

        H = Direct(self.father, self.lo, self.up, self.level, self.id)

        H.create_children()
        self.assertEqual(len(H.children), 2, "Wrong number of children")
        for c, l, u in zip(H.children, lo, up):
            self.assertTrue(
                (c.lo_bounds == np.array(l)).all(), "Wrong lower bounds"
            )
            self.assertTrue(
                (c.up_bounds == np.array(u)).all(), "Wrong upper bounds"
            )

        H.add_point([0], [[0, 0, 0]])

        self.assertEqual(H.min_score, 0, "Wrong min score")
        self.assertEqual(H.best_sol, [0, 0, 0], "Wrong best solution")
        self.assertEqual(H.all_scores[0], 0, "Wrong appending scores")
        self.assertEqual(H.solutions[0], [0, 0, 0], "Wrong appending solutions")
