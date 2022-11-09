# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
import os
from zellij.utils.loss_func import *


class TestLoss(unittest.TestCase):
    def setUp(self):
        class dummy:
            def __init__(self):
                pass

            def save(self, filename):
                with open(filename, "w") as f:
                    pass

        @Loss
        def f(x):
            return x[0] + x[1] + int(x[2].encode("utf-8").hex()) + x[3]

        self.f = f

        @Loss(save_model="zellij_test_file")
        def f_save(x):
            return [
                x[0] + x[1] + int(x[2].encode("utf-8").hex()) + x[3],
                2,
                3,
            ], dummy()

        self.f_save = f_save

        self.solution = [[4, 4, "v2", 2], [-5, -5, "v1", 2], [5, 5, "v3", 2]]

    def tearDown(self):
        try:
            os.remove("zellij_test.txt")
            os.remove("zellij_test_file")
        except Exception as e:
            pass

    def test_evaluation(self):

        self.assertEqual(
            self.f(self.solution),
            [7642, 7623, 7645],
            "Wrong results during evaluation of the loss function",
        )
        self.assertEqual(
            self.f.calls, 3, "Wrong counting of calls to the function"
        )

    def test_save_best(self):

        self.f(self.solution)
        self.assertEqual(self.f.best_score, 7623, "Wrong best score")
        self.assertEqual(
            self.f.best_sol, [-5, -5, "v1", 2], "Wrong best solution"
        )
        self.assertEqual(
            self.f.all_scores, [7642, 7623, 7645], "Wrong all scores"
        )
        self.assertEqual(
            self.f.all_solutions,
            [[4, 4, "v2", 2], [-5, -5, "v1", 2], [5, 5, "v3", 2]],
            "Wrong all solutions",
        )
        self.assertTrue(self.f.new_best, "Wrong new best detection")

    def test_save_file(self):

        with open("zellij_test.txt", "w") as f:
            f.write("a,b,c,d,score\n")

        self.f(self.solution, "zellij_test.txt")

        with open("zellij_test.txt", "r") as file:
            i = 0
            lines = [
                "a,b,c,d,score",
                "4,4,v2,2,7642",
                "-5,-5,v1,2,7623",
                "5,5,v3,2,7645",
            ]
            while line := file.readline().rstrip():
                self.assertEqual(line, lines[i], "Wrong file writing")

                i += 1

    def test_save_file_with_args_and_model_save(self):

        with open("zellij_test.txt", "w") as f:
            f.write("a,b,c,d,score\n")

        self.f_save(self.solution, "zellij_test.txt")

        with open("zellij_test.txt", "r") as file:
            i = 0
            lines = [
                "a,b,c,d,score",
                "4,4,v2,2,7642,2,3",
                "-5,-5,v1,2,7623,2,3",
                "5,5,v3,2,7645,2,3",
            ]
            while line := file.readline().rstrip():
                self.assertEqual(line, lines[i], "Wrong file writing")

                i += 1

        self.assertTrue(
            os.path.isfile("zellij_test_file"), "Error when saving model"
        )
