# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
import shutil
from zellij.core import Loss, MockModel


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.f_list = Loss(save=False, verbose=False)(
            MockModel(
                outputs={
                    "o1": lambda *args, **kwargs: args[0][0]
                    + args[0][1]
                    + int(args[0][2].encode("utf-8").hex())
                    + args[0][3]
                },
                return_model=False,
                return_format="list",
                verbose=False,
            )
        )

        self.f_dict = Loss(save=False, verbose=False)(
            MockModel(
                outputs={
                    "o1": lambda *args, **kwargs: args[0][0]
                    + args[0][1]
                    + int(args[0][2].encode("utf-8").hex())
                    + args[0][3]
                },
                return_model=False,
                return_format="dict",
                verbose=False,
            )
        )

        self.f_list_save = Loss(save="zellij_test_list", verbose=False)(
            MockModel(
                outputs={
                    "o1": lambda *args, **kwargs: args[0][0]
                    + args[0][1]
                    + int(args[0][2].encode("utf-8").hex())
                    + args[0][3]
                },
                return_model=False,
                return_format="list",
                verbose=False,
            )
        )

        self.f_dict_save = Loss(save="zellij_test_dict", verbose=False)(
            MockModel(
                outputs={
                    "o1": lambda *args, **kwargs: args[0][0]
                    + args[0][1]
                    + int(args[0][2].encode("utf-8").hex())
                    + args[0][3]
                },
                return_model=False,
                return_format="dict",
                verbose=False,
            )
        )

        self.f_kwargs_mode = Loss(save=False, verbose=False, kwargs_mode=True)(
            MockModel(
                outputs={
                    "o1": lambda a, b, c, d: a
                    + b
                    + int(c.encode("utf-8").hex())
                    + d
                },
                return_model=False,
                return_format="dict",
                verbose=False,
            )
        )
        self.f_kwargs_mode.labels = ["a", "b", "c", "d"]

        self.solution = [[4, 4, "v2", 2], [-5, -5, "v1", 2], [5, 5, "v3", 2]]

    def tearDown(self):
        try:
            shutil.rmtree("zellij_test_list")
            shutil.rmtree("zellij_test_dict")
        except Exception as e:
            pass

    def test_evaluation(self):

        self.assertEqual(
            self.f_list(self.solution),
            [7642, 7623, 7645],
            "Wrong results during evaluation of the loss function",
        )
        self.assertEqual(
            self.f_list.calls, 3, "Wrong counting of calls to the function"
        )

        self.assertEqual(
            self.f_dict(self.solution),
            [7642, 7623, 7645],
            "Wrong results during evaluation of the loss function",
        )
        self.assertEqual(
            self.f_dict.calls, 3, "Wrong counting of calls to the function"
        )

        self.assertEqual(
            self.f_kwargs_mode(self.solution),
            [7642, 7623, 7645],
            "Wrong results during evaluation of the loss function",
        )
        self.assertEqual(
            self.f_kwargs_mode.calls,
            3,
            "Wrong counting of calls to the function",
        )

    def test_save_best(self):

        # List
        self.f_list(self.solution)
        self.assertEqual(self.f_list.best_score, 7623, "Wrong best score")
        self.assertEqual(
            self.f_list.best_point, [-5, -5, "v1", 2], "Wrong best solution"
        )
        self.assertEqual(
            self.f_list.all_scores, [7642, 7623, 7645], "Wrong all scores"
        )
        self.assertEqual(
            self.f_list.all_solutions,
            [[4, 4, "v2", 2], [-5, -5, "v1", 2], [5, 5, "v3", 2]],
            "Wrong all solutions",
        )
        self.assertTrue(self.f_list.new_best, "Wrong new best detection")

        # DICT
        self.f_dict(self.solution)
        self.assertEqual(self.f_dict.best_score, 7623, "Wrong best score")
        self.assertEqual(
            self.f_dict.best_point, [-5, -5, "v1", 2], "Wrong best solution"
        )
        self.assertEqual(
            self.f_dict.all_scores, [7642, 7623, 7645], "Wrong all scores"
        )
        self.assertEqual(
            self.f_dict.all_solutions,
            [[4, 4, "v2", 2], [-5, -5, "v1", 2], [5, 5, "v3", 2]],
            "Wrong all solutions",
        )
        self.assertTrue(self.f_list.new_best, "Wrong new best detection")

    def test_save_file(self):

        self.f_list_save(self.solution)

        with open("zellij_test_list/outputs/all_evaluations.csv", "r") as file:
            i = 0
            lines = [
                "attribute0,attribute1,attribute2,attribute3,r0,objective",
                "4,4,v2,2,7642,7642",
                "-5,-5,v1,2,7623,7623",
                "5,5,v3,2,7645,7645",
            ]
            while line := file.readline().rstrip():
                self.assertEqual(line, lines[i], "Wrong file writing")

                i += 1

        self.f_dict_save(self.solution)

        with open("zellij_test_dict/outputs/all_evaluations.csv", "r") as file:
            i = 0
            lines = [
                "attribute0,attribute1,attribute2,attribute3,o1,objective",
                "4,4,v2,2,7642,7642",
                "-5,-5,v1,2,7623,7623",
                "5,5,v3,2,7645,7645",
            ]
            while line := file.readline().rstrip():
                self.assertEqual(line, lines[i], "Wrong file writing")

                i += 1


if __name__ == "__main__":
    unittest.main()
