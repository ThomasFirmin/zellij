import numpy as np
import abc


class Fractal():

        def __init__(self,lo_bounds,up_bounds,father,level,id,children=[],score=None,save=False):


            self.lo_bounds = lo_bounds
            self.up_bounds = up_bounds

            self.id = id
            self.father = father
            self.children = []
            self.score = score
            self.level = level

            self.min_score = float("inf")
            self.best_sol = None

            self.solutions = []
            self.color = []
            self.all_scores = []

            self.current_children = -1

            self.save = save

        def getNextchildren(self):

            if self.children == 0:
                return 0
            else:
                self.current_children += 1
                if self.current_children == len(self.children):

                    self.current_children = -1
                    return -1
                else:
                    return self.children[self.current_children]

        def write_solution(self,solution,score,new=False):
            self.solutions.append(solution)
            self.all_scores.append(score)

            if self.save:
                if new :
                    mode = "w"
                    self.f = open("results_fda.txt", mode)
                    self.f.close()

                else:
                    mode = "a"
                    self.f = open("results_fda.txt", mode)
                    self.f.write(str(solution.tolist())[1:-1] + "," + str(score) +"\n")
                    self.f.close()

        def add_point(score, solution, color= "black"):

            self.write_solution(solution,score)

            self.color.append(color)

            if score < H.min_score:
                self.min_score = score
                self.best_sol = solution

        @abc.abstractmethod
        def create_children(self):
            pass

        @abc.abstractmethod
        def isin(self,solution):
            pass

        @abc.abstractmethod
        def essential_infos(self):
            pass
