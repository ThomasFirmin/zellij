from abc import abstractmethod
import os

class Metaheuristic:

    def __init__(self,loss_func, search_space, f_calls,save = False,verbose=False):

        self.loss_func = loss_func
        self.search_space = search_space
        self.f_calls = f_calls

        self.save = save
        self.verbose=verbose

        self.best_found_score = float("inf")
        self.best_found_sol = None

        self.all_solutions = []
        self.all_scores = []

        if self.save:
            if os.path.isfile(self.__class__.__name__+"_save"+".txt"):
                suffix = 1
            while os.path.isfile(self.__class__.__name__+"_save"+f"_{suffix}"+".txt"):
                suffix += 1

            self.file_name = self.__class__.__name__+"_save"+f"_{suffix}"+".txt"

            file = open(self.file_name,"w")
            file.write(str(self.search_space.label)[1:-1].replace(" ","").replace("'","")+",loss_value\n")
            file.close()

            print("Results will be saved at"+os.path.abspath(self.file_name))


    def save_best(self,solution,score):

        self.all_solutions.append(solution)
        self.all_scores.append(scores)
        
        if score < self.best_found_score:
            self.best_found_score = score
            self.best_found_sol = solution

    def save_file(self,solution,score):
        file = open(self.file_name,"a")
        for solution,score in zip(pop,fits):
            file.write(str(solution)[2:-2].replace(" ","").replace("'","")+","+str(score).replace(" ","")+"\n")
        file.close()

    @abstractmethod
    def run(self):
        return

    @abstractmethod
    def show(self):
