from abc import abstractmethod
import os

class Metaheuristic:

    def __init__(self,loss_func, search_space, f_calls,save = False,verbose=False):

        ##############
        # PARAMETERS #
        ##############

        self.loss_func = loss_func
        self.search_space = search_space
        self.f_calls = f_calls

        self.save = save
        self.verbose=verbose

        #############
        # VARIABLES #
        #############

        self.filename = ""

    def create_file(self, *args):

        if self.save:

            if os.path.isfile(self.__class__.__name__+"_save"+".txt"):
                suffix = 1
            while os.path.isfile(self.__class__.__name__+"_save"+f"_{suffix}"+".txt"):
                suffix += 1

            self.file_name = self.__class__.__name__+"_save"+f"_{suffix}"+".txt"

            if len(args) != 0:
                add = ","+",".join(str(e) for e in args)
            else:
                add = ""

            file = open(self.file_name,"w")
            file.write(",".join(str(e) for e in self.search_space.labels)+add+",loss_value\n")
            file.close()

            print("Results will be saved at"+os.path.abspath(self.file_name))

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def show(self):
        pass
