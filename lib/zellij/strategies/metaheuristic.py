from abc import abstractmethod
import os

class Metaheuristic:

    """Metaheuristic

    Metaheuristic is a core object which define the structure of a metaheuristic in zellij.

    Attributes
    ----------

    loss_func : Loss
        Loss function to optimize. must be of type f(x)=y

    search_space : Searchspace
        Search space object containing bounds of the search space.

    f_calls : int
        Maximum number of loss_func calls

    save : boolean, optional
        if True save results into a file

    verbose : boolean, default=True
        Algorithm verbosity

    Methods
    -------
    create_file(self, *args)
        Create a saving file.


    See Also
    --------
    LossFunc : Parent class for a loss function.
    Searchspace : Define what a search space is in Zellij.
    """

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

        self.file_name = ""

    def create_file(self, *args):

        """create_file(self, *args)s

        Create a saving file.

        Parameters
        ----------
        *args : list[label]
            list of additionnal labels to add before the score/evaluation of a point.

        """

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
