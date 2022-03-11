import numpy as np
import copy

from zellij.utils.fractal import fractal_list
from zellij.utils.heuristics import heuristic_list
from zellij.utils.tree_search import tree_search_algorithm
# from zellij.transformer_utils.heuristic import heuristic_list
from zellij.utils.loss_func import FDA_loss_func

class FDA:

    def __init__(self,loss_func, search_space, f_calls,exploration,exploitation,fractal="hypersphere",heuristic="best",level=5,tree_search="BS",volume_kwargs={},explor_kwargs={},exploi_kwargs={},ts_kwargs={},verbose=True):

        assert level >= 1, "Fractal level must be >= 1"

        ##############
        # PARAMETERS #
        ##############

        self.loss_func = loss_func
        self.search_space = search_space

        self.heuristic = heuristic_list[heuristic]

        # Exploration and exploitation function
        if type(exploration) != list:
            self.exploration = [exploration]
        else:
            self.exploration = exploration

        self.exploitation = exploitation

        ## KWARGS
        if type(exploration) != list:
            self.explor_kwargs = [explor_kwargs]
        else:
            self.explor_kwargs = explor_kwargs

        self.exploi_kwargs = exploi_kwargs


        self.level = level
        self.max_loss_call = f_calls


        ###################
        # OTHER VARIABLES #
        ###################

        # Working variables
        self.up_bounds = np.array([1.0 for _ in self.search_space.values])
        self.lo_bounds = np.array([0.0 for _ in self.search_space.values])

        self.fractal_name= fractal
        self.fractal = fractal_list[fractal]

        # Initialize first fractal
        self.start_H = self.fractal("God",self.lo_bounds, self.up_bounds,0,0,**volume_kwargs)

        # Initialize tree search
        self.tree_algo = tree_search_algorithm[tree_search]
        self.tree_search = self.tree_algo([self.start_H],self.level,**ts_kwargs)


        # Initialize scoring and criterion variables
        self.best_score = float("inf")
        self.best_ind = None

        # Number of explored hypersphere
        self.n_h = 0
        # Number of loss function call
        self.loss_call = 0


        self.executed = False
        self.total_h = int(((self.search_space.n_variables*2)**(self.level+1)-1)/((self.search_space.n_variables*2)-1))-1

    # Evaluate a list of hypervolumes
    def evaluate(self,hypervolumes):

        # While there are hypervolumes to evaluate do...
        i = 0
        while i < len(hypervolumes) and self.loss_call < self.max_loss_call:

            # Select parent hypervolume
            H = hypervolumes[i]
            j = 0

            # While there are children do...
            while j < len(H.children) and self.loss_call < self.max_loss_call:

                # Select children of parent H
                child = H.children[j]

                j += 1

                # Link the loss function to the actual hypervolume (children)
                modified_loss_func = FDA_loss_func(child,self.loss_func)

                # Count the number of explored hypervolume
                self.n_h += 1

                # Exploitation
                if child.level == self.level:

                    # Select exploration metaheuristic
                    explor_idx = np.min([child.level,len(self.exploration)])-1
                    explor_idx_kwargs = np.min([child.level,len(self.explor_kwargs)])-1

                    # Compute bounds of the children hypervolume
                    lo = self.search_space.convert_to_continuous([child.lo_bounds],True)[0]
                    up = self.search_space.convert_to_continuous([child.up_bounds],True)[0]

                    # Create a search space for the metaheuristic
                    sp = self.search_space.subspace(lo,up)

                    print("  --O  |  Exploitation "+self.fractal_name+" n°",child.id," child of ",child.father.id," at level ",child.level,"\nNumber of explored "+self.fractal_name+": ", self.n_h,"/",self.total_h)

                    # Run exploration, scores and evaluated solutions are saved using FDA_loss_func class
                    exploration = self.exploration[explor_idx](modified_loss_func.evaluate, sp, **self.explor_kwargs[explor_idx_kwargs])
                    exploration.run()

                    # Run exploitation, scores and evaluated solutions are saved using FDA_loss_func class
                    exploitation = self.exploitation(modified_loss_func.evaluate, sp, **self.exploi_kwargs)
                    exploitation.run(child.best_sol,child.min_score)



                    # Save best found solution
                    if child.min_score < self.best_score:

                        print("Best solution found :",child.min_score,"<",self.best_score,"For exploration")
                        self.best_ind = child.best_sol
                        self.best_ind_c = self.search_space.convert_to_continuous([self.best_ind],True)[0]
                        self.best_score = child.min_score

                    child.best_sol_c = self.search_space.convert_to_continuous([child.best_sol],True)[0]
                    child.score = self.heuristic(child,self.best_ind_c,self.best_score)

                    print(f"\t\t=>Score:{child.score}")

                    self.loss_call += modified_loss_func.f_calls

                # Exploration
                else:

                    explor_idx = np.min([child.level,len(self.exploration)])-1
                    explor_idx_kwargs = np.min([child.level,len(self.explor_kwargs)])-1

                    # Compute bounds of the children hypervolume
                    lo = self.search_space.convert_to_continuous([child.lo_bounds],True,True)[0]
                    up = self.search_space.convert_to_continuous([child.up_bounds],True,True)[0]

                    # Create a search space for the metaheuristic
                    sp = self.search_space.subspace(lo,up)

                    print("  O_O  |  Exploration "+self.fractal_name+" n°",child.id," child of ",child.father.id," at level ",child.level,"\nNumber of explored "+self.fractal_name+": ", self.n_h,"/",self.total_h)

                    # Run exploration, scores and evaluated solutions are saved using FDA_loss_func class
                    exploration = self.exploration[explor_idx](modified_loss_func.evaluate, sp, **self.explor_kwargs[explor_idx_kwargs])
                    exploration.run()

                    # Save best found solution
                    if child.min_score < self.best_score:

                        print("Best solution found :",child.min_score,"<",self.best_score,"For exploration")
                        self.best_ind = child.best_sol
                        self.best_ind_c = self.search_space.convert_to_continuous([self.best_ind],False)[0]
                        self.best_score = child.min_score

                    child.best_sol_c = self.search_space.convert_to_continuous([child.best_sol],False)[0]
                    child.score = self.heuristic(child,self.best_ind_c,self.best_score)

                    print(f"\t\t=>Score:{child.score}")

                    self.loss_call += modified_loss_func.f_calls

                # Add children to tree search
                self.tree_search.add(child)

            i += 1

    def run(self, save=False):

        self.n_h = 0

        stop = True

        # Select initial hypervolume (root) from the search tree
        stop, hypervolumes = self.tree_search.get_next()
        print("Starting")

        while stop and self.loss_call < self.max_loss_call:

            for H in hypervolumes:
                if H.level < self.level:
                    H.create_children()

            self.evaluate(hypervolumes)

            stop, hypervolumes = self.tree_search.get_next()

        self.executed = True

        print("_________________________________________________________________")
        print("Loss function calls: ", self.loss_call)
        print("Number of explored "+self.fractal_name+": ", self.n_h,"/",int(((self.search_space.n_variables*2)**(self.level+1)-1)/((self.search_space.n_variables*2)-1))-1)
        print(self.best_score)
        print(self.best_ind)
        print("_________________________________________________________________")

        return self.start_H

    def show(self, function = False, circles = False):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import mpl_toolkits.mplot3d.art3d as art3d
        from matplotlib.lines import Line2D
        import pandas as pd

        H = self.start_H
        H.current_children = -1

        stop = True

        p = []
        s = []

        v = []
        s2 = []
        c = []
        r = []
        while stop:

            p += H.solutions
            s += H.all_scores
            v += [H.lo_bounds.tolist()]
            v += [H.up_bounds.tolist()]
            #v += [H.center.tolist()]
            s2 += [1000/(np.exp(H.level))]*2
            c += [H.level]*2
            #r += [H.radius[0]]

            inter = H.getNextchildren()

            while inter == -1:

                H = H.father

                if type(H) != str:
                    inter = H.getNextchildren()
                else:
                    inter = 2

            if type(inter) == int:
                stop = False
            else:
                H = inter
                H.current_children = -1

        self.search_space.show(pd.DataFrame(p, columns=self.search_space.label),s)
