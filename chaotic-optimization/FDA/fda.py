import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.lines import Line2D

import numpy as np
from hypersphere import Hypersphere
from hypercube import Hypercube

from fda_func import LHS,ILS
from tree_search import *


fractal_list = {"hypersphere": Hypersphere,"hypercube": Hypercube}

tree_search_algorithm = {"BFS": Breadth_first_search,"DFS": Depth_first_search, "BS": Beam_search, "BestFS":Best_First_search}

class FDA:

    def __init__(self,lo_bounds,up_bounds,loss_function,f_call=1000,fractal="hypersphere",exploration=LHS,exploitation=ILS,level=5,tree_search="DFS",inflation=1.75,save=False):

        # Initialisation variables
        self.save = save
        self.lo_bounds = lo_bounds
        self.up_bounds = up_bounds
        self.dim = len(up_bounds)
        self.loss_func = f_function
        self.exploration = exploration
        self.exploitation = exploitation

        self.level = level

        up_m_lo = self.up_bounds - self.lo_bounds
        center = self.lo_bounds + (up_m_lo)/2
        radius = up_m_lo/2

        self.fractal = fractal_list[fractal]
        self.start_H = self.fractal("God",self.lo_bounds, self.up_bounds,center,radius,inflation,0,0,save=self.save)

        self.best_score = float("inf")
        self.best_ind = None

        self.n_h = 0

        self.executed = False

        self.max_loss_call = loss_call
        self.loss_call = 0

        self.tree_algo = tree_search_algorithm[tree_search]
        self.tree_search = self.tree_algo(self.start_H,2*self.dim)

    def evaluate(self,hypervolumes):


        for H in hypervolumes:
            for child in H.children:

                self.n_h += 1

                if child.level == self.level:
                    order = 0
                    print("  --O  |  Exploitation hypersphere n°",child.id," child of ",child.father.id," at level ",child.level)
                    self.loss_call += self.exploitation(child,self.loss_func)

                else:
                    order = 1
                    print("  O_O  |  Exploration hypersphere n°",child.id," child of ",child.father.id," at level ",child.level)
                    self.loss_call += self.exploration(child,self.loss_func)

                child.score = child.min_score

                if child.min_score < self.best_score:
                    if order == 0:
                        type="Exploitation"
                    else:
                        type="Exploration"

                    print("Best solution found :",child.min_score,"<",self.best_score,"For ",type)
                    self.best_ind = child.best_sol
                    self.best_score = child.min_score

                self.tree_search.add(child)




    def run(self):

        self.n_h = 0
        self.executed = True

        stop = True

        l = 0

        stop, hypervolumes = self.tree_search.get_next()

        print("Starting")
        while stop and self.loss_call < self.max_loss_call:

            for H in hypervolumes:
                if H.level != self.level:
                    H.create_children()

            self.evaluate(hypervolumes)

            stop, hypervolumes = self.tree_search.get_next()

        return self.start_H



    def show(self, function = False, circles = False):

        if self.executed :

            print("_________________________________________________________________")
            print("Loss function calls: ", self.loss_call)
            print("Number of explored hypersphere: ", self.n_h,"/",int(((self.dim*2)**(self.level+1)-1)/((self.dim*2)-1))-1)
            print(self.best_score)
            print(self.best_ind)
            print("_________________________________________________________________")

            if self.dim == 2:
                nb = 0
                fig, ax = plt.subplots()

                if function:
                    x,y = np.meshgrid(np.linspace(self.lo_bounds[0], self.up_bounds[0], 100),np.linspace(self.lo_bounds[1], self.up_bounds[1], 100))
                    points = np.moveaxis(np.array([x,y]), 0,2).reshape(-1,2)
                    z = np.array(list(map(self.loss_func,points)))
                    ax.contourf(x,y, z.reshape(100,100),100,alpha=0.7)

                ax.set_aspect(1)
                H = self.start_H
                H.current_children = -1

                stop = True

                while stop:

                    p = np.array(H.solutions)
                    nb += len(H.solutions)
                    if p.shape[0] != 0:
                        ax.scatter(p[:,0]+np.random.random()/1000,p[:,1],s=1,c=H.color)

                    if circles:
                        circle = plt.Circle((H.center[0], H.center[1]), H.radius[0],fill=False, alpha=0.2)
                        ax.add_patch(circle)

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

                print(nb)
                ax.set_xlim(self.lo_bounds[0],self.up_bounds[0])
                ax.set_ylim(self.lo_bounds[1],self.up_bounds[1])
                ax.scatter(self.best_ind[0],self.best_ind[1],s=10,c="yellow")


                legend_elements = [Line2D([], [], marker='o', color='blue', label='Exploration',
                          markerfacecolor='blue', markersize=5, linestyle='None'),Line2D([], [], marker='o', color='cyan', label='Exploitation',
                                    markerfacecolor='cyan', markersize=5, linestyle='None'),Line2D([], [], marker='o', color='yellow', label='Best solution',
                                              markerfacecolor='yellow', markersize=10, linestyle='None')]

                plt.suptitle("Exploitation : "+self.exploration.__name__+" |Exploration : "+self.exploitation.__name__+" |Loss_calls : "+str(self.loss_call)+"")
                ax.set_title("Number of explored hypersphere: "+str(self.n_h)+"/"+str(int(((self.dim*2)**(self.level+1)-1)/((self.dim*2)-1))-1))
                ax.legend(handles=legend_elements, loc='upper right')


                plt.show()

            else:
                print("\nPlotting not implemented for dimensions > 2 :(\n")
        else:
            print("\nBefore plotting results run FDA.run()\n")
