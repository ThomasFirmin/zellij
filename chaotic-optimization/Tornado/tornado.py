import numpy as np
import random
from typing import List, Callable, Any, Tuple

import chaos_map as cmap


def f_exp(x,lengthscale):
    return np.exp(-np.linalg.norm(x, axis=2)/lengthscale[:,np.newaxis])

def f(xs,centers,lengthscales, coeffs):
    diffs = np.array([xs-i for i in centers])
    exps = f_exp(diffs, lengthscales)
    return np.sum(coeffs[:,np.newaxis]*exps, axis=0)

class Tornado :

    def __init__(self, loss_func, lo_bounds, up_bounds, gds=False, windowed_cgs = 0.05 , return_history=True, chaos_map_func=["henon_map"],f_call=1000, M_global=200, M_local=50, N_level_cgs=50, N_level_cls=5, N_level_cfs=5, N_symetric_p=8, red_rate=0.5, **kwarg):

        """ Documentation Tornado Algorithm:
        \n CGS: Chaotic Global Search, explore widely the search space
        \n CLS: Chaotic Global Search, explore the neighborhood by computing, moving and deacreasing areas arround a solution, can escape from local optima
        \n CFS: Choatic Fine Search, explore the neighborhood of a solution by computing exponential zoom on the area arround the solution, can refine a solution
        \n loss_function: function that take a vector of float in entry and return a loss value (float)
        \n lo_bounds: lower bounds of the search space (int or float)
        \n up_bound: upper bounds of the search space (int or float)
        \n Dimensions of lower and upper bounds must be equal, the length of the vector is equal to the length of a problem's solution
        \n chaos_map_fun: name of the chaos map to use {henon,...} (default: henon)
        \n M_global: number of total iteration (default: 200)
        \n M_local: number of CLS/CFS cycle per iteration (default: 50)
        \n N_level_cgs: number of chaotic level for CGS (default: 50)
        \n N_level_cls: number of chaotic level for CLS (default: 5)
        \n N_level_cfs: number of chaotic level for CFS (default: 5)
        \n N_symetric_p: number of symetric points to compute to form a polygon for symetric optimization (default: 8)"""

        # Initialization

        #self.loss_func_args = kwargs.get('loss_func_kwargs',None)
        #self.loss_func_input_name = kwargs.get('loss_func_input_name',None)
        #self.chaos_map_args = kwargs.get('chaos_map_kwargs',None)

        self.f_call = f_call
        self.loss_func = loss_func
        self.chaos_map_name = chaos_map_func

        self.lo_bounds = np.array(lo_bounds)
        self.up_bounds = np.array(up_bounds)


        self.m_global = M_global
        self.m_local = M_local
        self.n_level_cgs = N_level_cgs
        self.n_level_cls = N_level_cls
        self.n_level_cfs = N_level_cfs
        self.n_symetric_p = N_symetric_p
        self.best_x = None
        self.best_y = float("inf")

        # Working/Recurrent variables

        self.dim = len(self.up_bounds)

        if len(self.chaos_map_name)==1:
            self.map_size = np.max([self.n_level_cgs+self.m_global*self.n_level_cgs, self.m_local+self.m_global*self.n_level_cls,self.m_local+self.m_global*self.n_level_cfs])
            self.chaos_map = cmap.select(self.chaos_map_name[0])
            self.chaos_variables = self.chaos_map(self.map_size,self.dim)
        else:
            self.chaos_variables = np.empty((1,self.dim),dtype=float)
            self.map_size = int(np.ceil(np.max([self.n_level_cgs+self.m_global*self.n_level_cgs, self.m_local+self.m_global*self.n_level_cls,self.m_local+self.m_global*self.n_level_cfs])/len(self.chaos_map_name)))

            for i in self.chaos_map_name:
                self.chaos_map = cmap.select(i)
                self.chaos_variables = np.vstack((self.chaos_variables,self.chaos_map(self.map_size,self.dim)))

            np.random.shuffle(self.chaos_variables)

        self.one_m_chaos_variables = 1-self.chaos_variables

        self.up_plus_lo = np.add(up_bounds,lo_bounds)
        self.up_m_lo = np.subtract(up_bounds,lo_bounds)
        self.center = np.multiply(0.5,self.up_plus_lo)
        self.radius = np.multiply(0.5,self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

        self.red_rate = red_rate
        trigo_val = 2*np.pi/self.n_symetric_p
        self.H = [np.zeros(self.n_symetric_p),np.zeros(self.n_symetric_p)]

        for i in range(1,self.n_symetric_p+1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i-1] = np.cos(trigo_val*i)
            self.H[1][i-1] = np.sin(trigo_val*i)

        self.gds = gds
        self.windowed_cgs = windowed_cgs

        # Replace CFS by GDS
        if self.gds:
            self.p=0
            self.n_level_gds = self.n_level_cfs
            self.n_level_cfs=0

        # If windowed CGS is selected, compute the window decrease factor at each step of CGS
        if self.windowed_cgs > 0:
            self.coef = np.log(self.windowed_cgs/(self.n_level_cgs-1))/np.log(self.red_rate)

        # To save points
        self.return_history,self.save = return_history,return_history
        self.points = np.empty((1,self.dim),dtype=float)
        self.values = np.empty((1,1),dtype=float)
        self.color = np.empty((1,1),dtype=float)
        self.size = np.empty((1,1),dtype=float)
        #Penalty settings
        self.centers = None


    # Gradient for GDS
    def gradient(self,p,solution):

        # Initialize loss func calls
        loss_func = 0

        # Initialize the step: alpha^p
        alpha=1e-6
        eps=1e-320

        # Compute the step according to p, the number of GDS fails
        h=max(alpha**p,eps)

        # Initialize the gradient vector
        g = np.zeros(self.dim)

        # For each dimension of a given solution
        for i in range(self.dim):

            # Initialize both variables for gradient, add step and subtrac step
            xpk = np.copy(solution)
            xmk = np.copy(solution)

            xpk[i] = np.min([solution[i]+h,self.up_bounds[i]])
            xmk[i] = np.max([solution[i]-h,self.lo_bounds[i]])

            # Compute the loss value of each variable
            l1 = self.loss_func(xpk)
            l2 = self.loss_func(xmk)

            # Save the information about computed points
            self.save_info([xpk,xmk],[l1,l2],0.33,25,self.save)

            # add number of loss func calls
            loss_func += 2

            # Compute the gradient value
            inter = xpk[i]-xmk[i]

            # If inter not equal to the infinite, else replace inter by a huge number
            if inter != 0:
                g[i] = (l1-l2)/(inter)
            else:
                g[i] = np.sign(l1-l2)*1e100

        # return gradient and loss_func calls
        return g,loss_func


    # Compute the adjustable step for GDS
    def bkstep(self,p,solution,loss_value,s):

        # Initialize number of loss func calls
        loss_call = 0

        # Initialize step
        r1 = 1e-3
        # Initialize the step decreasing factor
        beta = 0.2

        # Compute the gradient of the current solution
        d,loss_call = self.gradient(p,solution)
        d = -d

        # Compute the norm of the current gradient
        norm =np.linalg.norm(d)

        # Compute the gradient step
        if norm == 0:
            lambda0 = 1
        else:
            lambda0 = 2*r1/(np.dot(d,d.T))

        if np.isnan(lambda0):
            lambda0 = 10e4

        lambda0 = lambda0/(p+1)

        k = 0

        lambdak = lambda0

        xt = solution+lambdak*d
        loss_value = self.loss_func(xt)
        loss_call += 1

        # Save computed solution
        self.save_info([xt],[loss_call],0.33,25,self.save)

        # while computed loss_value is lower than precedent loss value -> decrease the step by beta
        while loss_value > loss_value+r1*lambdak*(np.sum(d.T*d)) and k < 50:

            inter = lambdak
            lamdak = lambda0*beta
            lambda0 = inter
            k += 1

            # Compute the pertubated solution
            xt = solution+lambdak*d
            loss_value = self.loss_func(xt)

            # Save the computed solution
            self.save_info([xt],[loss_call],0.33,25,self.save)

            loss_call += 1

        return lambdak,loss_call

    # Save points for plotting
    def save_info(self,x,y,size,color,save):
        if save:
            self.points = np.vstack((self.points,x))
            self.values = np.append(self.values,y)
            self.color = np.append(self.color,[color]*len(x))
            self.size = np.append(self.size,[size]*len(x))

    def stochastic_round(self, solution, k):

        solution = solution.astype(np.float64)
        r = np.random.uniform(-1,1,len(solution))
        # perturbation on CFS zoom
        z = np.round(solution)+(k%2)*r

        return z

    # Compute a penalty on the loss function
    def penalty(self,x) :
        n = len(self.centers)
        return f(x,self.centers,np.ones(n), np.ones(n)*30)

    # Wrap the loss_func to penalize it
    def loss(self,x):
        res = np.zeros(len(x))

        if self.centers is not None :
            pen = self.penalty(x)
            for i in range(len(x)):
                res[i] = self.loss_func(x[i]) + pen[i]
            return res
        else :
            for i in range(len(x)):
                res[i] = self.loss_func(x[i])
            return res

    # CGS
    def chaotic_global_search(self,k):

        # For each level of chaos
        shift_map = (k-1)*self.n_level_cgs
        points = np.zeros((16*self.n_level_cgs,self.dim))
        n_points = 0

        for l in range(self.n_level_cgs):


            # Select chaotic_variables among the choatic map
            y = self.chaos_variables[l+shift_map]

            # Apply 3 transformations on the selected chaotic variables


            #up_m_lo_mu_y = np.multiply(self.up_m_lo,y)

            if self.windowed_cgs > 0:
                rc= 1-l*self.red_rate**self.coef
                print(rc)
                r_mul_y = np.multiply(rc*self.radius,y)
                #xx = [np.add(rc*self.lo_bounds,rc*self.up_m_lo*y), np.add(self.center,r_mul_y), np.subtract(self.up_bounds,r_mul_y)]
                xx = [self.center+r_mul_y,self.up_bounds-r_mul_y,self.center+rc*self.radius*(2*y-1)]
            else:
                r_mul_y = np.multiply(self.radius,y)
                #xx = [np.add(self.lo_bounds,self.up_m_lo*y), np.add(self.center,r_mul_y), np.subtract(self.up_bounds,r_mul_y)]
                xx = [self.center+r_mul_y,self.up_bounds-r_mul_y,self.center+self.radius*(2*y-1)]

            #xx = [np.add(self.lo_bounds,r_mul_y),np.subtract(self.up_bounds,r_mul_y)]

            # Randomly select a parameter index of a solution
            d=np.random.randint(self.dim)

            # for each transformation of the chaotic variable
            for i,x in enumerate(xx):

                x_ = np.subtract(self.up_plus_lo,x)
                sym = np.matrix([x,x,x_,x_])
                sym[1,d] = x_[d]
                sym[3,d] = x[d]
                points[n_points: n_points+4] = sym
                n_points += 4

        ys = self.loss(points)
        idx = ys.argmin()

        self.save_info(points,ys,0.99,100,self.save)

        best_x, best_y = points[idx], ys[idx]

        return best_x, best_y

    # CLS
    def chaotic_local_search(self,k,nl,solution,loss_value):

        # Initialization
        shift = nl+(k-1)*self.n_level_cls

        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.up_bounds-solution,solution-self.lo_bounds)

        # Local search area radius
        Rl = self.radius*self.red_rate

        center_m_solution = self.center - solution
        points = np.zeros((2*self.n_level_cls*self.n_symetric_p,self.dim))
        n_points = 0

        # for each level of chaos
        for l in range(self.n_level_cls):

            # Decomposition vector
            d = np.random.randint(self.dim)

            # zoom speed
            gamma = 1/(10**(2*self.red_rate*l)*(l+1))

            # for each parameter of a solution determine the improved radius
            xx = np.minimum(gamma*Rl,db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(xx,self.chaos_variables[shift]),np.multiply(xx,self.one_m_chaos_variables[shift])]


            # For both chaotic variable
            for x in xv :
                xi = np.outer(self.H[1],x)
                xi[:,d] = x[d]*self.H[0]
                xt = solution + xi

                points[n_points: n_points+self.n_symetric_p] = xt
                n_points += self.n_symetric_p

        ys = self.loss(points)
        idx = ys.argmin()

        self.save_info(points,ys,0.66,50,self.save)

        if ys[idx] < loss_value :
            return points[idx], ys[idx]
        else :
            return solution, loss_value

    # CFS
    def chaotic_fine_search(self,k,nl,solution,loss_value):

        shift = nl+(k-1)*self.n_level_cfs

        y = self.chaos_variables[shift]*self.chaos_variables[k-1]
        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.up_bounds-solution,solution-self.lo_bounds)

        r_g = np.zeros(self.dim)

        # Local search area radius
        Rl = self.radius*self.red_rate

        xc = solution
        zc = loss_value

        center_m_solution = self.center - solution
        points = np.zeros((2*self.n_level_cfs*self.n_symetric_p,self.dim))
        n_points = 0

        # for each chaotic level
        for l in range(self.n_level_cfs):
            # Decomposition vector
            d = np.random.randint(self.dim)

            # Exponential Zoom factor on the search window
            pc = 10**(l+1)

            # Compute the error/the perturbation applied to the solution
            error_g = np.absolute(solution-(self.stochastic_round(pc*solution,shift)/pc))

            r = np.random.random()

            # for each parameter of a solution determine the improved radius
            r_g = np.minimum((Rl*error_g)/(l**2+1),db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(r_g,y),np.multiply(r_g,y)]

            # For both chaotic variable
            for x in xv :
                xi = np.outer(self.H[1],x)
                xi[:,d] = x[d]*self.H[0]
                xt = solution + xi

                points[n_points: n_points+self.n_symetric_p] = xt
                n_points += self.n_symetric_p

        ys = self.loss(points)
        idx = ys.argmin()

        self.save_info(points,ys,0.33,25,self.save)

        if ys[idx] < loss_value :
            return points[idx], ys[idx]
        else :
            return solution, loss_value

    # GDS, can replace the CFS
    def gradient_descent_search(self,nl,solution,loss_value):

        stop = True
        l = 0

        xc = solution
        zc = loss_value

        # Compute the adaptative step for gradient
        lambda0,loss_func_call = self.bkstep(self.p, solution, loss_value, l+nl)

        while (l < self.n_level_gds) and stop :

            # Compute the gradient of the current solution
            gk,inter = self.gradient(self.p, xc)
            loss_func_call += inter

            # Compute the new solution with the gradient and adaptative step
            xt = xc-lambda0*gk


            if (xt > self.lo_bounds).any() or (xt > self.up_bounds).any():
                stop=False
            else:
                ys = self.loss_func(xt)
                self.save_info([xt],[ys],0.33,25,self.save)

                loss_func_call += 1

                if ys < zc :
                    xc = np.copy(xt)
                    zc = ys
                    l += 1
                else:
                    stop = False



        if l == 0: self.p += 1

        return xc,zc,loss_func_call

    # Tornado
    def run(self) -> Tuple[np.ndarray,float] :

        ne = 0
        k=1


        # while k & number of function call < total iteration & maximum number of loss function call
        while (k <= self.m_global) and (ne < self.f_call):

            # Compute CGS
            x_inter,loss_value = self.chaotic_global_search(k)

            # 12 * level: number of loss function evaluations
            ne += 4*self.n_level_cgs
            r = self.save_best(x_inter, loss_value)
            print("\n\n=======>   Nb CGS | Nb function calls | Best value on CGS")
            print("=======>",k,"<",self.m_global,"|",ne,"<",self.f_call, " |", loss_value)
            if r : print("=======> !!--!! Congrats, this is a new best global point found !!--!! ")

            nl = 0
            print("--> Nb Local | Nb function calls | Best value on CGS")

            while (nl<self.m_local) and (ne<self.f_call):
                r1,r2=False,False

                if self.n_level_cls >0:
                    # Compute CLS
                    x_inter,loss_value = self.chaotic_local_search(k,nl,x_inter,loss_value)
                    r1 = self.save_best(x_inter, loss_value)
                    ne += self.n_symetric_p*self.n_level_cls

                if self.n_level_cfs >0:
                    # Compute CFS
                    x_inter,loss_value = self.chaotic_fine_search(k,nl,x_inter,loss_value)
                    r2 = self.save_best(x_inter, loss_value)
                    ne += self.n_symetric_p*self.n_level_cfs

                nl += 1

                if (nl%(int(self.m_local/4))==0) :
                    print("-->", nl,"<",self.m_local, "  |", ne,"<",self.f_call, " |", loss_value)
                    if r1 or r2 :
                        print("--> !!--!! Congrats, this is a new best global point found in CLS / CFS ", r1, "/", r2," !!--!!")

            if self.gds:
                self.p=0
                for ng in range(1,6):

                    x_gds,y_gds,loss_func_call = self.gradient_descent_search(ng,x_inter,loss_value)
                    r2 = self.save_best(x_gds, y_gds)

                    ne += loss_func_call

                x_inter,loss_value = x_gds,y_gds


            if self.centers is not None :
                self.centers = np.concatenate([self.centers, x_inter[None]],axis=0)
            else :
                self.centers = x_inter[None]
            k += 1

        if self.return_history :
            return ne, self.centers, self.points, self.values, self.color, self.size,self.best_x,self.best_y
        else :
            return self.best_x, self.best_y

    def save_best(self, x,y):
        if y < self.best_y:
            self.best_x = x
            self.best_y = y
            return True
        return False
