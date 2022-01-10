import numpy as np
from zellij.strategies.utils.chaos_map import Chaos_map

class CGS(Metaheuristic):

    def __init__(self, loss_func, search_space, f_calls, level, chaos_map, create=False, save=False, verbose=True):

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.level = level

        if create and type(chaos_map)==str:
            self.map = Chaos_map(chaos_map,self.level,self.search_space.n_variables)
        elif type(chaos_map) != str:
            self.map = chaos_map

        #############
        # VARIABLES #
        #############

        self.up_bounds = np.array([1 for _ in range(self.search_space.n_variables)])
        self.lo_bounds = np.array([0 for _ in range(self.search_space.n_variables)])

        self.up_plus_lo = self.up_bounds + self.lo_bounds
        self.up_m_lo = self.up_bounds - self.lo_bounds

        self.center = np.multiply(0.5,self.up_plus_lo)
        self.radius = np.multiply(0.5,self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

        if self.save:
            self.create_file("algorithm")

    def run(self,shift=1, n_process=1,f_calls_init = 0):

        self.k = shift

        # For each level of chaos
        shift_map = (self.k-1)*self.level
        points = np.empty((0,self.search_space.n_variables),dtype=float)

        n_points = f_calls_init
        l = 0

        while l < self.level and n_points < self.f_calls:

            # Randomly select a parameter index of a solution
            d=np.random.randint(self.search_space.n_variables)

            # Select chaotic_variables among the choatic map
            y = self.map.chaos_map[l+shift_map]*self.map.chaos_map[self.k-1]
            # Apply 3 transformations on the selected chaotic variables
            r_mul_y = np.multiply(self.up_m_lo,y)

            # xx = [np.add(self.center,r_mul_y), np.add(self.center,np.multiply(self.radius,np.multiply(2,y)-1)), np.subtract(self.up_bounds,r_mul_y)]

            # for each transformation of the chaotic variable
            # for x in xx:
            #
            #     x_ = np.subtract(self.up_plus_lo,x)
            #     sym = np.matrix([x,x,x_,x_])
            #     sym[1,d] = x_[d]
            #     sym[3,d] = x[d]
            #     points = np.append(points,sym,axis=0)
            #     n_points += 4

            xx = [self.lo_bounds+r_mul_y,self.up_bounds-r_mul_y]

            # for each transformation of the chaotic variable
            sym = np.array([xx[0],xx[1],xx[0],xx[1]])
            sym[2,d] = xx[1][d]
            sym[3,d] = xx[0][d]

            points = np.append(points,sym,axis=0)
            n_points += 4

            l+= 1

        ys = self.loss_func.evaluate(self.search_space.convert_to_continuous(points,True),filename = self.filename, algorithm="CLS")
        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        return points[idx], ys[idx], points, ys

class CLS(Metaheuristic):

    def __init__(self,loss_func,search_space,f_calls,level,polygon,chaos_map,red_rate=0.5,save=False,verbose=True):

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.level = level
        self.polygon = polygon
        self.map = chaos_map
        self.red_rate = red_rate

        #############
        # VARIABLES #
        #############

        self.up_bounds = np.array([1 for _ in self.search_space.values])
        self.lo_bounds = np.array([0 for _ in self.search_space.values])

        self.up_plus_lo = self.up_bounds+self.lo_bounds
        self.up_m_lo = self.up_bounds - self.lo_bounds

        self.center = np.multiply(0.5,self.up_plus_lo)
        self.radius = np.multiply(0.5,self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

        trigo_val = 2*np.pi/self.polygon
        self.H = [np.zeros(self.polygon),np.zeros(self.polygon)]

        for i in range(1,self.polygon+1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i-1] = np.cos(trigo_val*i)
            self.H[1][i-1] = np.sin(trigo_val*i)

        if self.save:
            self.create_file("algorithm")

    def run(self,X0,Y0,chaos_level=0,shift=1, n_process=1, f_calls_init = 0):

        self.X0 = X0
        self.Y0 = Y0
        self.k = shift
        self.chaos_level = chaos_level

        # Initialization
        shift = self.chaos_level*(self.k-1)*self.level
        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.up_bounds-self.X0,self.X0-self.lo_bounds)

        # Local search area radius
        Rl = self.radius*self.red_rate

        center_m_solution = self.center - self.X0
        points = np.empty((0,self.search_space.n_variables),dtype=float)

        n_points = f_calls_init
        l = 0
        # for each level of chaos
        while l < self.level and n_points < self.f_calls:

            # Decomposition vector
            d = np.random.randint(self.search_space.n_variables)

            # zoom speed
            gamma = 1/(10**(2*self.red_rate*l)*(l+1))

            # for each parameter of a solution determine the improved radius
            xx = np.minimum(gamma*Rl,db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(xx,self.map.chaos_map[shift+l]),np.multiply(xx,self.map.inverted_choas_map[shift+l])]

            # For both chaotic variable
            for x in xv :
                xi = np.outer(self.H[1],x)
                xi[:,d] = x[d]*self.H[0]
                xt = self.X0 + xi

                points = np.append(points,xt,axis=0)
                n_points += self.polygon

            l+= 1

        ys = self.loss_func.evaluate(self.search_space.convert_to_continuous(points,True), filename = self.filename, algorithm="CLS")

        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        if ys[idx].any() < self.Y0 :
            return points[idx], ys[idx], points, ys
        else :
            return [self.X0], [self.Y0], points, ys

class CFS(Metaheuristic):

    def __init__(self,loss_func,search_space,f_calls,level,polygon,chaos_map,red_rate=0.5,save=False,verbose=True):

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.level = level
        self.polygon = polygon
        self.map = chaos_map
        self.red_rate = red_rate

        #############
        # VARIABLES #
        #############

        self.up_bounds = np.array([1 for _ in self.search_space.values])
        self.lo_bounds = np.array([0 for _ in self.search_space.values])

        self.up_plus_lo = self.up_bounds+self.lo_bounds
        self.up_m_lo = self.up_bounds - self.lo_bounds

        self.center = np.multiply(0.5,self.up_plus_lo)
        self.radius = np.multiply(0.5,self.up_m_lo)
        self.center_m_lo_bounds = self.center - self.lo_bounds

        trigo_val = 2*np.pi/self.polygon
        self.H = [np.zeros(self.polygon),np.zeros(self.polygon)]

        for i in range(1,self.polygon+1):
            # Initialize trigonometric part of symetric variables (CLS & CFS)
            self.H[0][i-1] = np.cos(trigo_val*i)
            self.H[1][i-1] = np.sin(trigo_val*i)

        if self.save:
            self.create_file("algorithm")

    def stochastic_round(self, solution, k):

        r = np.random.uniform(-1,1,len(solution))
        # perturbation on CFS zoom
        z = np.round(solution)+(k%2)*r

        return z

    def run(self,X0,Y0,chaos_level=0,shift=1, n_process=1, f_calls_init = 0):

        self.X0 = X0
        self.Y0 = Y0
        self.k = shift
        self.chaos_level = chaos_level

        shift = self.chaos_level*(self.k-1)*self.level

        y = self.map.chaos_map[shift]*self.map.chaos_map[self.k-1]
        # Limits of the search area, if parameter greater than center, then = 1 else = -1, used to avoid overflow
        db = np.minimum(self.up_bounds-self.X0,self.X0-self.lo_bounds)

        r_g = np.zeros(self.search_space.n_variables)

        # Randomly select the reduction rate
        #red_rate = random.random()*0.5

        # Local search area radius
        Rl = self.radius*self.red_rate

        xc = self.X0
        zc = self.Y0

        center_m_solution = self.center - self.X0
        points = np.empty((0,self.search_space.n_variables),dtype=float)

        n_points = f_calls_init
        l = 0
        # for each level of chaos
        while l < self.level and n_points < self.f_calls:
            # Decomposition vector
            d = np.random.randint(self.search_space.n_variables)

            # Exponential Zoom factor on the search window
            pc = 10**(l+1)

            # Compute the error/the perturbation applied to the solution
            error_g = np.absolute(self.X0-(self.stochastic_round(pc*self.X0,shift+l)/pc))

            r = np.random.random()

            # for each parameter of a solution determine the improved radius
            r_g = np.minimum((Rl*error_g)/(l**2+1),db)

            # Compute both chaotic variable of the polygonal model thanks to a chaotic map
            xv = [np.multiply(r_g,y),np.multiply(r_g,y)]

            # For both chaotic variable
            for x in xv :
                xi = np.outer(self.H[1],x)
                xi[:,d] = x[d]*self.H[0]
                xt = self.X0 + xi

                points = np.append(points,xt,axis=0)
                n_points += self.polygon

            l += 1

        ys = self.loss_func.evaluate(self.search_space.convert_to_continuous(points,True), filename = self.filename, algorithm="CFS")

        ys = np.array(ys)
        idx = np.array(np.argsort(ys))[:n_process]

        if ys[idx].any() < self.Y0 :
            return points[idx], ys[idx], points, ys
        else :
            return [self.X0], [self.Y0], points, ys

class Chaotic_optimization(Metaheuristic):

    def __init__(self, loss_func, search_space, f_calls,chaos_map="henon", exploration_ratio = 0.80,levels = (32,8,2), polygon=4, red_rate=0.5,save=False,verbose=True):

        ##############
        # PARAMETERS #
        ##############

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.chaos_map = chaos_map
        self.exploration_ratio = exploration_ratio
        self.polygon = polygon
        self.red_rate = red_rate

        self.CGS_level = levels[0]
        self.CLS_level = levels[1]
        self.CFS_level = levels[2]

        self.save = save

        #############
        # VARIABLES #
        #############

        if self.CGS_level > 0:
            if self.CLS_level != 0 or self.CFS_level !=0:
                self.iterations = np.ceil((self.f_calls*self.exploration_ratio)/(4*self.CGS_level))
                self.inner_iterations = np.ceil((self.f_calls*(1-self.exploration_ratio))/((self.CLS_level+self.CFS_level)*self.polygon*self.iterations))
            else:
                self.iterations = np.ceil(self.f_calls/(4*self.CGS_level))
                self.inner_iterations = 0
        else:
            raise ValueError("CGS level must be > 0")

        if type(chaos_map) == str:
            self.map_size = int(np.max([self.iterations*self.CGS_level,self.iterations*self.inner_iterations*self.CLS_level,self.iterations*self.inner_iterations*self.CFS_level]))
        else:
            self.map_size = int(np.ceil(np.max([self.iterations*self.CGS_level,  self.iterations*self.inner_iterations*self.CLS_level,self.iterations*self.inner_iterations*self.CFS_level])/len(self.chaos_map)))

        self.map = Chaos_map(self.chaos_map,self.map_size,self.search_space.n_variables)

        self.X = []
        self.Y = []
        self.min = float("inf")

        if self.save:
            self.create_file()

        if self.verbose:
            print(str(self))

    def run(self,n_process=1):

        cgs = CGS(self.loss_func,self.search_space,self.f_calls,self.CGS_level,self.map)
        cls = CLS(self.loss_func,self.search_space,self.f_calls,self.CLS_level,self.polygon,self.map,self.red_rate)
        cfs = CFS(self.loss_func,self.search_space,self.f_calls,self.CFS_level,self.polygon,self.map,self.red_rate)
        cgs.filename, cls.filename, cfs.filename = self.filename, self.filename, self.filename

        k = 1

        while k <= self.iterations and self.loss_func.calls < self.f_calls:

            if self.CGS_level > 0:
                x_inter,loss_value,X,Y = cgs.run(k,f_calls_init = f_calls)

            else:
                x_inter = [np.random.random(self.search_space.n_variables)]
                loss_value = self.loss_func.evaluate(x_inter)

            if self.verbose:
                out = "\n\n=======>   Iterations | Loss function calls | Best value from CGS"
                out += "\n=======>"+str(k)+"<"+str(self.iterations)+"|"+str(self.loss_func.calls)+"<"+str(self.f_calls)+" |"+str(loss_value)
                if self.loss_func.new_best : out += "\n=======> !!--!! New best solution found !!--!! "
                print(out)


            inner = 0

            while inner < self.inner_iterations and self.loss_func.calls <self.f_calls:

                if self.CLS_level > 0:
                    x_inter,loss_value,X,Y = cls.run(x_inter[0],loss_value[0],inner,k, f_calls_init = self.loss_func.calls)

                if self.CFS_level > 0:
                    x_inter,loss_value,X,Y = cfs.run(x_inter[0],loss_value[0],inner,k, f_calls_init =  self.loss_func.calls)

                if self.verbose:
                    out = "-->"+str(inner)+"<"+str(self.inner_iterations)+"  |"+str(f_calls)+"<"+str(self.f_calls)+" |"+str(loss_value)
                    out += "\n=======>"+str(k)+"<"+str(self.iterations)+"|"+str(f_calls)+"<"+str(self.f_calls)+" |"+str(loss_value)
                    if self.loss_func.new_best : out += "\n=======> !!--!! New best solution found !!--!! "
                    print(out)

                inner += 1
            k += 1

        ind_min = np.argsort(self.Y)[0:n_process]
        min = np.array(self.Y)[ind_min].tolist()
        best = np.array(self.X)[ind_min].tolist()

        return best,min

    def show(self, filename=None):

        import matplotlib.pyplot as plt
        import pandas as pd

        if filename == None:
            scores = np.array(self.Y)
        else:
            data = pd.read_table(filename,sep=",",decimal   =".")
            scores = data["loss_value"].to_numpy()

        if filename != None:
            self.search_space.show(data.iloc[:,0:self.search_space.n_variables],scores)
        else:
            self.search_space.show(pd.DataFrame(self.X, columns=self.search_space.label),self.Y)

    def __str__(self):
        return f"Max Loss function calls:{self.f_calls}\nDimensions:{self.search_space.n_variables}\nExploration/Exploitation:{self.exploration_ratio}|{1-self.exploration_ratio}\nRegular polygon:{self.polygon}\nZoom:{self.red_rate}\nIterations:\n\tGlobal:{self.iterations}\n\tInner:{self.inner_iterations}\nChaos Levels:\n\tCGS:{self.CGS_level}\n\tCLS:{self.CLS_level}\n\tCFS:{self.CFS_level}\nMap size:{self.map_size}x{self.search_space.n_variables}"
