import numpy as np
import abc
import copy

class Fractal(object):

    def __init__(self,lo_bounds,up_bounds,father,level,id,children=[],score=None):


        self.lo_bounds = np.array(lo_bounds)
        self.up_bounds = np.array(up_bounds)

        self.id = id
        self.father = father
        self.children = []
        self.score = score
        self.level = level

        self.min_score = float("inf")
        self.best_sol = None
        self.best_sol_c = None
        self.solutions = []
        self.all_scores = []

    def add_point(self,score, solution):

        for sol,sco in zip(solution,score):

            self.solutions.append(sol)
            self.all_scores.append(sco)

            if sco < self.min_score:
                self.min_score = sco
                self.best_sol = sol

    @abc.abstractmethod
    def create_children(self):
        pass

class Hypercube(Fractal):

    def __init__(self,father,lo_bounds,up_bounds,level,id,children=[],score=None):

        super().__init__(lo_bounds,up_bounds,father,level,id,children,score)

        self.dim = len(self.up_bounds)

    def create_children(self):

        level =  self.level+1

        up_m_lo = self.up_bounds - self.lo_bounds
        radius = np.abs(up_m_lo/2)
        bounds = [[self.lo_bounds,self.up_bounds]]

        # Hyperplan bisecting
        next_b = []
        for i in range(self.dim):
            next_b = []
            for b in bounds:

                # First part
                up = np.copy(b[1])
                up[i] = b[0][i] + radius[i]
                next_b.append([np.copy(b[0]),np.copy(up)])

                # Second part
                low = np.copy(b[0])
                low[i] = b[1][i]-radius[i]
                next_b.append([np.copy(low),np.copy(b[1])])

            bounds = copy.deepcopy(next_b)

        # Create Hypercube
        n_h = 0
        for b in bounds:
            h = Hypercube(self, b[0], b[1], level, n_h)
            self.children.append(h)
            n_h += 1

    def __repr__(self):
        if type(self.father) == str:
            id = "GOD"
        else:
            id = str(self.father.id)

        return "ID: "+str(self.id)+" son of "+id+"\n"+"BOUNDS: "+str(self.lo_bounds)+"|"+str(self.up_bounds)+"\n"

class Hypersphere(Fractal):

    def __init__(self,father,lo_bounds,up_bounds,level,id,inflation=1.75,children=[],score=None):

        super().__init__(lo_bounds,up_bounds,father,level,id,children,score)

        self.dim = len(self.up_bounds)

        up_m_lo = self.up_bounds - self.lo_bounds
        center = self.lo_bounds + (up_m_lo)/2
        radius = np.abs(up_m_lo/2)

        self.center = center
        self.radius = radius


        self.inflation = inflation

        self.radius = self.radius*self.inflation

    def create_children(self):

        level =  self.level+1

        r_p = self.radius/(1+np.sqrt(2))

        n_h = 0
        for i in range(self.dim):

            n_h += 1
            center = np.copy(self.center)
            center[i] += ((-1)**i)*(self.radius[i]-r_p[i])

            lo = np.maximum(center-r_p,self.lo_bounds)
            up = np.minimum(center+r_p,self.up_bounds)

            h = Hypersphere(self, lo, up, level,n_h,inflation=self.inflation)
            self.children.append(h)

            n_h += 1
            center = np.copy(self.center)
            center[i] -= ((-1)**i)*(self.radius[i]-r_p[i])

            lo = np.maximum(center-r_p,self.lo_bounds)
            up = np.minimum(center+r_p,self.up_bounds)

            h = Hypersphere(self, lo, up, level,n_h,inflation=self.inflation)
            self.children.append(h)

    def __repr__(self):
        if type(self.father) == str:
            id = "GOD"
        else:
            id = str(self.father.id)

        return "ID: "+str(self.id)+" son of "+id+" at level "+str(self.level)+"\n"+"BOUNDS: "+str(self.lo_bounds)+"|"+str(self.up_bounds)+"\n"

class Direct(Fractal):

    def __init__(self,father,lo_bounds,up_bounds,level,id=1.75,children=[],score=None):

        super().__init__(lo_bounds,up_bounds,father,level,id,children,score)

        self.dim = len(self.up_bounds)

    def create_children(self):

        level =  self.level+1

        n_h = 0

        u_m_l = self.up_bounds-self.lo_bounds

        i = np.argmax(u_m_l)

        new_val = u_m_l[i]/2

        lo = np.copy(self.lo_bounds)
        up = np.copy(self.up_bounds)
        up[i] -= new_val

        h = Direct(self, lo, up, level,n_h)
        self.children.append(h)

        n_h += 1

        lo = np.copy(self.lo_bounds)
        up = np.copy(self.up_bounds)
        lo[i] += new_val

        h = Direct(self, lo, up, level,n_h)
        self.children.append(h)

    def __repr__(self):
        if type(self.father) == str:
            id = "GOD"
        else:
            id = str(self.father.id)

        return "ID: "+str(self.id)+" son of "+id+" at level "+str(self.level)+"\n"+"BOUNDS: "+str(self.lo_bounds)+"|"+str(self.up_bounds)+"\n"


fractal_list = {"hypersphere": Hypersphere,"hypercube": Hypercube,"direct":Direct}
