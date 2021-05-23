import numpy as np
from fractal import Fractal

class Hypersphere(Fractal):

    def __init__(self,father,lo_bounds,up_bounds,center,radius,inflation,level,id,children=[],score=None,save=False):

        super().__init__(lo_bounds,up_bounds,father,level,id,children,score,save)

        self.dim = len(self.up_bounds)

        self.center = center
        self.radius = radius

        self.inflation = inflation


    def create_children(self,start_id=0):

        level =  self.level+1

        r_p = self.radius/(1+np.sqrt(2))

        n_h = start_id
        for i in range(self.dim):

            n_h += 1
            center = np.copy(self.center)
            center[i] += ((-1)**i)*(self.radius[i]-r_p[i])
            center[i] = np.min([center[i],self.up_bounds[i]])
            center[i] = np.max([center[i],self.lo_bounds[i]])

            h = Hypersphere(self, self.lo_bounds, self.up_bounds, center, r_p*self.inflation, self.inflation, level,n_h,save=self.save)
            self.children.append(h)

            n_h += 1
            center = np.copy(self.center)
            center[i] -= ((-1)**i)*(self.radius[i]-r_p[i])
            center[i] = np.max([center[i],self.lo_bounds[i]])
            center[i] = np.min([center[i],self.up_bounds[i]])

            h = Hypersphere(self,self.lo_bounds,self.up_bounds, center, r_p*self.inflation, self.inflation, level,n_h,save=self.save)
            self.children.append(h)

    def isin(self,solution,dim):
        dist = np.sqrt(np.sum((np.square(solution-self.center))))

        return dist < self.radius[dim]

    def essential_infos(self):
        return [self.center,self.radius,self.lo_bounds,self.up_bounds]

    def __repr__(self):
        if type(self.father) == str:
            id = "GOD"
        else:
            id = str(self.father.id)

        return "ID: "+str(self.id)+","+id+"level:"+str(self.level)+" score: "+str(self.score)+"\n"
