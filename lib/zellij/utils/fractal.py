import numpy as np
import abc
import copy

class Fractal(object):
    """Fractal

    Fractal is an abstract class used in Fractal Decomposition. This class is used to build a rooted tree of fractals. Each object contains a reference to its father, references to its children, its bounds,\
    and its heuristic value (the score) computed after each exploration. Fractals are simplified and continuous subspaces, builded thanks to the original SearchSpace object.

    Attributes
    ----------

    lo_bounds : list[float]
        Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

    up_bounds : list[float]
        Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

    id : int
        Identifier of a fractal. Combined to the id of itf parents, the id is unique.

    father : Fractal
        Reference to the parent of the current fractal.

    children : list[Fractal]
        References to all children of the current fractal.

    score : {float, int}
        Heuristic value associated to the fractal after an exploration

    level : int
        Current level of the fractal in the rooted tree. See Tree_search.

    min_score : {float, int}
        Score associated to the best found solution inside the fractal

    best_sol : list[{float, int, str}]
        Best found solution inside the fractal in its mixed format.

    best_sol_c : list[float]
        Best found solution inside the fractal in its continuous format.

    all_solutions : float
        Historic of all evaluated solutions inside the fractal.

    all_scores : float
        Historic of all evaluated scores inside the fractal..

    Methods
    -------
     __init__(self,lo_bounds,up_bounds,father,level,id,children=[],score=None)
        Initialize Fractal class

    add_point(self,score, solution)
        Adds a point to the fractal

    create_children(self)
        Abstract method which defines how fractal children should be created


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal rooted tree.
    SearchSpace : Initial search space used to build fractal.
    Hypercube : Inherited Fractal type
    Hypersphere : Inherited Fractal type
    """

    def __init__(self,lo_bounds,up_bounds,father,level,id,children=[],score=None):

        """__init__(self, open, max_depth, Q=1, reverse=False)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal
            Reference to the parent of the current fractal.

        level : int
            Current level of the fractal in the rooted tree. See Tree_search.

        id : int
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no children are given, they will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """


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

        """add_point(self,score, solution)

        This method adds a point associated to its evaluation by the loss function (f(solution)=score), to the historic of the fractal,\
         and determine if this point is the best one among all evaluated points inside the fractal.

        Parameters
        ----------
        score : {int, float}
            Score associated to the evaluated solution

        solution : list[{int, float, str}]
            It corresponds to a point in the mixed format inside the fractal and associated to its evaluation (score).

        """

        for sol,sco in zip(solution,score):

            self.solutions.append(sol)
            self.all_scores.append(sco)

            if sco < self.min_score:
                self.min_score = sco
                self.best_sol = sol

    @abc.abstractmethod
    def create_children(self):
        """create_children(self)

        Abstract method which will create children of the current Fractal object, according to certain rules (Hypercube, Hypersphere...)

        """
        pass

class Hypercube(Fractal):
    """Hypercube

    The hypercube is a basic hypervolume to decompose the SearchSpace. It's also one of the most computationally inefficient in high dimension.\
    The decomposition complexity of an Hypercube with equalsize Hypercubes, is equal to $2^d$, d is the dimension.\
    However building a single hypercube is low complexity task, and the space coverage is very good, 100% of the initial hypercube is covered by its children.

    Attributes
    ----------

    dim : int
        Number of dimensions

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hypercubes based on the current Hypercube.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal rooted tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypersphere : Another hypervolume, with different properties
    """

    def __init__(self,father,lo_bounds,up_bounds,level,id,children=[],score=None):

        """__init__(self,father,lo_bounds,up_bounds,level,id,children=[],score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal
            Reference to the parent of the current fractal.

        level : int
            Current level of the fractal in the rooted tree. See Tree_search.

        id : int
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no children are given, they will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

        super().__init__(lo_bounds,up_bounds,father,level,id,children,score)

        self.dim = len(self.up_bounds)

    def create_children(self):

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

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

    """Hypersphere

    The Hypersphere is a basic hypervolume to decompose the SearchSpace. It is one of the most computationally efficient, to decompose the SearchSpace.
    To decompose an hypersphere by equalsize hypersphere the complexity is equal to $2*d$, d is the dimension, moreover building an hypersphere is easy, it only needs a center and its radius.
    However the space coverage is poor, indeed the volume of an hypersphere tends to 0, when the dimension tends to infinity. To partially tackle this problem, an inflation rate allows bigger hypersphere,
    but this will create overlapping hypervolumes.

    Attributes
    ----------

    dim : int
        Number of dimensions

    inflation : float
        Inflation rate of hyperspheres. Be carefull a too large inflation can result to hypersphere with identical center and radius.

    center : list[float]
        List of floats containing the coordinates

    radius : list[float]
        List of floats containing the radius for each dimension (in case the initial SearchSpace is not an hypercube).

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hyperspheres based on the current Hypersphere.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal rooted tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties
    """

    def __init__(self,father,lo_bounds,up_bounds,level,id,inflation=1.75,children=[],score=None):

        """__init__(self,father,lo_bounds,up_bounds,level,id,children=[],score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal
            Reference to the parent of the current fractal.

        level : int
            Current level of the fractal in the rooted tree. See Tree_search.

        id : int
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no children are given, they will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

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

        """create_children(self)

        Method which defines how to build children Hypercubes based on the current Hypercube.
        It uses Hyperplan bisecting to build children. To build an Hypercube, it only requires lower and upper bounds.

        """

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

    """Direct

    Dividing Rectangles. This section must be completed.

    Attributes
    ----------

    dim : int
        Number of dimensions

    Methods
    -------

    create_children(self)
        Method which defines how to build children Hyper-rectangles based on the current Hyper-rectangle.


    See Also
    --------
    LossFunc : Defines what a loss function is
    Tree_search : Defines how to explore and exploit a fractal rooted tree.
    SearchSpace : Initial search space used to build fractal.
    Fractal : Parent class. Basic object to define what a fractal is.
    Hypercube : Another hypervolume, with different properties
    """

    def __init__(self,father,lo_bounds,up_bounds,level,id=1.75,children=[],score=None):

        """__init__(self,father,lo_bounds,up_bounds,level,id,children=[],score=None)

        Parameters
        ----------
        lo_bounds : list[float]
            Contains the lower bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        up_bounds : list[float]
            Contains the upper bounds for each dimension of the fractal. Each fractal is bounded by its circumscribed hypercube.

        father : Fractal
            Reference to the parent of the current fractal.

        level : int
            Current level of the fractal in the rooted tree. See Tree_search.

        id : int
            Identifier of a fractal. Combined to the id of itf parents, the id is unique.

        children : list[Fractal], default=[]
            References to all children of the current fractal. If no children are given, they will be built by the method create_children during the tree building.

        score : {float, int}, default=None
            Heuristic value associated to the fractal after an exploration.  If no score is given, it will be built after the execution of an exploration strategy inside the fractal.

        """

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
