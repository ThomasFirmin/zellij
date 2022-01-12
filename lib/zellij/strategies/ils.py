import numpy as np

# Intensive local search
class ILS(Metaheuristic):

    """ILS

    Intensive local search is an exploitation algorithm comming from original FDA paper.
    It evaluate a point in each dimension arround an initial solution.
    Distance of the computed point to the initial one is decreasing according to a reduction rate.
    At each iteration the algorithm moves to the best solution found.

    Attributes
    ----------

    up_bounds : list
        List of float containing the upper bounds of the search space converted to continuous.
    lo_bounds : list
        List of float containing the lower bounds of the search space converted to continuous.
    center : float
        List of floats containing the coordinates of the search space center converted to continuous.
    radius : float
        List of floats containing the radius for each dimensions of the search space converted to continuous.
    red_rate : float
        determine the step reduction rate ate each iteration.
    precision : float
        dtermine the stopping criterion. When the step is lower than <precision> the algorithm stops.

    Methods
    -------

    run(self, n_process=1)
        Runs ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, loss_func, search_space, f_calls,red_rate=0.5,precision=1e-5,save=False,verbose=True):

        """__init__(self, loss_func, search_space, f_calls,save=False,verbose=True)

        Initialize HHS class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        red_rate : float, default=0.5
            determine the step reduction rate ate each iteration.
            
        precision : float, default=1e-5
            dtermine the stopping criterion. When the step is lower than <precision> the algorithm stops.

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.red_rate = red_rate
        self.precision = precision

        self.upper = np.array([1 for _ in self.search_space.values])
        self.lower = np.array([0 for _ in self.search_space.values])

        self.up_bounds = np.array(self.search_space.convert_to_continuous([[x[1] for x in self.search_space.values]],sub_values=True)[0])
        self.lo_bounds = np.array(self.search_space.convert_to_continuous([[x[0] for x in self.search_space.values]],sub_values=True)[0])

        up_m_lo = self.up_bounds - self.lo_bounds
        self.radius = up_m_lo/2

    def run(self,X0,Y0,n_process=1,save=False):

        X0 = np.array(self.search_space.convert_to_continuous([X0],sub_values=True)[0])

        loss_call = 0

        scores = [0]*3
        solutions = [np.copy(X0),np.copy(X0),np.copy(X0)]
        scores[0] = Y0

        step = np.max(self.radius)

        while step > self.precision and loss_call < self.f_calls:
            i = 0
            while i < self.search_space.n_variables and loss_call < self.f_calls:

                walk = solutions[0][i] + step
                db = np.min([self.upper[i],walk])
                solutions[1][i] = db
                scores[1] = self.loss_func(self.search_space.convert_to_continuous([solutions[1]],True,True))[0]

                walk = solutions[0][i] - step
                db = np.max([self.lower[i],walk])
                solutions[2][i] = db

                scores[2] = self.loss_func(self.search_space.convert_to_continuous([solutions[2]],True,True))[0]

                min_index = np.argmin(scores)
                solutions = [np.copy(solutions[min_index]),np.copy(solutions[min_index]),np.copy(solutions[min_index])]

                i+= 1
                loss_call += 2

            step = self.red_rate * step
