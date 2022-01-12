import numpy as np

# Hypersphere Heuristic Search
class HHS(Metaheuristic):

    """HHS

    Hypersphere Heuristic Search  is an exploration algorithm comming from original FDA paper.
    It used to evaluate the center of an Hypersphere, and fixed points on each dimension arround this center.

    It works on a continuous searh space.

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

    Methods
    -------

    run(self, n_process=1)
        Runs HHS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, loss_func, search_space, f_calls,save=False,verbose=True):

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

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(loss_func,search_space,f_calls,save,verbose)

        self.up_bounds = np.array([1 for _ in self.search_space.values])
        self.lo_bounds = np.array([0 for _ in self.search_space.values])
        up_m_lo = self.up_bounds - self.lo_bounds
        self.radius = up_m_lo/2

        up_p_lo = self.up_bounds + self.lo_bounds
        self.center = up_p_lo/2

    def run(self,n_process=1):
        loss_call = 0
        i = 0
        while i < self.search_space.n_variables and loss_call<self.f_calls:
            inf = np.copy(self.center)
            sup = np.copy(self.center)

            inf[i] = np.max([self.center[i]-self.radius[i]/np.sqrt(self.search_space.n_variables),self.lo_bounds[i]])
            sup[i] = np.min([self.center[i]+self.radius[i]/np.sqrt(self.search_space.n_variables),self.up_bounds[i]])

            score1 = self.loss_func(self.search_space.convert_to_continuous([inf],True))
            score2 = self.loss_func(self.search_space.convert_to_continuous([sup],True))

            loss_call += 2
