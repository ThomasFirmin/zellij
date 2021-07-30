import numpy as np

# Hypersphere Heuristic Search
class HHS:
    def __init__(self, loss_func, search_space, f_calls):
        self.loss_func = loss_func
        self.search_space = search_space

        self.f_calls = f_calls

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
