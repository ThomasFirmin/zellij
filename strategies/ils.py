import numpy as np

# Intensive local search
class ILS:

    def __init__(self, loss_func, search_space, f_calls,red_rate=0.5,precision=1e-5):

        self.loss_func = loss_func
        self.search_space = search_space
        self.f_calls = f_calls
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
