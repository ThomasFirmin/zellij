import numpy as np
import copy

tree_search_algorithm = {"BFS": Breadth_first_search,"DFS": Depth_first_search, "BS": Beam_search, "BestFS":Best_first_search,\
"CBFS":Cyclic_best_first_search,"DBFS":Diverse_best_first_search,"EGS":Epsilon_greedy_search}

class Tree_search():
    def __init__(self,open,max_depth):
        self.open = open
        self.close = []
        self.max_depth = max_depth

    @abc.abstractmethod
    def add(self,c):
        pass

    @abc.abstractmethod
    def get_next(self):
        pass

class Breadth_first_search(Tree_search):

    def __init__(self,open,max_depth,reverse=False):

        super().__init__(open,max_depth)

        self.reverse = reverse

    def add(self,c):
        self.open.append(c)
        self.open = sorted(self.open,reverse=self.reverse,key= lambda x: x.score)[:]

    def get_next(self):

        if len(self.open) > 0:

            idx = len(self.open)

            self.close += self.open

            self.open = []

            return True,self.close[-idx:]

        else:
            return False,-1

class Depth_first_search(Tree_search):

    def __init__(self,open,max_depth,Q=1,reverse=False):

        super().__init__(open,max_depth)

        self.reverse = reverse

        self.Q = Q
        self.next_frontier = []

    def add(self,c):

        self.next_frontier.append(c)

    def get_next(self):

        if len(self.next_frontier) > 0:
            self.open = sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:]+self.open
            self.next_frontier = []

        if len(self.open) > 0:

            idx_min = np.min([len(self.open),self.Q])

            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.open.pop(0)

            return True,self.close[-idx_min:]

        else:
            return False,-1

class Best_first_search(Tree_search):

        def __init__(self,open,max_depth,Q=1,reverse=False):

            super().__init__(open,max_depth)

            self.reverse = reverse

            self.Q = Q
            self.next_frontier = []

        def add(self,c):

            self.next_frontier.append(c)

        def get_next(self):

            if len(self.next_frontier) > 0:
                self.open = sorted(self.open+sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:],reverse=self.reverse,key= lambda x: x.score)
                self.next_frontier = []

            if len(self.open) > 0:

                idx_min = np.min([len(self.open),self.Q])
                self.close += self.open[0:idx_min]

                for _ in range(idx_min):
                    self.open.pop(0)

                return True,self.close[-idx_min:]

            else:
                return False,-1

class Beam_search(Tree_search):

    def __init__(self,open,max_depth,Q=1,reverse=False, beam_length = 10):

        super().__init__(open,max_depth)

        self.reverse = reverse

        self.open = open
        self.next_frontier = []

        self.Q = Q

        self.beam_length = beam_length

    def add(self,c):

        self.next_frontier.append(c)

    def get_next(self):

        if len(self.next_frontier) > 0:
            self.open = sorted(self.next_frontier+sorted(self.open,reverse=self.reverse,key= lambda x: x.score),reverse=self.reverse,key= lambda x: x.score)[:self.beam_length]
            self.next_frontier = []

        if len(self.open) > 0:

            idx_min = np.min([len(self.open),self.Q])
            self.close += self.open[0:idx_min]

            for _ in range(idx_min):
                self.open.pop(0)

            return True,self.close[-idx_min:]

        else:
            return False,-1

class Diverse_best_first_search(Tree_search):

    def __init__(self,open,max_depth,Q=1,reverse=False, P=0.1, T=0.5):

        super().__init__(open,max_depth)

        self.reverse = reverse

        self.open = open
        self.open[0].g_value=1
        self.open[0].min_score=1000000
        self.next_frontier = []

        self.Q = Q
        self.P = P
        self.T = T

    def add(self,c):

        c.g_value = c.min_score
        start = c.father

        while type(start.father) != str:
            c.g_value += start.father.min_score
            start = start.father

        self.next_frontier.append(c)

    def fetch_node(self):

        p_total = 0

        h_values = [i.min_score for i in self.open]
        g_values = [i.g_value for i in self.open]

        p = []

        combination = []

        hmin,hmax = np.min(h_values),np.max(h_values)
        gmin,gmax = np.min(g_values),np.max(g_values)

        if np.random.random() < self.P:
            G = np.random.choice(g_values)
        else:
            G = gmax

        for h,g in zip(h_values,g_values):

            if g > G:
                p.append(0)
            else:
                p.append(self.T**(h-hmin))

            p_total += p[-1]

        idx = np.random.choice(len(self.open),p=np.array(p)/p_total)

        return idx

    def get_next(self):

        if len(self.next_frontier) > 0:
            self.open = sorted(self.next_frontier+sorted(self.open,reverse=self.reverse,key= lambda x: x.score),reverse=self.reverse,key= lambda x: x.score)
            self.next_frontier = []

        if len(self.open) > 0:

            idx = self.fetch_node()
            self.close += [self.open[idx]]
            self.open.pop(idx)

            return True,[self.close[-1]]

        else:
            return False,-1

class Cyclic_best_first_search(Tree_search):

    def __init__(self,open,max_depth,Q=1,reverse=False):

        super().__init__(open,max_depth)

        self.reverse = reverse

        self.next_frontier = []

        self.Q = Q


        self.L = [False]*(self.max_depth+1)
        self.L[0] = True
        self.i = 0
        self.contour = [[] for i in range(self.max_depth+1)]
        self.contour[0] = open

        self.best_scores = float("inf")
        self.first_complete = False

    def add(self,c):

        # Verify if a node must be pruned or not.
        # A node can be pruned only if at least one exploitation has been made
        if not self.first_complete:
            self.next_frontier.append(c)

            if c.level == self.max_depth:
                self.first_complete = True
                self.best_score = c.min_score
        else:
            if c.min_score < self.best_score:
                self.best_score = c.min_score
                self.next_frontier.append(c)

    def get_next(self):

        if len(self.next_frontier) > 0:

            modified_levels = []
            for h in self.next_frontier:

                self.contour[h.level].append(h)
                modified_levels.append(h.level)

                if not self.L[h.level]:
                    self.L[h.level] = True

            modified_levels = np.unique(modified_levels)
            for l in modified_levels:

                self.contour[l] = sorted(self.contour[l],reverse=self.reverse,key= lambda x: x.score)

            self.next_frontier = []

        if np.any(self.L):

            search = True
            found = True

            l = 0
            i = -1

            while l < len(self.L) and search:

                if self.L[l]:

                    if found:
                        i = l
                        found = False

                    if l > self.i:
                        self.i = l
                        search = False

                l += 1

            if search and not found:
                self.i = i

            idx_min = np.min([len(self.contour[self.i]),self.Q])

            self.close += self.contour[self.i][0:idx_min]

            for _ in range(idx_min):
                self.contour[self.i].pop(0)

            if len(self.contour[self.i]) == 0:
                self.L[self.i] = False

            return True,self.close[-idx_min:]

        else:
            return False,-1

class Epsilon_greedy_search(Tree_search):

        def __init__(self,open,max_depth,epsilon=0.1,reverse=False):

            super().__init__(open,max_depth)

            self.reverse = reverse

            self.epsilon = epsilon
            self.next_frontier = []

        def add(self,c):

            self.next_frontier.append(c)

        def get_next(self):

            if len(self.next_frontier) > 0:
                self.open = sorted(self.open+sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:],reverse=self.reverse,key= lambda x: x.score)
                self.next_frontier = []

            if len(self.open) > 0:

                if np.random.random() > self.epsilon:

                    self.close += [self.open[0]]
                    self.open.pop(0)

                else:

                    if len(self.open) > 1:
                        idx = np.random.randint(1,len(self.open))
                        self.close += [self.open[idx]]
                        self.open.pop(idx)
                    else:
                        self.close += [self.open[0]]
                        self.open.pop(0)


                return True,[self.close[-1]]

            else:
                return False,-1
