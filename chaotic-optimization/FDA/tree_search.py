import numpy as np
import abc
from collections import deque

class Tree_search():
    def __init__(self,open):
        self.open = [open]
        self.close = []

    @abc.abstractmethod
    def add(self,c):
        pass

    @abc.abstractmethod
    def get_next(self):
        pass

class Breadth_first_search(Tree_search):

    def __init__(self,open,n_fractal):

        super().__init__(open)

        self.n_fractal = n_fractal

        self.actual_level = self.open[0].level

        self.level_size = self.n_fractal**(self.actual_level+1)
        self.next_frontier = []

        self.beam_length = 1

    def add(self,c):

        self.next_frontier.append(c)

        if len(self.next_frontier) == self.level_size:

            self.actual_level = c.level

            self.level_size = self.n_fractal**(self.actual_level+1)

            self.open = sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:]

            self.next_frontier = []

    def get_next(self):

        if len(self.open) > 0:

            if len(self.open) < self.beam_length:
                idx = len(self.open)
            else:
                idx = self.beam_length

            self.how_many = idx
            self.close += self.open[:idx]

            for _ in range(idx):
                self.open.pop(0)

            return True,self.close[-idx:]

        else:
            return False,-1

class Depth_first_search(Tree_search):

    def __init__(self,open,n_fractal):

        super().__init__(open)

        self.n_fractal = n_fractal

        self.actual_level = self.open[0].level

        self.next_frontier = []

        self.beam_length = 1

    def add(self,c):

        self.next_frontier.append(c)

        if len(self.next_frontier) == self.n_fractal*self.beam_length:

            self.open = sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:] + self.open

            self.next_frontier = []

    def get_next(self):

        if len(self.open) > 0:

            if len(self.open) < self.beam_length:
                idx = len(self.open)
            else:
                idx = self.beam_length

            self.how_many = idx
            self.close += self.open[:idx]

            for _ in range(idx):
                self.open.pop(0)

            return True,self.close[-idx:]

        else:
            return False,-1

class Best_First_search(Tree_search):

        def __init__(self,open,n_fractal):

            super().__init__(open)

            self.n_fractal = n_fractal

            self.actual_level = self.open[0].level

            self.next_frontier = []

            self.how_many = n_fractal
            self.beam_length = 1

        def add(self,c):

            self.next_frontier.append(c)

            if len(self.next_frontier) == self.how_many*self.beam_length:

                self.open = sorted(self.open+sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score),reverse=self.reverse,key= lambda x: x.score)

                self.next_frontier = []

        def get_next(self):

            if len(self.open) > 0:

                if len(self.open) < self.beam_length:
                    idx = len(self.open)
                else:
                    idx = self.beam_length

                self.how_many = idx
                self.close += self.open[:idx]

                for _ in range(idx):
                    self.open.pop(0)

                return True,self.close[-idx:]

            else:
                return False,-1

class Beam_search(Tree_search):

    def __init__(self,open,n_fractal, beam_length = 2):

        super().__init__(open)

        self.n_fractal = n_fractal
        self.actual_level = self.open[0].level

        self.next_frontier = []

        self.how_many = n_fractal

        self.beam_length = beam_length

    def add(self,c):

        self.next_frontier.append(c)

        if len(self.next_frontier) == self.how_many*self.beam_length:

            self.open = sorted(self.open+sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:],reverse=self.reverse,key= lambda x: x.score)

            self.next_frontier = []

    def get_next(self):

        if len(self.open) > 0:

            if len(self.open) < self.beam_length:
                idx = len(self.open)
            else:
                idx = self.beam_length

            self.how_many = idx
            self.close += self.open[:idx]

            for _ in range(idx):
                self.open.pop(0)

            return True,self.close[-idx:]

        else:
            return False,-1
