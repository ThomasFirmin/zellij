from abc import abstractmethod, ABC
from typing import Optional, Tuple
import numpy as np


class SolutionHandler(ABC):
    _handlers = []

    def __init__(self) -> None:
        super().__init__()

    def add(self, handler: "SolutionHandler"):
        self._handlers.append(handler)
        return self

    @abstractmethod
    def apply(self, X: Optional[list] = None, Y: Optional[np.ndarray] = None):
        pass

    def __call__(self,  X: Optional[list] = None, Y: Optional[np.ndarray] = None):
        for sub_handler in self._handlers:
            X, Y = sub_handler(X, Y)
        return self.apply(X, Y)


class InjectionSolutionHandler(SolutionHandler):
    _frame: list[Tuple[list[float], float]]
    _frame_capacity: int
    _store_factor: float

    @property
    def frame_size(self):
        return len(self._frame)

    @property
    def frame_capacity(self):
        return self._frame_capacity

    @property
    def frame_available(self):
        return self._frame_capacity-len(self._frame)

    @property
    def frame_avg_fitness(self):
        return np.mean([t[1] for t in self._frame])

    @property
    def frame_std_fitness(self):
        return np.std([t[0] for t in self._frame])

    @property
    def is_frame_full(self):
        return self._frame_capacity == len(self._frame)

    def __init__(self, frame_capacity: int, k: float = 0.2) -> None:
        super().__init__()

        assert isinstance(frame_capacity, int) and frame_capacity >= 0

        self._frame_capacity = frame_capacity
        self._frame = []
        self._k = k

    def apply(self, X: Optional[list] = None, Y: Optional[np.ndarray] = None):

        if X is None or Y is None:
            return X, Y

        # If the frame is empty, the current solutions are stored in it
        frame_size = len(self._frame)
        frame_aval = self._frame_capacity-frame_size
        if frame_size < self._frame_capacity:

            idxs = np.argsort(Y)
            if len(idxs) > frame_aval:
                idxs = idxs[0:frame_aval]
            for i in idxs:
                self._frame.append((X[i], Y[i]))
            return X, Y
        
        else:

            # Define how to access items stored in the frame.
            frame_idxs = np.arange(0, frame_size)
            np.random.shuffle(frame_idxs)

            # Take the first k elements from the frame and add them to the population.
            k = int(frame_size*self._k)
            idxs = frame_idxs[0:k]
            new_X = X+[self._frame[t][0] for t in idxs]
            new_Y = np.concatenate([Y, [self._frame[t][1] for t in idxs]])

            # Select only the best items
            idxs = new_Y.argsort()
            if len(idxs) > len(X):
                idxs = idxs[0:len(X)]
            new_X = [new_X[i] for i in idxs]
            new_Y = new_Y[idxs]

            # Select the remaining (frame_size - k) elements
            frame_idxs = frame_idxs[k:]
            temp_X = X+[self._frame[t][0] for t in idxs]
            temp_Y = np.concatenate([Y, [self._frame[t][1] for t in idxs]])

            # Update the solutions contained in the frame
            best_idxs = temp_Y.argsort()
            if len(frame_idxs) > len(idxs):
                frame_idxs = frame_idxs[0:len(idxs)]

            for frame_index, best_index in zip(frame_idxs, best_idxs):
                self._frame[frame_index] = (
                    temp_X[best_index],
                    temp_Y[best_index]
                )

            return new_X, new_Y


__all__ = ["SolutionHandler", "InjectionSolutionHandler"]
