from abc import ABC, abstractmethod

import numpy as np


class Unknown(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def see(self, word, vec):
        pass

    @abstractmethod
    def get(self, word):
        pass


class UnknownVector(Unknown):
    def __init__(self, dim):
        Unknown.__init__(self)
        self.v = np.zeros(dim, dtype=np.float)
        self.count = 0

    def see(self, _, vec):
        self.v += vec
        self.count += 1

    def get(self, _):
        return self.v / self.count


class UnknownRandom(Unknown):
    def __init__(self, dim):
        Unknown.__init__(self)
        self.gave = {}
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)

    def see(self, _, vec):
        self.min = np.minimum(self.min, vec)
        self.max = np.maximum(self.max, vec)

    def get(self, word):
        if word not in self.gave:
            new_vec = np.zeros(self.min.shape[0])
            for i in range(self.min.shape[0]):
                new_vec[i] = np.random.uniform(self.min[i], self.max[i])
            self.gave[word] = new_vec
        return self.gave[word]