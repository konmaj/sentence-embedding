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


class UnknownZero(Unknown):
    def __init__(self, dim):
        Unknown.__init__(self)
        self.v = np.zeros(dim, dtype=np.float)

    def see(self, _, _v):
        pass

    def get(self, _):
        return self.v


class UnknownVector(Unknown):
    def __init__(self, dim):
        Unknown.__init__(self)
        self.v = np.zeros(dim, dtype=np.float)
        self.count = 0

    def see(self, _, vec):
        self.v += vec
        self.count += 1

    def get(self, word):
        print(word)
        return self.v / self.count


class UnknownRandom(Unknown):
    def __init__(self, dim):
        Unknown.__init__(self)
        self.gave = {}
        self.dim = dim
        self.count = 0
        self.sum_len = 0

    def see(self, _, vec):
        self.count += 1
        self.sum_len += np.linalg.norm(vec)

    def get(self, word):
        if word not in self.gave:
            new_vec = np.zeros(self.dim)
            while np.linalg.norm(new_vec) == 0:
                for i in range(self.dim):
                    new_vec[i] = np.random.uniform(-1, 1)
            new_vec *= (self.sum_len / self.count) / np.linalg.norm(new_vec)
            self.gave[word] = new_vec
        return self.gave[word]


class NoUnknown(Unknown):
    def see(self, _, _vec):
        pass

    def get(self, _):
        assert False
