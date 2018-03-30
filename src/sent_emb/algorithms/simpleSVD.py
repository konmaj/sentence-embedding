import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import gzip

from sklearn.utils.extmath import randomized_svd

from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file
from sent_emb.algorithms.path_utility import OTHER_RESOURCES_DIR
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.downloader.downloader import get_word_frequency
from sent_emb.evaluation.model import BaseAlgorithm


WORD_FREQUENCY_FILE = OTHER_RESOURCES_DIR.joinpath('word_frequency', 'all.num.gz')


class Prob(ABC):
    @abstractmethod
    def transform(self, sents):
        '''
            :param sents: sentences in the task on which Prob class should compute probability
            :return: self
        '''
        pass

    @abstractmethod
    def get(self, word):
        pass


class SimpleProb(Prob):
    def __init__(self):
        self.all = 0
        self.count = {}

    def transform(self, sents):
        self.all = 0
        self.count = {}
        for sent in sents:
            for word in sent:
                if word not in self.count:
                    self.count[word] = 0
                    self.count[word] += 1
                self.all += 1
        return self

    def get(self, word):
        if not self.count:
            raise RuntimeError('SimpleProb: get() was called before fit()')
        return self.count[word] / self.all


class ExternalProb(Prob):
    def __init__(self):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()

    def transform(self, sents):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()
        get_word_frequency()
        self.simple.transform(sents)
        for lines in gzip.open(WORD_FREQUENCY_FILE):
            sep = lines.split()
            word = sep[1]
            c = int(sep[0])
            self.all += c
            self.count[word] = c
        return self

    def get(self, word):
        if not self.count:
            raise RuntimeError('ExternalProb: get() was called before fit()')
        if word in self.count:
            return self.count[word] / self.all
        else:
            return self.simple.get(word)


class ExternalProbFocusUnknown(Prob):
    def __init__(self):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()

    def transform(self, sents):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()
        get_word_frequency()
        self.simple.transform(sents)
        for lines in gzip.open(WORD_FREQUENCY_FILE):
            sep = lines.split()
            word = sep[1]
            c = int(sep[0])
            self.all += c
            self.count[word] = c
        return self

    def get(self, word):
        if not self.count:
            raise RuntimeError('ExternalProb: get() was called before fit()')
        if word in self.count:
            return self.count[word] / self.all
        else:
            if self.simple.count[word] > 5:
                return self.simple.get(word)
            else:
                return self.simple.count[word] / self.all


class GloVe():
    def __init__(self, unknown):
        self.unknown = unknown

    def embeddings(self, words):
        result = {}

        def process(word, vec, _):
            self.unknown.see(word, vec)
            result[word] = vec

        read_file(GLOVE_FILE, process)

        for word in words:
            if word not in result:
                result[word] = self.unknown.get(word)

        return result


class SimpleSVD(BaseAlgorithm):
    def __init__(self, embeddings=GloVe(UnknownVector(GLOVE_DIM)), param_a=0.001, prob=ExternalProb()):
        '''
            unknown: handler of words not appearing in GloVe
            param_a: parameter of scale for probabilities
            prob: object of class Prob, which provides probability of words in corpus
        '''
        self.embeddings = embeddings
        self.param_a = param_a
        self.prob = prob
        self.unknown_prob_mult = 1

    def fit(self, sents):
        pass

    def transform(self, sents):
        self.prob.transform(sents)

        words = set()
        for sent in sents:
            for word in sent:
                words.add(word)

        wordvec = self.embeddings.embeddings(words)

        result = np.zeros((sents.shape[0], GLOVE_DIM), dtype=np.float)
        count = np.zeros((sents.shape[0], 1))

        for idx, sent in enumerate(sents):
            for word in sent:
                result[idx] += wordvec[word] * self.param_a\
                                   / (self.param_a + self.prob.get(word))
                count[idx][0] += 1

        result /= count

        # Subtract first singular vector
        _, _, u = randomized_svd(result, n_components=1)
        for i in range(result.shape[0]):
            u2 = u * np.transpose(u)
            result[i] -= u2.dot(result[i])

        return result
