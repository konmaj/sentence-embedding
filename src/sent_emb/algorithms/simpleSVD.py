import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import gzip
from urllib.request import urlretrieve

from sklearn.utils.extmath import randomized_svd

from sent_emb.algorithms.glove_utility import GloVe
from sent_emb.algorithms.path_utility import OTHER_RESOURCES_DIR
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.downloader.downloader import mkdir_if_not_exist
from sent_emb.evaluation.model import BaseAlgorithm


WORD_FREQUENCY_FILE = OTHER_RESOURCES_DIR.joinpath('word_frequency', 'all.num.gz')


def get_word_frequency():
    print('Checking for word frequency:')
    url = 'http://www.kilgarriff.co.uk/BNClists/all.num.gz'

    path = OTHER_RESOURCES_DIR
    mkdir_if_not_exist(path)
    word_frequency_path = path.joinpath('word_frequency')
    if mkdir_if_not_exist(word_frequency_path):
        print('Word frequency not found')
        urlretrieve(url, word_frequency_path.joinpath(Path(url).name))
    else:
        print('Found word frequency')


class Prob(ABC):
    @abstractmethod
    def transform(self, sents):
        """
            :param sents: sentences in the task on which Prob class should compute probability
            :return: self
        """
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

    def get_resources(self):
        pass


class ExternalProb(Prob):
    def __init__(self):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()

    def transform(self, sents):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()
        self.simple.transform(sents)
        for lines in gzip.open(WORD_FREQUENCY_FILE):
            sep = lines.split()
            word = sep[1].decode("utf-8")
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

    @staticmethod
    def get_resources():
        get_word_frequency()


class ExternalProbFocusUnknown(Prob):
    def __init__(self):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()

    def transform(self, sents):
        self.all = 0
        self.count = {}
        self.simple = SimpleProb()
        self.simple.transform(sents)
        for lines in gzip.open(WORD_FREQUENCY_FILE):
            sep = lines.split()
            word = sep[1].decode("utf-8")
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

    @staticmethod
    def get_resources():
        get_word_frequency()


class SimpleSVD(BaseAlgorithm):
    def __init__(self, word_embeddings=GloVe(UnknownVector(300)), param_a=0.001, prob=ExternalProb()):
        """
            param_a: parameter of scale for probabilities
            prob: object of class Prob, which provides probability of words in corpus
        """
        self.word_embeddings = word_embeddings
        self.param_a = param_a
        self.prob = prob
        self.unknown_prob_mult = 1

    def get_resources(self, task):
        self.word_embeddings.get_resources(task)
        self.prob.get_resources()

    def fit(self, sents):
        pass

    def transform(self, sents):
        self.prob.transform(sents)

        wordvec = self.word_embeddings.embeddings(sents)

        result = np.zeros((sents.shape[0], self.word_embeddings.get_dim()), dtype=np.float)
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
