import numpy as np
from pathlib import Path
from abc import abstractmethod

from sklearn.utils.extmath import randomized_svd

from sent_emb.algorithms.unkown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file
from sent_emb.downloader.downloader import get_word_frequency

WORD_FREQUENCY_FILE = Path('/', 'opt', 'resources', 'other', 'word_frequency', 'en.txt')


def embeddings_param(sents, unknown, param_a, prob, unknown_prob_mult):
    '''
        sents: numpy array of sentences to compute embeddings
        unkown: handler of words not appearing in GloVe
        param_a: scale for probabilities
        prob: set of estimated probabilities of words

        returns: numpy 2-D array of embeddings;
        length of single embedding is arbitrary, but have to be
        consistent across the whole result
    '''

    where = {}
    words = set()

    result = np.zeros((sents.shape[0], GLOVE_DIM), dtype=np.float)
    count = np.zeros((sents.shape[0], 1))

    for idx, sent in enumerate(sents):
        for word in sent:
            if word != '':
                if word not in where:
                    where[word] = []
                where[word].append(idx)

    def process(word, vec, _):
        words.add(word)
        unknown.see(word, vec)
        if word in where:
            for idx in where[word]:
                result[idx] += vec * param_a / (param_a + prob.get(word))
                count[idx][0] += 1

    read_file(GLOVE_FILE, process)

    for word in where:
        if word not in words:
            for idx in where[word]:
                result[idx] += unknown.get(word) * param_a / (param_a + unknown_prob_mult * prob.get(word))
                count[idx][0] += 1

    result /= count

    # Subtract first singular vector
    _, _, u = randomized_svd(result, n_components=1)
    for i in range(result.shape[0]):
        U = u * np.transpose(u)
        result[i] -= U.dot(result[i])

    return result


class Prob:
    @abstractmethod
    def get(self, word):
        pass


class SimpleProb(Prob):
    def __init__(self, sents):
        self.all = 0
        self.count = {}
        for sent in sents:
            for word in sent:
                if word not in self.count:
                    self.count[word] = 0
                    self.count[word] += 1
                self.all += 1

    def get(self, word):
        return self.count[word] / self.all


class ExternalProb(Prob):
    def __init__(self, sents):
        get_word_frequency()
        self.all = 0
        self.count = {}
        self.simple = SimpleProb(sents)
        for lines in open(WORD_FREQUENCY_FILE):
            word = lines.split(' ')[0]
            c = int(lines.split(' ')[1])
            self.all += c
            self.count[word] = c

    def get(self, word):
        if word in self.count:
            return self.count[word] / self.all
        else:
            return self.simple.get(word)


def embeddings(sents):
    return embeddings_param(sents, UnknownVector(GLOVE_DIM), 0.001, ExternalProb(sents), 1)
