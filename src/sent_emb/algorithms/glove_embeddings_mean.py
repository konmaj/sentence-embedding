import numpy as np

from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file


class GloveMean(BaseAlgorithm):
    def __init__(self, unknown=UnknownVector(GLOVE_DIM)):
        self.unknown = unknown

    def fit(self, sents):
        pass

    def transform(self, sents):
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
            self.unknown.see(word, vec)
            if word in where:
                for idx in where[word]:
                    result[idx] += vec
                    count[idx][0] += 1

        read_file(GLOVE_FILE, process)

        for word in where:
            if word not in words:
                for idx in where[word]:
                    result[idx] += self.unknown.get(word)
                    count[idx][0] += 1

        result /= count

        return result
