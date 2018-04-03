import numpy as np

from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.algorithms.glove_utility import GloVe


class WordVectorsMean(BaseAlgorithm):
    def get_resources(self, task):
        self.word_embeddings.get_resources(task)

    def fit(self, sents):
        pass

    def transform(self, sents):
        wordvec = self.word_embeddings.embeddings(sents)

        result = np.zeros((sents.shape[0], self.word_embeddings.get_dim()), dtype=np.float)
        count = np.zeros((sents.shape[0], 1))

        for idx, sent in enumerate(sents):
            for word in sent:
                result[idx] += wordvec[word]
                count[idx][0] += 1

        result /= count

        return result


class GloveMean(WordVectorsMean):
    def __init__(self, unknown=UnknownVector(300)):
        self.word_embeddings = GloVe(unknown)


