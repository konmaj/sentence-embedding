import numpy as np

from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.algorithms.unknown import UnknownVector, UnknownZero, UnknownRandom
from sent_emb.algorithms.glove_utility import GloVe


class WordVectorsMean(BaseAlgorithm):
    def get_resources(self, dataset):
        self.word_embeddings.get_resources(dataset)

    def fit(self, sents):
        pass

    def transform(self, sents):
        wordvec = self.word_embeddings.embeddings(sents)

        result = np.zeros((len(sents), self.word_embeddings.get_dim()), dtype=np.float)
        count = np.zeros((len(sents), 1))

        for idx, sent in enumerate(sents):
            for word in sent:
                result[idx] += wordvec[word]
                count[idx][0] += 1

        result /= count

        return result


class GloveMean(WordVectorsMean):
    def __init__(self, unknown_str='average'):
        if unknown_str == 'average':
            unknown = UnknownVector(300)
        elif unknown_str == 'zero':
            unknown = UnknownZero(300)
        elif unknown_str == 'random':
            unknown = UnknownRandom(300)
        else:
            print('Invalid unknown_str={} argument in GloveMean constructor'.format(unknown_str))
            assert False
        self.word_embeddings = GloVe(unknown)


class GloveMeanNormalized(WordVectorsMean):
    def __init__(self, unknown_str='average'):
        if unknown_str == 'average':
            unknown = UnknownVector(300)
        elif unknown_str == 'zero':
            unknown = UnknownZero(300)
        elif unknown_str == 'random':
            unknown = UnknownRandom(300)
        else:
            print('Invalid unknown_str={} argument in GloveMeanNormalized constructor'.format(unknown_str))
            assert False
        self.word_embeddings = GloVe(unknown, should_normalize=True)
