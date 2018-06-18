import numpy as np
import spacy

from sent_emb.evaluation.model import BaseAlgorithm
from sent_emb.algorithms.unknown import UnknownVector
from sent_emb.algorithms.glove_utility import GloVe
from sent_emb.algorithms.glove_embeddings_mean import GloveMean

# other tags we treat as irrelevant
POS_TAGS = ['ADJ', 'ADP', 'ADV', 'CONJ', 'NOUN', 'PRON', 'PROPN', 'VERB']
N_TAGS = len(POS_TAGS)


class GlovePosMean(BaseAlgorithm):
    def __init__(self, tag_groups=[(POS_TAGS, 1), (['NOUN', 'VERB'], 0.5)],
                 simple_mean_coeff=0.01):
        """
        :param tag_groups: list of pairs (tags list, weight)
            for each such group there is created seperate mean with given
            weight
        :param simple_mean_coeff: multiplier of simple mean of all words
            always set as non-zero, otherwise some sentences may receive
            embeddings that are all 0s.
        """
        self.tag_groups = tag_groups
        self.simple_mean_coeff = simple_mean_coeff

        unknown_str = 'average'
        unknown = UnknownVector(300)
        self.word_embeddings = GloVe(unknown)
        self.simple_mean = GloveMean(unknown_str)

        self.nlp = spacy.load('en')

    def get_resources(self, dataset):
        self.word_embeddings.get_resources(dataset)

    def fit(self, sents):
        self.simple_mean.fit(sents)

    def transform(self, sents):
        simple_embeddings = self.simple_mean.transform(sents)
        wordvec = self.word_embeddings.embeddings(sents)

        n_sents = len(sents)
        n_groups = len(self.tag_groups)
        glove_dim = self.word_embeddings.get_dim()

        result = np.zeros((n_sents, n_groups, glove_dim), dtype=np.float)
        count = np.zeros((n_sents, n_groups, 1))

        for idx, sent in enumerate(sents):
            sent_string = ' '.join(sent)
            doc = self.nlp(sent_string)
            for word in doc:
                pos = word.pos_
                if pos in POS_TAGS and word.text in wordvec:
                    for idg, group in enumerate(self.tag_groups):
                        tags, wei = group
                        if pos in tags:
                            result[idx][idg] += wei * wordvec[word.text]
                            count[idx][idg][0] += 1

        result /= np.maximum(count, [1])

        return np.concatenate([self.simple_mean_coeff * simple_embeddings,
                               result.reshape((n_sents, n_groups * glove_dim))], axis=1)
