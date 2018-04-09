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
    def __init__(self, unknown=UnknownVector(300)):
        self.word_embeddings = GloVe(unknown)
        self.simple_mean = GloveMean(unknown)
        self.nlp = spacy.load('en')

    def get_resources(self, dataset):
        self.word_embeddings.get_resources(dataset)

    def fit(self, sents):
        self.simple_mean.fit(sents)

    def transform(self, sents):
        simple_embeddings = self.simple_mean.transform(sents)
        wordvec = self.word_embeddings.embeddings(sents)

        n_sents = len(sents)
        glove_dim = self.word_embeddings.get_dim()

        result = np.zeros((n_sents, N_TAGS, glove_dim), dtype=np.float)
        count = np.zeros((n_sents, N_TAGS, 1))

        for idx, sent in enumerate(sents):
            sent_string = ' '.join(sent)
            doc = self.nlp(sent_string)
            for word in doc:
                pos = word.pos_
                if pos in POS_TAGS and word.text in wordvec:
                    pos_id = POS_TAGS.index(pos)
                    result[idx][pos_id] += wordvec[word.text]
                    count[idx][pos_id][0] += 1

        result /= np.maximum(count, [1])

        return np.concatenate([0.01 * simple_embeddings,
                               result.reshape((n_sents, N_TAGS * glove_dim))], axis=1)
