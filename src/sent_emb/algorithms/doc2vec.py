import numpy as np
import gensim
import multiprocessing

from sent_emb.evaluation.model import BaseAlgorithm


class Doc2Vec(BaseAlgorithm):
    def __init__(self, vector_dim=300, min_count=1, epochs=10,
                 n_threads=multiprocessing.cpu_count()):
        self.vector_dim = vector_dim
        self.min_count = min_count
        self.epochs = epochs
        self.n_threads = n_threads
        self.model = None

    def get_resources(self, dataset):
        pass

    def fit(self, sents):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_dim,
                                                   min_count=self.min_count,
                                                   epochs=self.epochs,
                                                   workers=self.n_threads)

        train_corpus = [gensim.models.doc2vec.TaggedDocument(sent, [idx])
                        for idx, sent in enumerate(sents)]
        self.model.build_vocab(train_corpus)

        self.model.train(train_corpus,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)

    def transform(self, sents):
        if self.model is None:
            raise RuntimeError('Doc2Vec: transform() was called before fit()')

        n_sents = len(sents)
        emb_list = [self.model.infer_vector(sent) for sent in sents]
        result = np.concatenate(emb_list)

        assert result.shape[0] % n_sents == 0
        return result.reshape((n_sents, result.shape[0] // n_sents))
