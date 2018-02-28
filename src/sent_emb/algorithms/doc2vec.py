import numpy as np
import gensim
import multiprocessing

VECTOR_DIM = 300
MIN_COUNT = 1
EPOCHS = 10
N_THREADS = multiprocessing.cpu_count()

model = None


def train(sents):
    global model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=VECTOR_DIM,
                                          min_count=MIN_COUNT,
                                          epochs=EPOCHS,
                                          workers=N_THREADS)

    train_corpus = [gensim.models.doc2vec.TaggedDocument(sent, [idx])
                    for idx, sent in enumerate(sents)]
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)


def embeddings(sents):
    global model

    n_sents = sents.shape[0]
    emb_list = [model.infer_vector(sent) for sent in sents]
    result = np.concatenate(emb_list)

    return result.reshape((n_sents, result.shape[0] // n_sents))
