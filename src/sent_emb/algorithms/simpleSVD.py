import numpy as np
from sklearn.utils.extmath import randomized_svd

from sent_emb.algorithms.unkown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file


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
                result[idx] += vec * param_a / (param_a + prob[word])
                count[idx][0] += 1

    read_file(GLOVE_FILE, process)

    for word in where:
        if word not in words:
            for idx in where[word]:
                result[idx] += unknown.get(word) * param_a / (param_a + unknown_prob_mult * prob[word])
                count[idx][0] += 1

    result /= count

    # Subtract first singular vector
    _, _, u = randomized_svd(result, n_components=1)
    for i in range(result.shape[0]):
        U = u * np.transpose(u)
        result[i] -= U.dot(result[i])

    return result


def simple_prob(sents):
    all = 0
    count = {}
    for sent in sents:
        for word in sent:
            if word not in count:
                count[word] = 0
            count[word] += 1
            all += 1
    for word, c in count.items():
        count[word] = c / all
    return count


def embeddings(sents):
    return embeddings_param(sents, UnknownVector(GLOVE_DIM), 0.001, simple_prob(sents), 1)
