import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K

from sent_emb.algorithms.unkown import UnknownVector
from sent_emb.algorithms.glove_utility import GLOVE_DIM, GLOVE_FILE, read_file

model = None
encoder = None
default_maxlen = 80
default_target_dim = 100

def parse_sents(sents, unknown=UnknownVector(GLOVE_DIM), maxlen=default_maxlen):
    where = {}
    words = set()

    result = np.zeros((sents.shape[0], maxlen, GLOVE_DIM), dtype=np.float)

    for ids, sent in enumerate(sents):
        for idw, word in enumerate(sent):
            if word != '':
                if word not in where:
                    where[word] = []
                where[word].append((ids, idw))

    def process(word, vec, _):
        words.add(word)
        unknown.see(word, vec)
        if word in where:
            for (ids, idw) in where[word]:
                result[ids, idw] = vec

    read_file(GLOVE_FILE, process)

    result = result.reshape((result.shape[0], result.shape[1]*result.shape[2]))

    return result, result.shape[1]


def train_model(sents, unknown=UnknownVector(GLOVE_DIM), maxlen=default_maxlen, target_dim=default_target_dim):
    global model
    global encoder

    x_train, input_dim = parse_sents(sents, unknown, maxlen)

    model = Sequential()

    model.add(Dense(500, input_dim=input_dim, activation='relu'))
    encoder = Dense(target_dim, activation='relu')
    model.add(encoder)
    model.add(Dense(500, activation='relu'))
    model.add(Dense(input_dim, activation='linear'))

    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(x_train, x_train, epochs=10, batch_size=64, shuffle=True)


def get_single_embedding(sentences):
    global model
    global encoder

    get_encoding = K.function([model.layers[0].input, encoder.input, K.learning_phase()],
                                      [encoder.output])

    return get_encoding([sentences])[0]


def embeddings(sents, unknown=UnknownVector(GLOVE_DIM), maxlen=default_maxlen, target_dim=default_target_dim):
    '''
        sents: numpy array of sentences to compute embeddings
        unknown: handler of words not appearing in GloVe
        maxlen: maximal sentence length
        target_dim: the embedding dimension

        returns: numpy 2-D array of embeddings;
        length of single embedding is equal to target_dim
    '''

    x_test, dim = parse_sents(sents, unknown, maxlen)

    result = get_single_embedding(x_test)

    return result
